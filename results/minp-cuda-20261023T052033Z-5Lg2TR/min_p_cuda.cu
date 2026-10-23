// SPDX-License-Identifier: Apache-2.0
// CUDA implementation of vLLM 0.31.0 min-p filtering for small decode batches.
//
// Contract (same as vllm/v1/worker/gpu/sample/min_p.py):
//   * logits is 2-D [num_rows, vocab], columns contiguous, row stride arbitrary;
//   * expanded_idx_mapping[row] selects the request, min_p[req] is a float32
//     probability in [0, 1]; rows with min_p == 0.0 are left untouched;
//   * threshold = max(row) + log(min_p), elements < threshold become -inf;
//   * bytes outside [0, vocab) (row padding) must not be touched.
//
// Why a second implementation: upstream runs one CTA per row and walks the
// vocabulary twice with 1024-wide blocks.  At batch <= 8 only a handful of CTAs
// exist, so a 0.6B model's 26 SMs mostly idle.  This kernel keeps one CTA per
// row but sweeps the row with 16-byte vector loads and a block-wide max
// reduction, then filters in a second pass that is served from L2.  The
// numerical path is deliberately identical to upstream (fp32 threshold,
// libdevice logf, fp16 compare) so logits stay byte-identical.
//
// NAN_MODE: mode 0 reduces with fmaxf (NaN ignored); mode 1 makes a NaN
// anywhere in the row produce a NaN threshold (NaN-propagating tl.max).  The
// pinned upstream was measured to behave like mode 0 (see the nan_probe output
// of scripts/minp_cuda.py), so mode 0 is the default and the in-path arm.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float to_float(const __half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(const float v) { return v; }
__device__ __forceinline__ float to_float(const __nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T neg_inf();
template <>
__device__ __forceinline__ __half neg_inf<__half>() {
  return __ushort_as_half(static_cast<unsigned short>(0xFC00u));
}
template <>
__device__ __forceinline__ float neg_inf<float>() {
  return __int_as_float(0xff800000);
}
template <>
__device__ __forceinline__ __nv_bfloat16 neg_inf<__nv_bfloat16>() {
  return __ushort_as_bfloat16(static_cast<unsigned short>(0xFF80u));
}

__device__ __forceinline__ float nan_value() {
  return __int_as_float(0x7fc00000);
}

struct RowMax {
  float value;
  int nan_seen;
};

__device__ __forceinline__ void account(RowMax& acc, float v) {
  if (isnan(v)) {
    acc.nan_seen = 1;
  } else if (v > acc.value) {
    acc.value = v;
  }
}

// Block-wide max; also folds a NaN-seen flag so both NAN_MODEs share one reduce.
__device__ __forceinline__ RowMax block_reduce(RowMax acc) {
  const unsigned mask = 0xffffffffu;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc.value = fmaxf(acc.value, __shfl_xor_sync(mask, acc.value, off));
    acc.nan_seen |= __shfl_xor_sync(mask, acc.nan_seen, off);
  }
  __shared__ float s_value[32];
  __shared__ int s_nan[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  if (lane == 0) {
    s_value[warp] = acc.value;
    s_nan[warp] = acc.nan_seen;
  }
  __syncthreads();
  __shared__ float s_result_value;
  __shared__ int s_result_nan;
  if (warp == 0) {
    RowMax block;
    block.value = (lane < warps) ? s_value[lane] : neg_inf<float>();
    block.nan_seen = (lane < warps) ? s_nan[lane] : 0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      block.value = fmaxf(block.value, __shfl_xor_sync(mask, block.value, off));
      block.nan_seen |= __shfl_xor_sync(mask, block.nan_seen, off);
    }
    if (lane == 0) {
      s_result_value = block.value;
      s_result_nan = block.nan_seen;
    }
  }
  __syncthreads();
  RowMax out;
  out.value = s_result_value;
  out.nan_seen = s_result_nan;
  return out;
}

template <typename T, int NAN_MODE>
__global__ void min_p_kernel(T* __restrict__ logits, int64_t stride,
                             const int32_t* __restrict__ mapping,
                             const float* __restrict__ min_p, int vocab) {
  const int row = blockIdx.x;
  const float p = min_p[mapping[row]];
  if (p == 0.0f) {
    return;
  }
  T* base = logits + static_cast<int64_t>(row) * stride;
  constexpr int kVec = 16 / sizeof(T);
  const int tid = threadIdx.x;
  const int step = blockDim.x * kVec;
  const bool aligned16 =
      (reinterpret_cast<uintptr_t>(base) & static_cast<uintptr_t>(15)) == 0;

  RowMax acc;
  acc.value = neg_inf<float>();
  acc.nan_seen = 0;
  if (aligned16) {
    for (int i = tid * kVec; i < vocab; i += step) {
      if (i + kVec <= vocab) {
        uint4 raw = *reinterpret_cast<const uint4*>(base + i);
        const T* vals = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          account(acc, to_float(vals[k]));
        }
      } else {
        for (int j = i; j < vocab; ++j) {
          account(acc, to_float(base[j]));
        }
      }
    }
  } else {
    for (int i = tid; i < vocab; i += blockDim.x) {
      account(acc, to_float(base[i]));
    }
  }
  const RowMax reduced = block_reduce(acc);
  float row_max = reduced.value;
  if (NAN_MODE != 0 && reduced.nan_seen) {
    row_max = nan_value();
  }
  const float threshold = row_max + logf(p);

  if (aligned16) {
    for (int i = tid * kVec; i < vocab; i += step) {
      if (i + kVec <= vocab) {
        uint4 raw = *reinterpret_cast<const uint4*>(base + i);
        T* vals = reinterpret_cast<T*>(&raw);
        bool changed = false;
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          if (to_float(vals[k]) < threshold) {
            vals[k] = neg_inf<T>();
            changed = true;
          }
        }
        if (changed) {
          *reinterpret_cast<uint4*>(base + i) = raw;
        }
      } else {
        for (int j = i; j < vocab; ++j) {
          if (to_float(base[j]) < threshold) {
            base[j] = neg_inf<T>();
          }
        }
      }
    }
  } else {
    for (int i = tid; i < vocab; i += blockDim.x) {
      if (to_float(base[i]) < threshold) {
        base[i] = neg_inf<T>();
      }
    }
  }
}

template <typename T>
void launch(torch::Tensor& logits, const torch::Tensor& mapping,
            const torch::Tensor& min_p, int nan_mode) {
  const int rows = static_cast<int>(logits.size(0));
  const int vocab = static_cast<int>(logits.size(1));
  const int64_t stride = logits.stride(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  T* data = reinterpret_cast<T*>(logits.data_ptr());
  const int32_t* map = mapping.data_ptr<int32_t>();
  const float* p = min_p.data_ptr<float>();
  if (nan_mode == 0) {
    min_p_kernel<T, 0><<<rows, kThreads, 0, stream>>>(data, stride, map, p, vocab);
  } else {
    min_p_kernel<T, 1><<<rows, kThreads, 0, stream>>>(data, stride, map, p, vocab);
  }
}

// Applies min-p in place.  nan_mode selects the NaN semantics (see header).
void min_p_apply(torch::Tensor logits, torch::Tensor expanded_idx_mapping,
                 torch::Tensor min_p, int64_t nan_mode) {
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D");
  TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
  TORCH_CHECK(logits.is_contiguous() || logits.stride(1) == 1,
              "min-p ABI requires contiguous columns");
  TORCH_CHECK(expanded_idx_mapping.scalar_type() == torch::kInt32,
              "expanded_idx_mapping must be int32");
  TORCH_CHECK(min_p.scalar_type() == torch::kFloat32, "min_p must be float32");
  TORCH_CHECK(expanded_idx_mapping.is_contiguous() && min_p.is_contiguous(),
              "mapping and min_p must be contiguous");
  const int rows = static_cast<int>(logits.size(0));
  if (rows == 0 || logits.size(1) == 0) {
    return;
  }
  TORCH_CHECK(expanded_idx_mapping.size(0) >= rows,
              "mapping shorter than the number of rows");
  switch (logits.scalar_type()) {
    case torch::kFloat16:
      launch<__half>(logits, expanded_idx_mapping, min_p,
                     static_cast<int>(nan_mode));
      break;
    case torch::kBFloat16:
      launch<__nv_bfloat16>(logits, expanded_idx_mapping, min_p,
                            static_cast<int>(nan_mode));
      break;
    case torch::kFloat32:
      launch<float>(logits, expanded_idx_mapping, min_p,
                    static_cast<int>(nan_mode));
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype for the CUDA min-p kernel");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("min_p_apply", &min_p_apply, "in-place min-p filtering",
        pybind11::arg("logits"), pybind11::arg("expanded_idx_mapping"),
        pybind11::arg("min_p"), pybind11::arg("nan_mode") = 0);
}
