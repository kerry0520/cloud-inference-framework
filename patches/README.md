# min-p 两阶段内核：可审查补丁

把项目里已经验证过的 min-p 两阶段内核，整理成对 vLLM 0.31.0（pinned 提交 `db9527a`）一处文件的最小改动。补丁是纯增量：**147 insertions、0 deletions**，不新增依赖、不改调用方签名、不改其他文件。

## 文件

| 文件 | 作用 |
| --- | --- |
| `0001-mrv2-min-p-two-stage.patch` | 统一 diff，目标 `vllm/v1/worker/gpu/sample/min_p.py`（index `b71ae6f..d0cb01e`） |
| `mrv2_min_p.py` | 打补丁后的完整文件，用于审阅和直接替换；与 apply 补丁的结果逐字节相同（sha256 `a4da6ec1…`） |

## 应用与回滚

```bash
# 在 pinned 上游 checkout（本项目用 /opt/cloud-inference-framework/vllm-src）
cd vllm-src
git apply --check /path/to/patches/0001-mrv2-min-p-two-stage.patch
git apply /path/to/patches/0001-mrv2-min-p-two-stage.patch

# 回滚
git checkout -- vllm/v1/worker/gpu/sample/min_p.py
```

或直接覆盖：

```bash
cp patches/mrv2_min_p.py vllm-src/vllm/v1/worker/gpu/sample/min_p.py
```

本项目没有修改 `/opt` 下的 `vllm-src/`（只读上游），也没有改 `.venv` 里安装的 wheel。验证时补丁只 apply 到临时副本；真实引擎验证由 worker extension 从路径加载 `patches/mrv2_min_p.py`，再包住 `SamplingStates.apply_min_p`，安装的 wheel 保持不变。

## 派发边界

并行路径只在同时满足下列条件时启用，否则走上游原内核（行为逐字节不变）：

- 行数（本次 logits 的 request 数）`1 <= rows <= 8`；
- 词表宽度 `vocab >= 32000`；
- dtype 是 `float16` / `bfloat16` / `float32`；
- `logits.stride(1) == 1`（行内连续；允许行步长大于词表宽）。

边界是两端都含的闭区间；`min_p == 0` 在两层都会提前返回，`states.py` 上层本来还会整批跳过内核。除上述条件外一律回退上游实现。

### 为什么只开这个小窗口

局部倍率（同目录 micro 证据，V151936：batch1 9.7×、batch2 8.8×、batch4 6.2×、batch8 3.8×）随 batch 增大而衰减；V4096 的 batch2–8 上两阶段反而比原内核慢约 12–14%（`0.88×`），V151936 的 batch32 基本无收益（`1.03×`）。两阶段只是把"每行一个 CTA 扫两遍"换成"每行每 4096 列一个 CTA 求部分最大值 + 一次归约"，仍扫两遍；好处来自并行度，代价是超大 batch 时 CTA 过碎、归约与暂存开销上升。所以窗口取得保守，没打算覆盖全场景。

## 兼容性与所有权

- 不改 `apply_min_p(logits, expanded_idx_mapping, min_p) -> None` 的签名、返回值和"原地修改"语义；调用点 `states.py:10` / `states.py:94` 无需改动。
- 不引入全局状态、模块级缓存或跨调用生命周期；`_apply_parallel` 每次调用新分配 `torch.empty((rows, num_parts), dtype=float32)` 暂存部分最大值，调用返回后即释放。
- 不改变数值契约：保留 `logits >= row_max + log(min_p)` 的元素，其余原位写 `-inf`；padding 列（`vocab_size` 之外）不读不写；`min_p == 0` 行不动。
- 内核名沿用候选阶段已验证的 `_min_p_partial_max` / `_min_p_apply_threshold`，上游原内核仍叫 `_min_p_kernel`，便于用 trace 按内核名核对走的是哪条路径。

## 已验证

- `scripts/verify_minp_patch.py`（`run_stage.sh patch`）：
  - 补丁 apply 到 pinned 文件的干净副本后，与 `mrv2_min_p.py` 逐字节相同；checkout 里安装的 wheel 文件与 pinned 源文件逐字节相同（`4cb5e2f4…`）。
  - 96 组用例（3 dtype × 10 vocab × 3 batch + 6 特例，storage 留 `vocab+7` 检查 padding）与上游逐字节相同；其中 21 组实际走并行路径、75 组走回退路径，启动哪个内核由计数器逐个用例记录，不靠推断。
  - 边界探针：`rows=8/vocab=32000` 与 `rows=1/32000` 走并行且一致；`rows=8/31999`、`rows=9/151936`、`rows=0` 走回退且一致；`stride(1)==2` 被谓词拒绝；`float64`/`float8` 被 dtype 守卫拒绝。
  - 结果目录 `results/patch-20261017T095423Z`（同结论的早期运行 `...094747Z` 也保留；后者 `apply/` 内混入的临时 `.git`/`__pycache__` 已删除，仅留 apply 后的 `min_p.py` 与 json）。
- 真实引擎（`run_stage.sh minp-patch-engine`，worker extension 从路径加载补丁模块）：4 次真实 MRV2 logits，`checked=4, mismatches=0`；内核计数 `_min_p_kernel=0, _min_p_partial_max=4, _min_p_apply_threshold=4`，即真实路径确实走了新内核。结果目录 `results/minp-patch-engine-20261017T095122Z`。

## 未验证 / 待补

- 其它 vLLM 版本（补丁按 pinned `db9527a` 生成，版本变动需重生成）、多卡/TP、异步调度、投机解码等路径未专门测试。
- 没有专门测 CUDA graph 捕获下的分配行为；已有 trace 证据显示采样器内核在 CUDA 图外（尾部 375–384 的 graph node id 为 0），但那条证据来自候选阶段，不是本补丁。
- `vocab` 接近 32000 时的收益只有约 2–3×，`vocab < 32000` 完全回退，这些场景不比上游差但没有收益。
- 整请求端到端影响仍只有本机 batch1/2/4 的小样本结论（约 1% 量级、batch1 不稳定），不能外推到服务器或其他 batch。
- 向上游提 issue/PR、或把补丁合入本项目以外的地方，都需要用户明确授权；本轮未做。

## 证据

批次记录与限制见 [第四阶段记录](../docs/第四阶段min-p框架补丁-20261017.md)；原始证据清单见 `results/fourth-stage-evidence-manifest.json`。失败与迭代目录（`patch-20261017T094720Z`、`patch-20261017T094738Z`、`patch-20261017T094747Z`（通过但目录脏，已清垃圾并重跑）、`minp-patch-engine-20261017T094858Z`、`...094949Z`、`...095037Z`）全部保留、不覆盖。
