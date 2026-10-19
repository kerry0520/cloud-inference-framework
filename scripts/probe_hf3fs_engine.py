"""Controlled real-engine save probe; local file backend, not a 3FS cluster."""
import argparse
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['CUDA_HOME'] = '/opt/cloud-inference-framework/cuda-toolkit'
os.environ['PATH'] = '/opt/cloud-inference-framework/.venv/bin:' + os.environ['CUDA_HOME'] + '/bin:' + os.environ['PATH']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    spec = importlib.util.spec_from_file_location('probe_metadata', args.source)
    metadata = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metadata
    spec.loader.exec_module(metadata)
    import uvicorn

    records = []

    class ControlledState(metadata.GlobalMetadataState):
        suppress_lookup = False

        def initialize_rank(self, rank, num_pages):
            super().initialize_rank(rank, 2)

        def batch_key_exists(self, keys):
            if self.suppress_lookup:
                return [False] * len(keys)
            return super().batch_key_exists(keys)

        def allocate_pages_for_keys(self, rank, keys):
            with self.global_lock:
                cached = [key for key, _ in keys if key in self.key_metadata and self.key_metadata[key].is_complete()]
                result = super().allocate_pages_for_keys(rank, keys)
                records.append({'keys': keys, 'complete_hits': cached, 'result': result})
                return result

    service = metadata.Hf3fsMetadataServer()
    service.state = ControlledState()
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(service.app, log_level='warning'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    llm = None
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        from vllm import LLM, SamplingParams
        import vllm
        from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs import hf3fs_connector as connector
        connector.Hf3fsMetadataClient = partial(metadata.Hf3fsGlobalMetadataClient, base_url=f'http://127.0.0.1:{port}')
        llm = LLM(model='/opt/cloud-inference-framework/models/Qwen3-0.6B',
                  dtype='bfloat16', max_model_len=128, max_num_seqs=1,
                  max_num_batched_tokens=128, gpu_memory_utilization=0.45,
                  enforce_eager=True, enable_prefix_caching=False, block_size=16,
                  async_scheduling=False,
                  kv_transfer_config={'kv_connector': 'HF3FSKVConnector', 'kv_role': 'kv_both',
                      'kv_connector_extra_config': {'hf3fs_storage_path': str(args.out / 'storage'),
                          'hf3fs_file_size': 64 * 1024 * 1024, 'hf3fs_client_numjobs': 1}})
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        prefix = list(range(20, 36))
        first = llm.generate([{'prompt_token_ids': prefix + [80]}], params, use_tqdm=False)
        deadline = time.monotonic() + 10
        while not service.state.key_metadata and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(service.state.key_metadata) == 1, records
        # Simulate the scheduler's earlier miss snapshot while the prefix save
        # has completed by the time this request reaches its save worker.
        service.state.suppress_lookup = True
        prompt = prefix + list(range(100, 117))
        second = llm.generate([{'prompt_token_ids': prompt}], params, use_tqdm=False)
        service.state.suppress_lookup = False
        mixed = [r for r in records if r['complete_hits'] and len(r['keys']) == 2]
        assert mixed, records
        failed = any(page < 0 for page in mixed[-1]['result'].values())
        assert failed == args.baseline, mixed
        warm_tokens = None
        if not args.baseline:
            assert len(service.state.key_metadata) == 2
            warm = llm.generate([{'prompt_token_ids': prompt}], params, use_tqdm=False)
            warm_tokens = list(warm[0].outputs[0].token_ids)
            assert warm_tokens == list(second[0].outputs[0].token_ids)
        report = {'baseline': args.baseline, 'vllm': vllm.__version__, 'backend': 'existing HF3FS file fallback',
                  'runner': 'MRV1 eager, TP1, async scheduling disabled, local APC disabled',
                  'intervention': 'metadata advertises two pages; second lookup forced to miss to simulate earlier scheduler snapshot',
                  'mixed_save_failed': failed, 'allocations': records,
                  'first_tokens': list(first[0].outputs[0].token_ids),
                  'cold_second_tokens': list(second[0].outputs[0].token_ids), 'warm_tokens': warm_tokens,
                  'limitations': 'One controlled run, installed vLLM engine with pinned main metadata module; no real 3FS, TP, concurrency stress, performance or quality evaluation',
                  'exitcode': 0}
        (args.out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


if __name__ == '__main__':
    main()
