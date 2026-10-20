"""Run HF3FS metadata tests without importing the unrelated vLLM GPU stack."""
import argparse
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time


MODULE = 'vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_metadata_server.py'
TEST = 'tests/v1/kv_connector/unit/test_hf3fs_metadata_server.py'
IMPORT = 'vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_metadata_server'


def http_probe():
    import uvicorn
    module = importlib.import_module('hf3fs_metadata_server')
    service = module.Hf3fsMetadataServer()
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(service.app, log_level='warning'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    client = module.Hf3fsGlobalMetadataClient(f'http://127.0.0.1:{port}', max_retries=0)
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        client.initialize(0, 2)
        first = dict(client.allocate_pages_for_keys(0, [('cached', '')]))
        client.confirm_write_for_keys(0, list(first.items()))
        mixed = dict(client.allocate_pages_for_keys(0, [('cached', ''), ('new', '')]))
        assert mixed['new'] >= 0, mixed
        assert mixed['new'] != mixed['cached']
        client.confirm_write_for_keys(0, [('new', mixed['new'])])
        assert client.batch_key_exists(['cached', 'new']) == [True, True]
        assert client.get_key_locations(0, ['cached', 'new']) == [mixed['cached'], mixed['new']]
        # Leasing every complete key pins the pool: allocation must fail rather
        # than hand out a page a reader may still be loading from.
        held = client.acquire_keys(0, ['cached', 'new'])
        assert held == {'cached': mixed['cached'], 'new': mixed['new']}, held
        assert client.acquire_keys(0, ['missing']) == {'missing': None}
        assert dict(client.allocate_pages_for_keys(0, [('third', '')])) == {'third': -1}
        # Dropping one lease makes that key the LRU victim; the other survives.
        client.release_keys(0, ['cached'])
        third = dict(client.allocate_pages_for_keys(0, [('third', '')]))
        assert third['third'] == mixed['cached'], third
        assert client.batch_key_exists(['cached', 'new']) == [False, True]
        print(json.dumps({'mixed': mixed, 'third': third, 'evicted': 'cached',
                          'transport': 'real localhost HTTP', 'model_eval': False}))
    finally:
        client._session.close()
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--base', default='43caf5b3520b704fa766f00fc2b24da25c78e084')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.baseline:
        source = subprocess.check_output(['git', '-C', str(args.repo), 'show', f'{args.base}:{MODULE}'])
    else:
        source = (args.repo / MODULE).read_bytes()
    (args.out / 'hf3fs_metadata_server.py').write_bytes(source)
    tests = (args.repo / TEST).read_text(encoding='utf-8')
    assert tests.count(IMPORT) == 1
    (args.out / 'test_metadata.py').write_text(tests.replace(IMPORT, 'hf3fs_metadata_server'), encoding='utf-8')
    env = dict(os.environ, PYTHONPATH=str(args.out.resolve()))
    command = [sys.executable, '-m', 'pytest', '--noconftest', '-q', str(args.out / 'test_metadata.py')]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    (args.out / 'pytest.log').write_text(result.stdout, encoding='utf-8')
    print(result.stdout)
    report = {'baseline': args.baseline, 'pytest_exitcode': result.returncode, 'pytest_command': command,
              'isolation': 'Full source module and upstream tests; only import path rewritten; conftest disabled',
              'serving_model_evaluation': 'not run'}
    if not args.baseline and result.returncode == 0:
        sys.path.insert(0, str(args.out.resolve()))
        http_probe()
        report['http_exitcode'] = 0
    (args.out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
