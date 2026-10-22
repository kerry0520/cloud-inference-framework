"""HTTP probe of the rebased HF3FS metadata server: leases protect pages from eviction.

Runs the real FastAPI app on a localhost socket (uvicorn) and drives it over
HTTP with the same request shapes the connector sends.  This checks the
response path that the orjson rebase touches (ORJSONResponse instead of the
removed _json_response helper), not just the in-process state class.
"""

import json
import socket
import sys
import threading
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hf3fs_metadata_server as srv  # noqa: E402


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main() -> int:
    result = {"steps": [], "pass": False}
    server = srv.Hf3fsMetadataServer(lease_ttl_s=120.0)
    port = free_port()
    base = f"http://127.0.0.1:{port}"

    try:
        import uvicorn
    except ImportError:
        result["limitations"] = "uvicorn missing; probe not run"
        (HERE / "http_probe.json").write_text(json.dumps(result, indent=2))
        return 3

    config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error")
    httpd = uvicorn.Server(config)
    thread = threading.Thread(target=httpd.run, daemon=True)
    thread.start()

    session = requests.Session()
    ready = False
    for _ in range(200):
        try:
            session.get(base + "/docs", timeout=0.5)
            ready = True
            break
        except Exception:
            time.sleep(0.05)
    if not ready:
        result["error"] = "server did not start"
        (HERE / "http_probe.json").write_text(json.dumps(result, indent=2))
        return 4

    def call(method: str, path: str, payload=None):
        resp = session.request(method, base + path, json=payload, timeout=10)
        body = None
        if resp.content:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
        result["steps"].append(
            {"path": path, "status": resp.status_code, "body": body}
        )
        return resp.status_code, body

    try:
        assert call("POST", "/rank/0/initialize", {"role": "worker", "num_pages": 2})[0] == 200
        code, body = call(
            "POST", "/keys/batch_allocate", {"rank": 0, "keys": [["a", ""], ["b", ""]]}
        )
        assert code == 200 and body["results"] == [["a", 0], ["b", 1]], body
        assert call(
            "POST", "/keys/confirm_write", {"rank": 0, "confirmations": [["a", 0], ["b", 1]]}
        )[0] == 204

        # Lease both keys, then ask for a third: the pool is full and leased.
        code, body = call("POST", "/keys/acquire", {"rank": 0, "keys": ["a", "b"]})
        assert code == 200 and body["locations"] == {"a": 0, "b": 1}, body
        code, body = call("POST", "/keys/batch_allocate", {"rank": 0, "keys": [["c", ""]]})
        assert code == 200 and body["results"] == [["c", -1]], body

        # Release one lease; the next allocation must evict the released key.
        assert call("POST", "/keys/release", {"rank": 0, "keys": ["a"]})[0] == 204
        code, body = call("POST", "/keys/batch_allocate", {"rank": 0, "keys": [["c", ""]]})
        assert code == 200 and body["results"] == [["c", 0]], body
        assert call("POST", "/keys/confirm_write", {"rank": 0, "confirmations": [["c", 0]]})[0] == 204

        code, body = call("POST", "/keys/acquire", {"rank": 0, "keys": ["a", "b", "c"]})
        assert code == 200 and body["locations"] == {"a": None, "b": 1, "c": 0}, body
        code, body = call("POST", "/keys/batch_exists", {"keys": ["a", "b", "c"]})
        assert code == 200 and body["exists"] == [False, True, True], body

        result["pass"] = True
        exitcode = 0
    except AssertionError as exc:
        result["error"] = str(exc)
        exitcode = 1
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        exitcode = 2

    httpd.should_exit = True
    thread.join(timeout=5)
    (HERE / "http_probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())
