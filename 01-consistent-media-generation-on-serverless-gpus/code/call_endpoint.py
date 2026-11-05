"""Call the Sana template's OpenAI-shaped image API without a GPU on the client."""
import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from common import DEFAULT_CONFIG, asset_record, config, emit, interval, metrics, output_dir, positive, write_json
from generate import FAKE_PNG

def request(base, token, route, body=None, timeout=30):
    headers = {"Authorization":f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base.rstrip("/") + route, headers=headers,
        data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def wait_ready(base, token, deadline_s=600):
    deadline = time.monotonic() + deadline_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Readiness deadline reached; inspect logs and stop the Endpoint")
        try:
            data = request(base, token, "/v1/models", timeout=min(30, remaining))
            if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
                return data
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(min(2, max(0, deadline-time.monotonic())))

def run(config_path=DEFAULT_CONFIG, n=1, out=None, fake=False, created_at=None, running_at=None):
    cfg = config(config_path)
    cfg["requested_model_revision"] = cfg.pop("model_revision")
    # The stock template exposes a model ID, not a revision. Do not pretend we pinned it.
    cfg["model_revision"] = None
    cfg["image"] = os.environ.get("ENDPOINT_IMAGE")
    base, token = os.environ.get("BASE_URL"), os.environ.get("TOKEN")
    if not fake and (not base or not token or not base.startswith("https://")):
        raise ValueError("Set HTTPS BASE_URL (without /v1) and TOKEN")
    if bool(created_at) != bool(running_at):
        raise ValueError("Pass both --created-at and --running-at, or neither")
    startup = interval(created_at, running_at) if created_at else None
    if running_at:
        interval(running_at, datetime.now(timezone.utc).isoformat())
    folder = output_dir(out)
    m = metrics(cfg, "endpoint", fake)
    m["startup_s"] = startup
    emit(folder, "run.started", fake=fake)
    try:
        start = time.perf_counter()
        if not fake:
            models = wait_ready(base, token)
            if cfg["model"] not in [x.get("id") for x in models["data"]]:
                raise ValueError("Endpoint model does not match generation.json")
            m["probe_wait_s"] = time.perf_counter()-start
            ready_at = datetime.now(timezone.utc).isoformat()
            m["ready_wait_s"] = interval(running_at, ready_at) if running_at else None
            m["ready_observed_at"] = ready_at
        emit(folder, "endpoint.ready", duration_s=m["ready_wait_s"])
        params = cfg["params"]
        body = {"model":cfg["model"], "prompt":cfg["prompt"], "seed":cfg["seed"], "n":1,
            "size":f'{params["width"]}x{params["height"]}', "response_format":"b64_json",
            **{k:params[k] for k in ("num_inference_steps", "guidance_scale", "negative_prompt")}}
        for index in range(n):
            if fake:
                png = FAKE_PNG
            else:
                start = time.perf_counter()
                payload = request(base, token, "/v1/images/generations", body, timeout=300)
                m["latency_s"].append(time.perf_counter()-start)
                png = base64.b64decode(payload["data"][0]["b64_json"], validate=True)
                if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("Expected a PNG response")
            path = folder / f"asset-{index:03d}.png"
            path.write_bytes(png)
            m["assets"].append(asset_record(path))
            emit(folder, "asset.saved", file=path.name, fake=fake)
        m["status"] = "completed"
    except Exception as exc:
        m["status"] = "failed"
        emit(folder, "run.failed", error_type=type(exc).__name__)
        raise
    finally:
        emit(folder, "run.finished", status=m["status"])
        write_json(folder / "metrics.json", m)
        print(folder / "metrics.json")
        print("Stop/delete the Endpoint in the console, including after an error.")
    return folder

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--n", type=positive, default=1)
    p.add_argument("--out")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--created-at", help="ISO timestamp with timezone, immediately before Create")
    p.add_argument("--running-at", help="ISO timestamp with timezone, first observed RUNNING")
    a = p.parse_args()
    run(a.config, a.n, a.out, a.fake, a.created_at, a.running_at)

if __name__ == "__main__":
    main()
