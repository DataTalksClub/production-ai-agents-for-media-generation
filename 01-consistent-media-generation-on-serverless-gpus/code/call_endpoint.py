"""Call an OpenAI-compatible image Endpoint (such as the Sana cookbook template) from a CPU client."""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from common import (
    DEFAULT_CONFIG,
    asset_record,
    config,
    emit,
    interval,
    metrics,
    output_dir,
    positive,
    write_json,
)


def request(base, token, route, body=None, timeout=30):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        base.rstrip("/") + route,
        headers=headers,
        data=None if body is None else json.dumps(body).encode(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_ready(base, token, deadline_s=600):
    """Poll /v1/models until the Endpoint returns a non-empty model list (handles 502 during weight load)."""
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
        time.sleep(min(2, max(0, deadline - time.monotonic())))


class ImageGenerator:
    """Client helper adapted from the Build a Media Agent workshop."""

    def __init__(self, endpoint: str, token: str | None, model: str):
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.model = model

    def generate_image(
        self,
        prompt: str,
        *,
        seed: int = 7,
        size: str = "1024x1024",
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        negative_prompt: str = "",
        timeout: int = 300,
    ) -> bytes:
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "seed": seed,
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
        }
        payload = request(
            self.endpoint,
            self.token,
            "/v1/images/generations",
            request_data,
            timeout=timeout,
        )
        png = base64.b64decode(payload["data"][0]["b64_json"], validate=True)
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Expected a PNG response")
        return png


def run(
    config_path=DEFAULT_CONFIG,
    n=1,
    out=None,
    created_at=None,
    running_at=None,
):
    cfg = config(config_path)
    cfg["requested_model_revision"] = cfg.pop("model_revision")
    # The template Endpoint serves a model ID and does not expose its resolved Hub commit.
    cfg["model_revision"] = None
    cfg["image"] = os.environ.get("ENDPOINT_IMAGE")
    base, token = os.environ.get("BASE_URL"), os.environ.get("TOKEN")
    if not base or not token or not base.startswith("https://"):
        raise ValueError("Set HTTPS BASE_URL (without /v1) and TOKEN")
    if bool(created_at) != bool(running_at):
        raise ValueError("Pass both --created-at and --running-at, or neither")
    startup = interval(created_at, running_at) if created_at else None
    if running_at:
        interval(running_at, datetime.now(timezone.utc).isoformat())
    folder = output_dir(out)
    m = metrics(cfg, "endpoint")
    m["startup_s"] = startup
    emit(folder, "run.started")
    try:
        start = time.perf_counter()
        models = wait_ready(base, token)
        if cfg["model"] not in [x.get("id") for x in models["data"]]:
            raise ValueError("Endpoint model does not match generation.json")
        m["probe_wait_s"] = round(time.perf_counter() - start, 4)
        ready_at = datetime.now(timezone.utc).isoformat()
        m["ready_wait_s"] = interval(running_at, ready_at) if running_at else None
        m["ready_observed_at"] = ready_at
        emit(folder, "endpoint.ready", duration_s=m["ready_wait_s"])
        params = cfg["params"]
        generator = ImageGenerator(base, token, cfg["model"])
        for index in range(n):
            start = time.perf_counter()
            png = generator.generate_image(
                cfg["prompt"],
                seed=cfg["seed"],
                size=f'{params["width"]}x{params["height"]}',
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                negative_prompt=params["negative_prompt"],
            )
            m["latency_s"].append(round(time.perf_counter() - start, 4))
            path = folder / f"asset-{index:03d}.png"
            path.write_bytes(png)
            m["assets"].append(asset_record(path))
            emit(folder, "asset.saved", file=path.name)
        m["status"] = "completed"
    except Exception as exc:
        m["status"] = "failed"
        emit(folder, "run.failed", error_type=type(exc).__name__)
        raise
    finally:
        emit(folder, "run.finished", status=m["status"])
        write_json(folder / "metrics.json", m)
        print(folder / "metrics.json")
        print("Stop/delete the Endpoint in the console when finished.")
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--n", type=positive, default=1)
    p.add_argument("--out")
    a = p.parse_args()
    run(a.config, a.n, a.out)


if __name__ == "__main__":
    main()
