"""Shared helpers for configuration, run directories, metrics, traces, and cost calculations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("generation.json")
MODULE = "01-consistent-media-generation-on-serverless-gpus"


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def config(path=DEFAULT_CONFIG):
    value = json.loads(Path(path).read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", value["model_revision"]):
        raise ValueError("model_revision must be an immutable 40-char Hub commit")
    if not value["prompt"].strip():
        raise ValueError("prompt must not be empty")
    if not isinstance(value["seed"], int) or not 0 <= value["seed"] < 2**63:
        raise ValueError("seed must be a nonnegative integer below 2**63")
    p = value["params"]
    if not math.isfinite(p["guidance_scale"]) or p["guidance_scale"] < 0:
        raise ValueError("guidance_scale must be finite and nonnegative")
    for k in ("width", "height"):
        if not isinstance(p[k], int) or p[k] % 32 or not 256 <= p[k] <= 2048:
            raise ValueError(f"invalid {k}: use a multiple of 32 between 256 and 2048")
    if not 1 <= p["num_inference_steps"] <= 100:
        raise ValueError("steps must be between 1 and 100")
    return value


def output_dir(path):
    p = (
        Path(path)
        if path
        else Path("runs") / MODULE / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    p.mkdir(parents=True, exist_ok=False)
    return p


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def emit(out, event, **details):
    with (Path(out) / "trace.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": Path(out).name,
                    "event": event,
                    **details,
                },
                allow_nan=False,
            )
            + "\n"
        )


def metrics(cfg, mode):
    return {
        "schema_version": 1,
        "mode": mode,
        "startup_s": None,
        "ready_wait_s": None,
        "load_s": None,
        "latency_s": [],
        "billable_s": None,
        "billing_basis": "unavailable",
        "config": cfg,
        "env": {
            k: None
            for k in ("gpu", "preset", "driver", "cuda", "torch", "diffusers", "peak_vram_gb")
        },
        "assets": [],
        "status": "running",
    }


def asset_record(path):
    data = Path(path).read_bytes()
    return {
        "file": Path(path).name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def interval(start, finish):
    a, b = (datetime.fromisoformat(s.replace("Z", "+00:00")) for s in (start, finish))
    if a.tzinfo is None or b.tzinfo is None or b < a:
        raise ValueError("timestamps need timezones and chronological order")
    return (b - a).total_seconds()


def calculate_cost(
    startup_s: float,
    ready_s: float,
    latency_s: list[float],
    price_per_hour: float,
    counts: tuple[int, ...] | list[int] = (1, 10, 100),
    idle_s: float = 0.0,
    api_price: float | None = None,
) -> dict:
    """Compute marginal cost c, fixed cost F, projected cost per asset at N, and API breakeven."""
    values = [startup_s, ready_s, *latency_s, price_per_hour, idle_s]
    if not latency_s or any(
        v is None or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values
    ):
        raise ValueError("Need finite nonnegative startup, readiness/load and latency measurements")
    if not counts or any(n < 1 for n in counts):
        raise ValueError("Asset counts must be positive")
    if api_price is not None and (not math.isfinite(api_price) or api_price < 0):
        raise ValueError("Invalid API price")

    rate = price_per_hour / 3600.0
    t_gen = statistics.median(latency_s)
    marginal = rate * t_gen
    fixed = rate * (startup_s + ready_s + idle_s)
    breakeven = None
    if api_price is not None:
        breakeven = max(1, math.ceil(fixed / (api_price - marginal))) if api_price > marginal else 0

    return {
        "marginal_compute_usd": marginal,
        "fixed_run_usd": fixed,
        "projected_compute_usd_per_asset": {str(n): fixed / n + marginal for n in counts},
        "breakeven_assets": breakeven,
    }
