"""Local file/trace helpers; no network or GPU imports."""
import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("generation.json")
MODULE = "01-consistent-media-generation-on-serverless-gpus"

def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number

def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number

def config(path):
    value = json.loads(Path(path).read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", value["model_revision"]):
        raise ValueError("model_revision must be an immutable Hub commit")
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
    p = Path(path) if path else Path("runs") / MODULE / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    p.mkdir(parents=True, exist_ok=False)
    return p

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def emit(out, event, **details):
    with (Path(out) / "trace.jsonl").open("a") as f:
        f.write(json.dumps({"schema_version":1, "timestamp":datetime.now(timezone.utc).isoformat(),
                            "run_id":Path(out).name, "event":event, **details}, allow_nan=False) + "\n")

def metrics(cfg, mode, fake):
    return {"schema_version":1, "mode":mode, "fake":fake, "startup_s":None,
            "ready_wait_s":None, "load_s":None, "latency_s":[], "billable_s":None,
            "billing_basis":"unavailable", "config":cfg,
            "env":{k:None for k in ("gpu","preset","driver","cuda","torch","diffusers")},
            "assets":[], "status":"running"}

def asset_record(path):
    return {"file":Path(path).name,"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()}

def interval(start, finish):
    a, b = (datetime.fromisoformat(s.replace("Z", "+00:00")) for s in (start, finish))
    if a.tzinfo is None or b.tzinfo is None or b < a:
        raise ValueError("timestamps need timezones and chronological order")
    return (b-a).total_seconds()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(config(DEFAULT_CONFIG), indent=2))
