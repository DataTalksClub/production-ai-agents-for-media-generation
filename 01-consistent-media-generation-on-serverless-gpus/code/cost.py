"""HW1 sequential-run cost model; prices are supplied by the learner, never bundled."""
import argparse
import json
import math
import statistics
from pathlib import Path
from common import emit, nonnegative, positive

def calculate(m, price_per_hour, counts, idle_s=0, api_price=None):
    if m.get("fake"):
        raise ValueError("Fake runs are not GPU measurements")
    if m.get("status") != "completed":
        raise ValueError("Use a completed run for the sequential cost projection")
    ready = m["ready_wait_s"] if m["mode"] == "endpoint" else m["load_s"]
    values = [m["startup_s"], ready, *m["latency_s"], price_per_hour, idle_s]
    if not m["latency_s"] or any(v is None or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Need finite nonnegative startup, readiness/load and latency measurements")
    if not counts or any(n < 1 for n in counts):
        raise ValueError("Asset counts must be positive")
    if api_price is not None and (not math.isfinite(api_price) or api_price < 0):
        raise ValueError("Invalid API price")
    rate = price_per_hour / 3600
    marginal = rate * statistics.median(m["latency_s"])
    fixed = rate * (m["startup_s"] + ready + idle_s)
    breakeven = None
    if api_price is not None:
        breakeven = max(1, math.ceil(fixed / (api_price-marginal))) if api_price > marginal else 0
    result = {"marginal_compute_usd":marginal, "fixed_run_usd":fixed,
        "projected_compute_usd_per_asset":{str(n):fixed/n+marginal for n in counts},
        "breakeven_assets":breakeven, "billing_basis":"sequential_projection_not_invoice"}
    if m.get("billable_s") is not None:
        result["observed_wall_clock_compute_proxy_usd"] = rate * nonnegative(m["billable_s"])
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics", nargs="?", type=Path)
    p.add_argument("--price-per-hour", type=nonnegative)
    p.add_argument("--n", nargs="+", type=positive, default=[1,10,100])
    p.add_argument("--idle-seconds", type=nonnegative, default=0)
    p.add_argument("--api-price", type=nonnegative)
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()
    if a.fake:
        fixture = {"status":"completed", "mode":"job", "startup_s":60, "load_s":30, "latency_s":[10,12,11]}
        result = calculate(fixture, 1, [1,10,100], api_price=0.01)
        result["synthetic_arithmetic_only"] = True
    else:
        if a.metrics is None or a.price_per_hour is None:
            p.error("metrics and --price-per-hour are required")
        result = calculate(json.loads(a.metrics.read_text()), a.price_per_hour, a.n, a.idle_seconds, a.api_price)
        emit(a.metrics.parent, "cost.calculated", **result)
    print(json.dumps(result, indent=2, allow_nan=False))
