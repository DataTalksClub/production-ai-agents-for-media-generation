"""Lesson 05: the projected cost of one Job run next to its observed cost.

    nebius ai job get <job_ID> --format json > job.json
    python code/reconcile_run.py --job job.json --metrics metrics_batch.json --rate 4.50 \
        --disk-gib 500 --project 1000

Projected: F + n × c from metrics_batch.json (model loads, median per asset) and the startup.
Observed: the run's billed duration × rate, plus the container disk. The docs don't say whether
the startup (create → container start) bills, so observed is a range: without it, with it.
Compare both with Billing → Usage; Usage is hourly, so it won't match to the second.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

DISK_USD_PER_GIB_MONTH = 0.071  # Network SSD, Compute pricing, read 2026-11-16
HOURS_PER_MONTH = 730

# Key names to look for in `nebius ai job get --format json`. First match wins; the script
# prints the path it used. Override with --created-at / --started-at / --finished-at.
TIME_KEYS = {
    "created": ("created_at", "create_time", "creation_time"),
    "started": ("started_at", "start_time", "running_at"),
    "finished": ("finished_at", "end_time", "completed_at", "finish_time"),
}


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def find_key(obj: Any, names: tuple[str, ...], path: str = "") -> tuple[str, str] | None:
    """Depth-first search for the first string value under one of `names`; returns (path, value)."""
    if isinstance(obj, dict):
        for name in names:
            if isinstance(obj.get(name), str) and obj[name]:
                return f"{path}.{name}".lstrip("."), obj[name]
        for key, value in obj.items():
            found = find_key(value, names, f"{path}.{key}")
            if found:
                return found
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found = find_key(value, names, f"{path}[{i}]")
            if found:
                return found
    return None


def job_times(job: dict, overrides: dict[str, str | None]) -> dict[str, datetime]:
    """created / started / finished from the Job JSON, with command-line overrides."""
    times = {}
    for kind, names in TIME_KEYS.items():
        if overrides.get(kind):
            times[kind] = parse_time(overrides[kind])
            typer.echo(f"{kind:>8}: {overrides[kind]} (from the command line)")
            continue
        found = find_key(job, names)
        if not found:
            raise typer.BadParameter(
                f"no {kind} time in the Job JSON (looked for {', '.join(names)}); "
                f"pass --{kind}-at"
            )
        times[kind] = parse_time(found[1])
        typer.echo(f"{kind:>8}: {found[1]} ({found[0]})")
    if not times["created"] <= times["started"] <= times["finished"]:
        raise typer.BadParameter("expected created <= started <= finished")
    return times


def projection(metrics: dict, t_start: float, rate_per_hour: float, n: int) -> dict[str, float]:
    """F + n × c from the run's own medians: the lesson's projection, warm-up left out."""
    steps = list(metrics["steps"].values())
    load_s = sum(s.get("load_s") or 0.0 for s in steps)
    # Per asset: the median over assets of each asset's time summed over all steps (as the notebook).
    ids = steps[0]["latency_s"].keys()
    per_asset_s = statistics.median(sum(s["latency_s"][i] for s in steps) for i in ids)
    warmup_s = sum(next(iter(s["latency_s"].values())) - s["median_s"] for s in steps)
    rate = rate_per_hour / 3600
    return {
        "t_start_s": t_start,
        "load_s": load_s,
        "per_asset_s": per_asset_s,
        "warmup_s": warmup_s,
        "fixed_usd": rate * (t_start + load_s),
        "marginal_usd": rate * per_asset_s,
        "seconds": t_start + load_s + n * per_asset_s,
        "usd": rate * (t_start + load_s + n * per_asset_s),
    }


def observed(seconds: float, rate_per_hour: float, disk_gib: float) -> dict[str, float]:
    """Compute and container-disk cost for a billed duration."""
    compute = rate_per_hour * seconds / 3600
    disk = disk_gib * DISK_USD_PER_GIB_MONTH / HOURS_PER_MONTH * seconds / 3600
    return {"seconds": seconds, "compute_usd": compute, "disk_usd": disk, "usd": compute + disk}


def main(
    metrics: Annotated[Path, typer.Option(help="metrics_batch.json from the run")],
    rate: Annotated[float, typer.Option(help="USD per GPU-hour the run paid (regular or spot)")],
    job: Annotated[Path | None, typer.Option(help="nebius ai job get <id> --format json")] = None,
    disk_gib: Annotated[float, typer.Option(help="container disk size, GiB")] = 250,
    project: Annotated[int, typer.Option(help="also project the cost of this many assets")] = 1000,
    usage_usd: Annotated[float | None, typer.Option(help="spend from Billing → Usage")] = None,
    created_at: Annotated[str | None, typer.Option(help="override, ISO 8601")] = None,
    started_at: Annotated[str | None, typer.Option(help="override, ISO 8601")] = None,
    finished_at: Annotated[str | None, typer.Option(help="override, ISO 8601")] = None,
) -> None:
    run = json.loads(metrics.read_text())
    n = run["n"]
    overrides = {"created": created_at, "started": started_at, "finished": finished_at}
    times = job_times(json.loads(job.read_text()) if job else {}, overrides)
    t_start = (times["started"] - times["created"]).total_seconds()
    container_s = (times["finished"] - times["started"]).total_seconds()
    to_worker_s = (parse_time(run["started_at"]) - times["started"]).total_seconds()

    proj = projection(run, t_start, rate, n)
    low = observed(container_s, rate, disk_gib)
    high = observed(container_s + t_start, rate, disk_gib)
    at_scale = projection(run, t_start, rate, project)

    typer.echo(f"\nRun: n = {n}, {run.get('platform')}, {run.get('billing')}, ${rate:.2f}/h")
    typer.echo(f"  startup (created → started)   {t_start:9.1f} s")
    typer.echo(f"  container start → worker start{to_worker_s:9.1f} s  (not in metrics_batch.json)")
    typer.echo(f"  worker runtime_s              {run['runtime_s']:9.1f} s")
    typer.echo(f"  started → finished            {container_s:9.1f} s")
    typer.echo(f"  model loads                   {proj['load_s']:9.1f} s")
    typer.echo(f"  per asset (median, all steps) {proj['per_asset_s']:9.1f} s")
    typer.echo(f"  warm-up left out              {proj['warmup_s']:9.1f} s")

    typer.echo(f"\nProjected, n = {n}: {proj['seconds']:.0f} s, ${proj['usd']:.2f}"
               f"  (F = ${proj['fixed_usd']:.2f}, c = ${proj['marginal_usd']:.4f})")
    for label, o in (("without startup", low), ("with startup", high)):
        typer.echo(f"Observed, {label:>15}: {o['seconds']:.0f} s, ${o['usd']:.2f}"
                   f"  (compute ${o['compute_usd']:.2f} + disk ${o['disk_usd']:.3f})")
    typer.echo(f"Gap, observed − projected: ${low['usd'] - proj['usd']:+.2f} to "
               f"${high['usd'] - proj['usd']:+.2f}")
    if usage_usd is not None:
        inside = low["usd"] <= usage_usd <= high["usd"]
        typer.echo(f"Billing → Usage: ${usage_usd:.2f}, observed range ${low['usd']:.2f}–"
                   f"${high['usd']:.2f}: {'inside' if inside else 'outside'}")
    typer.echo(f"\nProjected, N = {project}: {at_scale['seconds'] / 3600:.1f} h on one GPU, "
               f"${at_scale['usd']:.2f} compute, disk and warm-up left out")


if __name__ == "__main__":
    typer.run(main)
