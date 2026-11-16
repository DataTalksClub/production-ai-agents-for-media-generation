"""Lesson 04: time the Sana Endpoint with the Job's settings, and compare it with a Job run.

    python code/bench_endpoint.py --gpu H100 --n 10 --compare assets/example_runs/h100.json

Sends one request at a time, for the first n kids' illustration prompts from kids.csv, with
the worker's settings (models.SANA: 512x512, 20 steps, guidance 4.5, seed 7). Times each
request from the laptop: network, server and response included. Writes the times as JSON in
the shape of a metrics_batch.json step, so the two compare line by line.

Reads the Endpoint URL and token from code/.env (lesson 02), or asks for them.
"""

from __future__ import annotations

import csv
import json
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import typer

import common
from image import ImageGenerator
from models import SANA

KIDS_CSV = Path(__file__).resolve().parent.parent / "assets" / "kids.csv"


def prompts(n: int, path: Path = KIDS_CSV) -> dict[str, str]:
    """The first n kids' illustration prompts, by kid id: the same ones the Job draws."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:n]
    return {row["id"]: row["prompt"].strip() for row in rows}


def summarize(latency_s: dict[str, float]) -> dict:
    values = list(latency_s.values())
    return {
        "latency_s": {k: round(v, 3) for k, v in latency_s.items()},
        "first_s": round(values[0], 3),
        "median_s": round(statistics.median(values), 3),
        "median_after_first_s": round(statistics.median(values[1:]), 3) if len(values) > 1 else None,
    }


def compare(endpoint: dict, job_metrics: dict) -> dict:
    """Per-image gap between the Endpoint and the Job's illustration step."""
    job = job_metrics["steps"]["illustration"]
    gap = endpoint["median_s"] - job["median_s"]
    return {
        "endpoint_median_s": endpoint["median_s"],
        "job_median_s": job["median_s"],
        "job_gpu": job_metrics.get("gpu"),
        "gap_s": round(gap, 3),
        "ratio": round(endpoint["median_s"] / job["median_s"], 2),
    }


def main(
    gpu: Annotated[str, typer.Option(help="the Endpoint's GPU, as shown in the console")],
    n: Annotated[int, typer.Option(help="requests, one prompt per kid")] = 10,
    out: Annotated[Path, typer.Option(help="where to write the times")] = Path(
        "runs/bench/endpoint.json"
    ),
    compare_with: Annotated[
        Path | None, typer.Option("--compare", help="a metrics_batch.json to compare with")
    ] = None,
    kids: Annotated[Path, typer.Option(help="kids.csv")] = KIDS_CSV,
) -> None:
    base, token = common.setup_endpoint()
    common.wait_ready(base, token)
    gen = ImageGenerator(base, token, model=SANA.model)
    latency_s = {}
    with tempfile.TemporaryDirectory() as tmp:
        for kid_id, prompt in prompts(n, kids).items():
            gen.generate_image(
                prompt,
                out=Path(tmp) / f"{kid_id}.png",
                seed=SANA.seed,
                size=SANA.size,
                num_inference_steps=SANA.num_inference_steps,
                guidance_scale=SANA.guidance_scale,
                negative_prompt=SANA.negative_prompt,
            )
            latency_s[kid_id] = gen.last_s

    result = {
        "mode": "endpoint",
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": gpu,
        "endpoint_host": urlparse(base).hostname,  # no token, no full URL
        "model": SANA.model,
        "settings": {"size": SANA.size, "seed": SANA.seed, **SANA.params},
        "illustration": summarize(latency_s),
    }
    if compare_with:
        result["compare"] = compare(result["illustration"], json.loads(compare_with.read_text()))
    out.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(out, result)
    typer.echo(json.dumps({k: result[k] for k in ("gpu", "illustration")}, indent=2))
    if compare_with:
        typer.echo(json.dumps(result["compare"], indent=2))
    typer.echo(f"Saved {out}. Stop the Endpoint when you're done: it bills while RUNNING.")


if __name__ == "__main__":
    typer.run(main)
