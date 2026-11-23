"""Check an events.jsonl against the Module 2 trace contract and print its totals.

Usage: uv run python check_trace.py runs/02-multi-agent-media-workflows/<run_id>/events.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import ValidationError
from rich import print

from tracing import STOP_REASONS, Event, summarize

PAID_KINDS = ("model_request", "judge", "tool_call", "ticket")


def check(events: list[Event]) -> list[str]:
    """Return every contract violation in a trace; an empty list means it passes."""
    problems: list[str] = []
    seen: dict[str, Event] = {}
    for i, e in enumerate(events, 1):
        where = f"line {i} ({e.kind} {e.name})"
        if e.event_id in seen:
            problems.append(f"{where}: duplicate event_id")
        if e.parent_id and e.parent_id not in seen:
            problems.append(f"{where}: parent_id points to no earlier event")
        if e.cost_usd < 0:
            problems.append(f"{where}: negative cost")
        if e.cost_usd and e.kind not in PAID_KINDS:
            problems.append(f"{where}: {e.kind} events cost nothing")
        if e.kind == "ticket" and e.cost_usd and e.name not in ("completed", "failed"):
            problems.append(f"{where}: a video is charged once, on completed or failed")
        if e.cache_hit and (e.cost_usd or e.t_gen_s is not None):
            problems.append(f"{where}: a cache hit has cost_usd 0 and t_gen_s null")
        if e.possible_duplicate and e.kind != "ticket":
            problems.append(f"{where}: possible_duplicate is for ticket events")
        if e.stop_reason and e.stop_reason not in STOP_REASONS:
            problems.append(f"{where}: unknown stop_reason {e.stop_reason!r}")
        if e.kind == "node" and (e.tokens_in or e.tokens_out):
            problems.append(f"{where}: node events carry no tokens")
        if e.kind == "model_request" and (e.workload != "llm" or e.tokens_in is None):
            problems.append(f"{where}: model_request needs workload llm and tokens")
        if e.kind == "tool_call":
            parent = seen.get(e.parent_id or "")
            if not e.call_id:
                problems.append(f"{where}: tool_call without call_id")
            elif parent is None or e.call_id not in (parent.tool_call_ids or []):
                problems.append(f"{where}: call_id is not linked to the model request")
        if e.kind == "judge":
            if e.rubric_version is None or e.judge_model is None or e.reasons is None:
                problems.append(f"{where}: judge needs rubric_version, judge_model, reasons")
        elif e.verdict or e.rubric_version:
            problems.append(f"{where}: verdict and rubric_version are for judge events")
        seen[e.event_id] = e
    return problems


def main(path: Path) -> None:
    """Check a trace file and print its totals."""
    try:
        events = [Event.model_validate_json(line) for line in path.read_text().splitlines()]
    except ValidationError as exc:
        print(f"[red]Not a valid trace:[/red] {exc}")
        raise typer.Exit(1) from exc
    problems = check(events)
    for problem in problems:
        print(f"[red]FAIL[/red] {problem}")
    print(json.dumps(summarize(events), indent=2))
    print(f"{len(events)} events, {len(problems)} problems")
    raise typer.Exit(1 if problems else 0)


if __name__ == "__main__":
    typer.run(main)
