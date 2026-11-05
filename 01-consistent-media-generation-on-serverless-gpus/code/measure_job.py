"""Attach observed lifecycle intervals to a downloaded metrics.json (also works for Endpoints)."""
import argparse
import json
from pathlib import Path
from common import emit, interval, write_json

def enrich(m, created, running, finished):
    result = dict(m)
    result["startup_s"] = interval(created, running)
    interval(running, finished)
    result["billable_s"] = interval(created, finished)
    result["billing_basis"] = "create_to_terminal_wall_clock_proxy_not_invoice"
    result["lifecycle"] = {"created_at":created, "running_at":running, "finished_at":finished}
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics", nargs="?", type=Path)
    p.add_argument("--created-at")
    p.add_argument("--running-at")
    p.add_argument("--finished-at")
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()
    if a.fake:
        print(json.dumps(enrich({"fake":True}, "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", "2026-01-01T00:02:00Z")))
    else:
        if not all((a.metrics, a.created_at, a.running_at, a.finished_at)):
            p.error("metrics and all three timezone-aware timestamps are required")
        result = enrich(json.loads(a.metrics.read_text()), a.created_at, a.running_at, a.finished_at)
        emit(a.metrics.parent, "lifecycle.recorded", billing_basis=result["billing_basis"])
        write_json(a.metrics, result)
