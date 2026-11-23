# Worked trace

`events.jsonl` is a hand-written trace of one full run: plan, a keyframe that gets `RETRY`,
a second keyframe that gets `PASS`, one video, checks. `expected.json` holds the totals the
aggregation rules give for it. `check_trace.py` must pass on it:

```bash
cd 02-multi-agent-media-workflows/code
uv run python check_trace.py ../assets/worked-trace/events.jsonl
```

All numbers are [illustrative]: invented to show how the totals add up, not measured.
