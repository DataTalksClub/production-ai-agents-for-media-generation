---
video_url: ""
prev_url: 01-fan-out.md
next_url: 03-optimizing-cost.md
---
# 3.2 Spot GPUs and Design for Interruption

In this lesson we run batch work on spot GPUs, checkpoint every item, and prove that a killed run resumes without regenerating finished items. You're done when the resume `summary.json` shows the items done before the kill, 0 of them regenerated, and the saved vs on-demand amount for the run.

## Spot Jobs

## Preemption vs restart policy

## Resubmit with the same run_id

## Spot pricing and savings

## Per-item checkpoints

## Chunk size

## Kill and resume proof

## Also try: chunk-size sweep

<!-- nav:start -->
[← Previous: 3.1 One to 1,000: Fan-out with Jobs](01-fan-out.md) · [Next: 3.3 Optimizing Cost per Asset: Cold Start, VRAM and Speed →](03-optimizing-cost.md)
<!-- nav:end -->
