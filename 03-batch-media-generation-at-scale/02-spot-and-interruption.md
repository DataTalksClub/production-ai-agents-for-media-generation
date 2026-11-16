---
video_url: ""
prev_url: 01-batch-jobs.md
next_url: 03-optimizing-cost.md
---
# 3.2 Spot GPUs and Surviving Interruptions

In this lesson we run the batch on Spot GPUs, see why a Spot Job can be interrupted, save progress after each completed record, then interrupt a run and resume it without regenerating finished assets. You're done when the resumed run's `summary.json` shows the records done before the interruption, 0 of them regenerated, and the amount Spot saved vs on-demand.

## Spot GPUs and why Jobs get interrupted

## Preemption vs restart policy

## Spot pricing and savings

## Save progress after each record

## Resubmit with the same run_id

## Practice: interrupt and resume
