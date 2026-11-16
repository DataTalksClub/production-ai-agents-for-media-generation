---
video_url: ""
prev_url: 04-long-running-tools.md
next_url: ../03-batch-media-generation-at-scale/01-batch-jobs.md
---
# 2.5 Planner, Generator, Reviewer: the Judge Loop

In this lesson we write a versioned rubric from the brief, then build a planner, generator and reviewer loop that judges each keyframe against it and regenerates instead of accepting the first output. You're done when the judge returns `RETRY` on the fixture, the loop recovers to `PASS`, and every judge event in `events.jsonl` carries the rubric version.

## Roles: planner, generator, reviewer

## Rubric v1: criteria from the brief

Criteria reference brief fields, not one brief, so the same rubric judges the Module 3 records. How far to trust it is [4.1](../04-evaluation-observability-production/01-evaluating-media-quality.md), where you compare its verdicts with human labels on a golden set.

## Typed judge output: PASS, RETRY, WEAK

## When the judge reply doesn't parse

## Attempt caps

## Runtime gates

Deterministic checks run before the judge.

## Prove recovery with a fixture

## Metrics per run: judge tokens and cost per verdict

## Also try: one agent or several agents
