# Capstone Project: Production Multimodal Agent

Build a production multimodal agent and share it publicly. The capstone starts in Module 1 and has its first submission deadline on **15 Jan 2027**.

## Requirements

* A production multimodal agent that uses **at least 2 GPU media models** (any combination of image, video, audio)
* An **async workload**: the agent orchestrates long-running GPU jobs (job IDs, state persistence, resume — not blocking calls)
* **Batch run**: 100 records for image and audio assets, 10 records for video assets
* **Evaluation**: a golden set with human labels, the judge's verdicts compared with those labels, and a quality baseline for your run
* **Observability**: a trace per record that answers "why this cost / why this long", plus success rate, p95 time per asset and cost per accepted asset for the run
* **SLO**: per-run targets for success rate, p95 time per asset and cost per accepted asset, a pass/fail check and your go/no-go call
* Run it on **Nebius Serverless AI** (Jobs, Endpoints) as the reference execution layer
* Code is open-source (MIT), OpenAI- and S3-compatible

## Records

Bring your own records: a CSV or JSONL file with an id column. Records go in, generated media comes out.

## Review

* Capstones are **peer-reviewed** against an 8-area rubric (covering the system, agent and media layers plus cost, reliability, observability, reproducibility and presentation)
* To pass, submit your project, review peers, and share your work publicly

## Timeline

* Start: Module 1 (two weeks, Mon 16 – Sun 29 Nov 2026) — pick your records, get the first GPU generation working, and fill in the [Capstone Proposal Template](proposal-template.md)
* Modules 2–4 (weekly): Mon 30 Nov – Sun 20 Dec 2026
* Teaching ends: Sun 20 Dec 2026
* Attempt 1: submit by 15 Jan 2027; complete peer reviews by 22 Jan 2027
* Attempt 2: submit by 29 Jan 2027; complete final peer reviews by 5 Feb 2027
* Submission and peer-review deadlines are 23:59:59 UTC on the dates above.

## Peer-review rubric

Peers score each area 0, 1 or 2.

**System**

* 0: There is no working end-to-end system that takes records in and produces media out on Nebius Serverless AI.
* 1: A working pipeline runs on Nebius Serverless AI, but the batch is incomplete (fewer records than required, or only part of the path works).
* 2: A production multimodal system runs on Nebius Serverless AI: records in, media out, and a batch of 100 image and audio records, or 10 video records.

**Agent**

* 0: There is no agent, or only a single blocking script with no tools.
* 1: There is an agent, but it has fewer than two tools, or it does not orchestrate GPU work.
* 2: An agent with at least two tools orchestrates long-running GPU jobs, with job IDs and persisted state, not only blocking calls.

**Media**

* 0: No GPU media models are used, or only one model is used.
* 1: Two or more GPU media models are used, but generated assets are missing or are not stored in object storage.
* 2: At least two GPU media models (image, video or audio) are used, and assets are stored in object storage.

**Cost**

* 0: There is no cost report.
* 1: A cost figure is present, but it has no per-asset or per-request breakdown.
* 2: A cost report explains the cost per asset or per request.

**Reliability**

* 0: Failures and interruptions are not handled.
* 1: There is some retry or error handling, but a killed run cannot resume without redoing finished work.
* 2: A killed run resumes without redoing finished work.

**Observability**

* 0: There are no traces, metrics or logs.
* 1: Some logs or metrics exist, but there is no per-request trace.
* 2: Each request has a trace that answers why this cost and why this long, plus metrics, logs and a dashboard.

**Reproducibility**

* 0: The run cannot be repeated from the repository.
* 1: The code is public, but versions, prompts or inputs are not pinned enough to replay the run.
* 2: The run is reproducible from public MIT code, with pinned versions and inputs, and is OpenAI- and S3-compatible.

**Presentation**

* 0: The project is not shared publicly, or there is no way to understand what was built.
* 1: A public repository or post exists, but it does not explain how to run the project or what it produced.
* 2: The work is shared publicly, with a clear write-up of what was built, how to run it, and what the results were.
