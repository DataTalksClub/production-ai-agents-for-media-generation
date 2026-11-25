# Capstone Project: Production Multimodal Agent

Build a production multimodal agent and share it publicly. You start it in Module 1 with the [Capstone Proposal Template](https://github.com/DataTalksClub/production-ai-agents-for-media-generation/blob/main/projects/proposal-template.md).

Your project needs:

* A multimodal agent that uses at least 2 GPU media models (image, video or audio)
* An async workload: the agent orchestrates long-running GPU jobs with job IDs, persisted state and resume
* A batch run: 100 records for image and audio assets, 10 records for video assets
* Evaluation: a golden set with human labels, the judge's verdicts compared with those labels, and a quality baseline
* Observability: a trace per record, plus success rate, p95 time per asset and cost per accepted asset
* An SLO with a pass/fail check and your go/no-go call
* It runs on Nebius Serverless AI, and the code is open source (MIT), OpenAI- and S3-compatible

Bring your own records: a CSV or JSONL file with an id column.

Capstones are peer-reviewed against an 8-area rubric: system, agent, media, cost, reliability, observability, reproducibility and presentation, each scored 0, 1 or 2. To pass, submit your project, review 3 peers, and share your work publicly.

There are two attempts. Submission and peer-review deadlines are 23:59:59 UTC:

* Attempt 1: submit by 15 Jan 2027, review peers by 22 Jan 2027
* Attempt 2: submit by 29 Jan 2027, review peers by 5 Feb 2027

Full requirements and the rubric: [projects/README.md](https://github.com/DataTalksClub/production-ai-agents-for-media-generation/blob/main/projects/README.md).
