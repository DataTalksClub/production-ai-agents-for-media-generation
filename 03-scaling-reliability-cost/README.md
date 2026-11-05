# Module 3: Scaling, Reliability and Cost

Welcome to Module 3 of Production AI Agents for Generative Media! This week we make it scale: run the same generation for every record over parallel spot GPUs, survive a killed run without paying twice, cut the cost per asset, and stop the run when it breaks a budget or a quality bar.

> [!NOTE]
> This module builds on Module 1 and Module 2: Jobs, the Object Storage bucket and the cost terms from [1.4](../01-generative-media-workloads#14-serving-vs-batch-inference-the-same-generation-for-different-use-cases) and [1.5](../01-generative-media-workloads#15-measuring-startup-load-and-generation-costs), the idempotency key from [2.2](../02-agents-that-use-gpus#22-tools-and-structured-outputs), tickets from [2.4](../02-agents-that-use-gpus#24-long-running-tools-submit-poll-resume) and the judge from [2.5](../02-agents-that-use-gpus#25-planner-generator-reviewer-the-judge-loop).

> [!IMPORTANT]
> Parallel Jobs need quota. Serverless AI Jobs count toward Compute quotas: one VM, one disk and one allocation per running Job, plus GPUs for regular VMs. A new account starts with 2 regular and 2 preemptible VMs per region. Homework 3 runs 10 Jobs in parallel, so request an increase to at least 10 preemptible VMs, 10 disks and the matching allocations before the homework week. How to check and increase quotas: [3.1](#31-one-to-1000-fan-out-with-jobs).

> [!WARNING]
> Cancel your Jobs (`nebius ai job cancel <job_ID>`) and stop your Endpoints when you finish the module to avoid charges. Set a `--timeout` on every Job and a `budget_usd` on every run. Run the lessons on 10 to 100 records; all 200 records is the homework. Keep video to a 10-record sample.

* Video Link (Coming Soon)
* Slides (Coming Soon)

## Learning Objectives

By the end of this module, you will be able to:

* Predict the wall time and cost of a batch run before you launch it, then fan out records over parallel Jobs and fan the results back in
* Run batch work on spot (preemptible) GPUs, checkpoint every item, and prove that a killed run resumes without regenerating finished items
* Choose a chunk size from your own fixed cost, marginal cost and preemption count
* Lower the cost per asset with three levers: fixed cost (cold start), generation time, and the GPU you need (VRAM)
* Break the pipeline on purpose, including budget and quality thresholds, and fix each failure with a named pattern
* Optional: fine-tune an image model on a new style as a Job and use it in your pipeline

Default path for the whole module: [Sana](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-sana/README.md) on L40S for images, [Kokoro-82M](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-kokoro-82m/README.md) for voice and [Wan2.2-I2V-A14B](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-wan22-i2v-a14b/README.md) on H100 for the video sample. Records: [`examples/mediafactory/data/kids.csv`](../examples/mediafactory/data/kids.csv), 200 records (`id`, `name`, `age`, `wishlist`), and its first 10 rows in [`kids_10.csv`](../examples/mediafactory/data/kids_10.csv). Lessons that use 100 records take the first 100 rows of `kids.csv`. Every run writes `events.jsonl`, `metrics.json` and `summary.json` to its run folder in the bucket.

## 3.1 One to 1,000: Fan-out with Jobs

* Do the math before you launch, with your Module 1 numbers:
  * Wall time ≈ `t_start + t_ready + (N / K) × t_gen` for `N` records over `K` parallel Jobs
  * Total run cost: `K × F + N × c`. Total run cost per asset: `K × F / N + c`
  * Assumes each Job pays its own startup and model load, assets inside a Job run one after another, and no idle gaps
* Endpoint concurrency vs Job fan-out:
  * Endpoint: one warm model, many requests in flight. Bills while RUNNING, idle or not.
  * Jobs: `K` GPUs in parallel, each pays `F`, each stops on completion. Fits records in → assets out.
* Manifest → chunks → `K` Jobs → per-chunk results → fan-in `summary.json`
* Quotas set your `K`:
  * Every running Job is one VM, one disk and IP allocations, plus GPUs for a regular VM. All of them count toward Compute quotas, per region. Over quota, a Job fails with `QuotaExceeded` ([Serverless AI quotas](https://docs.nebius.com/serverless/pricing-quotas)).
  * Quotas that bound `K`: Regular VMs or Preemptible VMs, GPUs for your platform (regular VMs only), Disks, Disk capacity, Allocations
  * A new account starts on the default tier: 2 regular and 2 preemptible VMs, 4 disks per region. Nebius may move an account to the full tier (10 VMs) based on usage history ([Compute quotas](https://docs.nebius.com/compute/resources/quotas-limits)).
  * Check yours: web console → Administration → Limits → Quotas, or `nebius quotas quota-allowance list --parent-id <project_ID>` ([Quotas](https://docs.nebius.com/overview/quotas))
  * Increase: Administration → Limits → Quotas → Change quota. Only users with the `admin` role can request it; track the request in the support center.
  * Request it close to when you need it: a quota that stays well below its limit for a long time can be reduced automatically, after an email and a grace period, never below default
  * Size `K` from your quota: `K ≤ min(VMs, disks, GPUs ÷ GPUs per Job)` minus what is already running, such as an Endpoint
* Backpressure: a client-side semaphore for Endpoint calls, backoff on LLM `429`s, and a launcher that queues chunks when `K` Jobs are already running
* Heterogeneous pools: one group of Jobs per model and GPU. L40S for images and voice, H100 for video.
* Practice: 100 records over 2 Jobs with Sana (the default quota on a new account), outputs to the bucket, one `summary.json` per run

Done when: you wrote down your quota values for VMs, disks and GPUs, and `summary.json` in your bucket shows 100 records over 2 Jobs, with per-chunk counts, total cost and cost per asset, next to the wall time and cost you predicted.

## 3.2 Spot GPUs and Design for Interruption

* Spot (preemptible) Jobs: `nebius ai job create --preemptible --follows-spot-price`. The platform can stop them at any time. See [`nebius ai job create`](https://docs.nebius.com/cli/reference/ai/job/create).
* A preempted Job ends with an error status and is not restarted. `--restart-policy on-failure` restarts a container that exits with an error ([automatic job restarts](https://docs.nebius.com/serverless/jobs/auto-restarts)); it does not cover preemption.
* So the launcher can't tell a preemption from a failed container by status alone. It treats both the same:
  * Resubmit the chunk with the same `run_id`. The ledger skips items that are done, so only the gaps run again.
  * Cap resubmits per chunk (`max_chunk_attempts`). A real bug stops there instead of looping on the bill.
  * Log the Job's status and message in `events.jsonl`, so you can count preemptions afterwards
* The spot price is dynamic, and its maximum is one cent below on-demand. Take the current rate from [Compute pricing](https://docs.nebius.com/compute/resources/pricing) and write down the date. Report "saved vs on-demand" per run.
* Design for interruption: a preempted Job loses only its in-flight item, if every item is checkpointed
* Per-item checkpoints:
  * Output key from the record `id` and the idempotency key from 2.2. Skip the item if its ledger entry says `done`.
  * Ledger in the run folder: `items/<id>.json` with status, attempts, generation time and cost
  * Commit order: write the asset, then the ledger entry. A half-written asset never counts as done.
  * Dead-letter: items that fail `max_attempts` times go to `dead_letter.jsonl` and don't block the run
  * Re-run with the same `run_id`: only the gaps are generated
* Chunk size:
  * Small chunks: more Jobs, more fixed cost `F`
  * Large chunks: a preemption costs more wall time, and without checkpoints, more regenerated items
  * Total run cost per asset with chunks of `n` records: `F / n + c`, plus `P × (F + c) / N` for `P` preemptions in the run, with per-item checkpoints. Same assumptions as 3.1.
* Kill → resume proof: start 100 records on 2 spot Jobs, cancel them at about 50% with `nebius ai job cancel`, resume with the same `run_id`
* Also try: a chunk-size sweep (5, 10, 20 records per chunk on 40 records) and compare cost per asset and wall time

Done when: the resume `summary.json` shows the items done before the kill, 0 of them regenerated, and the saved vs on-demand amount for the run.

## 3.3 Optimizing Cost per Asset: Cold Start, VRAM and Speed

* Three levers in the Module 1 formula, total run cost per asset `F / N + c`:
  * `F`: startup and model load. Paid once per Job, so 3.1 and 3.2 multiply it.
  * `c = rate × t_gen`: generation time
  * `rate`: the GPU you need, set by VRAM
* Lower `F`, cold start:
  * The production image from [1.3](../01-generative-media-workloads#13-containers-for-generative-ai): built once, pushed, pinned
  * Weights from your bucket instead of a Hugging Face download in every Job: sync them once, mount read-only with `--volume s3://<bucket>:/models:ro` ([`nebius ai job create`](https://docs.nebius.com/cli/reference/ai/job/create)), pinned revision
  * Measure cold start with and without cached weights: `startup_s` and `load_s`
* Lower `t_gen`, speed ([Diffusers: Accelerate inference](https://huggingface.co/docs/diffusers/optimization/fp16)):
  * bf16 instead of fp32, attention backend, batch size, fewer steps
  * `torch.compile`: the first call is slow, later calls are faster. It raises `F` and lowers `c`, so it pays off only above a break-even `N = ΔF / Δc`.
  * Fewer steps change the asset: check it with the 2.5 judge before you keep it
* Lower `rate`, fit VRAM to a smaller GPU ([Diffusers: Reduce memory usage](https://huggingface.co/docs/diffusers/optimization/memory)):
  * Model or group offloading, VAE tiling, lower-precision weights
  * Each trades speed for memory: `t_gen` goes up when `rate` goes down. Compare total run cost per asset, not VRAM alone.
  * OOM debugging: read peak VRAM with `torch.cuda.max_memory_allocated()` and add it to `metrics.json` as `vram_peak_gb`
* Change one variable at a time. One `metrics.json` per change, with p50 and p95 of `latency_s[]`.
* Practice: a 10-record Job over `kids_10.csv`. A baseline, then at least two changes.

Done when: you have a table with the baseline and at least two changes (`startup_s`, `load_s`, p50 and p95 generation time, `vram_peak_gb`, total run cost per asset at `N = 200`) and you picked the configuration for Homework 3.

## 3.4 Breaking the Pipeline on Purpose

* Two run-level thresholds, checked by your launcher before each chunk:
  * Budget: `budget_usd` per run. Don't submit a chunk if spend so far plus its estimate (`F + n × c`) is above the budget.
  * Quality: the 2.5 judge pass rate on the first records. Stop the run if it falls below a floor you set, before you pay for the rest.
* Inject one failure at a time with a `FAULT=<name>` switch, run it on 10 records, and compare what should happen with what happened
* Failures and fix patterns:

| Failure | Fix pattern |
|---|---|
| Spot Job preempted mid-chunk (error status) | Per-item checkpoint, resubmit the chunk with the same `run_id`, capped by `max_chunk_attempts` (3.2) |
| Job hits its timeout | Per-item timeout, smaller chunk |
| Out of memory on a large record | Fallback configuration (lower resolution, offloading from 3.3), then dead-letter |
| `QuotaExceeded` on submit | Lower `K`, queue the remaining chunks |
| Endpoint not ready or returns `503` | Readiness wait, retry with backoff and jitter |
| LLM rate limit `429` | Client-side rate limiter, backoff |
| Judge returns invalid output | Treat as `RETRY`, never `PASS` |
| Bucket write fails | Retry; the ledger entry is written last, so state stays clean |
| Same chunk submitted twice | Idempotency key, skip if done |
| Bad record: empty field, huge value, prompt injection text | Validate at the edge, dead-letter |
| Weights download fails or disk full | Cached weights from 3.3, disk size check before the run |
| Budget runs out mid-run | Stop submitting; skip remaining records with a reason; the run still writes `summary.json` |
| Actual cost above the estimate | Use the actual cost from finished chunks for the next estimate |
| Judge pass rate below the floor | Stop the run, keep finished items, report the pass rate |

* Fast path: inject faults on 10 records with Sana. Faults on video only in the 10-record sample.

Done when: you ran at least five faults, including one budget fault and one quality fault, and your table shows the expected and actual behaviour for each.

## 3.6 Fine-Tune a Model on a New Style (Optional, Advanced)

Teach an image model a style it can't do out of the box, run the training as a long-running Job, and use the result as your image tool. Follows the [BFL challenge guide: teach FLUX 3 Image Klein a new style, then make a film with it](https://accessible-coneflower-adf.notion.site/BFL-challenge-teach-FLUX-3-Image-Klein-a-new-style-then-make-a-film-with-it-3fcc370222d5814b9fabe3cb5b71e8aa).

Not part of the default path or Homework 3. Training takes hours of GPU time: estimate the cost with the 1.5 formulas before you start.

* Pick a model:
  * Default: [FLUX 3 Image Klein](https://docs.bfl.ai/flux_3/flux3_image_klein). Train a LoRA on [Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) (undistilled), generate with [Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) (distilled). The LoRA loads into the distilled model as is.
  * Also try: any open-weight model from the [1.1 table](../01-generative-media-workloads#11-landscape-modalities-and-model-families) that has a LoRA training script. Check its model card.
  * Read the licence first. Klein is under the [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/LICENSE.md): non-commercial use, plus commercial outputs for qualifying users, subject to its terms.
* Pick a style the base model doesn't have: describe it by its attributes (palette, stroke, texture, materials) and generate it with the base model. If the result is already close, pick another style.
* Build the dataset. Most of the work, and most of the quality, is here:
  * 20 to 40 images from one source, about 1 megapixel or more, that you have the rights to
  * The style is the only thing every image shares. Vary subject, framing, setting and light.
  * Drop duplicates and low-resolution images. Crop out text, signatures, watermarks and borders.
  * Captions describe content, never style, and end with a trigger word (for example `zqvr style`)
  * An agent can do most of this step: scrape, remove duplicates, caption with a VLM, flag captions that contain style words. You review a contact sheet.
* Train as a Job (1.4): [AI Toolkit](https://github.com/ostris/ai-toolkit) or the [Diffusers training script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux3_klein.md), with the [settings from the Klein docs](https://docs.bfl.ai/flux_3/flux3_image_klein#train-a-lora). Write checkpoints and sample images to the bucket. Set a Job timeout. On spot, resume from the last checkpoint (3.2).
* Wrap it as a tool with the [2.4](../02-agents-that-use-gpus#24-long-running-tools-submit-poll-resume) pattern: `start_finetune(dataset) -> ticket`, `get_status(ticket) -> state`. On resume, continue from the existing ticket, not a new training Job.
* Pick the checkpoint by its samples, not by loss. Early checkpoints apply the style weakly; late ones copy the training set. Use prompts with subjects that are not in the dataset. The [2.5](../02-agents-that-use-gpus#25-planner-generator-reviewer-the-judge-loop) judge can compare base vs LoRA on the same prompt and seed.
* Use it: load the LoRA into your image generation, then pass the frames to image-to-video. The video model never sees the LoRA; the style reaches the video through the frames.

Done when: you have a base vs LoRA pair with the same prompt and seed that shows the new style, the LoRA is in your bucket, and the training Job is COMPLETED with nothing left running.

## Homework 3: Factory Scale Challenge

Scale the Personalized Media Factory to all 200 records in `kids.csv` over 10 Jobs in parallel: card text, image and voice for every record, video for a 10-record sample. Use spot Jobs, your 3.3 configuration, and survive at least one kill → resume. Compete on the leaderboard: the cheapest reliable run at acceptable quality wins.

* Before you start: a quota of at least 10 preemptible VMs (see the note at the top)
* Submit your `summary.json`: records, Jobs, completed, dead-lettered, total cost, cost per accepted asset, saved vs on-demand, p50 and p95 time per asset, judge pass rate, and the resume proof (items done before the kill, items regenerated after it)
* Leaderboard: ranked by cost per accepted asset, among runs that meet the completion and quality bars

Questions and submission: [homework.md](../cohorts/2026/homework/03-scaling-reliability-cost/homework.md).

## Capstone Step

Run your own records through your capstone pipeline on 100 records (10 for video) with one kill → resume. Keep `summary.json`: it is your capstone cost report. See [projects](../projects).

# Community Notes

Did you take notes? You can share them here:

* Add your notes here!
