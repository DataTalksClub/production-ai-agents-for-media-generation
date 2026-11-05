# Module 1: Generative Media Workloads

Welcome to Module 1 of Production AI Agents for Generative Media! These two weeks we make it work: run image, video and audio models on serverless GPUs, for serving and for batch, and measure what one asset costs.

> [!NOTE]
> Get an account, a project and GPU credits through the [Nebius AI Builder Program](https://dev.nebius.com/builders) before you start.

> [!WARNING]
> Make sure you stop or delete your resources (Endpoints, Jobs, JupyterLab) when you finish the module to avoid charges. An Endpoint bills while it's RUNNING, even when it serves no requests.

* Video Link (Coming Soon)
* Slides (Coming Soon)

## Learning Objectives

By the end of this module, you will be able to:

* Compare generative media modalities and model families (size, VRAM, latency, quality, licence) and explain why video is expensive
* Generate your first image, video or audio asset in minutes: through an Endpoint, or in JupyterLab on a GPU
* Explain what goes into a container for generative AI and what stays out of it (weights, secrets, outputs)
* Run the same generation for serving (Endpoint) and for batch inference (Job), and choose between them
* Measure startup, readiness and generation time, and turn them into marginal and total cost per asset

## 1.1 Landscape: Modalities and Model Families

* Modalities: text-to-image, image editing, text-to-video, image-to-video, text-to-speech, text-to-music
* What changes between them: output size, steps, frames, VRAM, seconds per asset
* Why video is expensive: frames × resolution × denoising steps
* Model families (high level): size, VRAM, latency, quality
* Licences: what you can and cannot ship. Read the model card.
* Real projects: 2–3 short cases from Nebius customers and partners. Workload, scale, what broke, what they changed.

The models used in this module, all with 1-click templates in the [Serverless AI Cookbook](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/README.md):

| Task | Model | Template GPU |
|---|---|---|
| Text-to-image | [Sana](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-sana/README.md) | L40S |
| Text-to-image | [Qwen-Image](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-qwen-image/README.md) | H100 |
| Text-to-image | [Z-Image-Turbo](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-z-image-turbo/README.md) | H100 |
| Image editing | [Qwen-Image-Edit-2511](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-qwen-image-edit-2511/README.md) | H100 |
| Text-to-video | [Wan2.1-T2V-1.3B](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-wan21-t2v-1-3b/README.md) | H100 |
| Image-to-video | [Wan2.2-I2V-A14B](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-wan22-i2v-a14b/README.md) | H100 |
| Text-to-speech | [Kokoro-82M](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-kokoro-82m/README.md) | L40S |
| Text-to-speech | [Chatterbox](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-chatterbox/README.md) | H100 |
| Text-to-music | [ACE-Step 1.5](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-ace-step-1-5/README.md) | H100 |

## 1.2 First Media Generation in a Few Minutes

Pick one option. The default path is Option A with [Sana](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-sana/README.md) on L40S. The other models in the 1.1 table are there to explore.

**Option A: Endpoint + Jupyter on your laptop**

* Deploy an Endpoint from a cookbook template in the web console (1-click, fields pre-filled)
* Endpoint lifecycle: create → STARTING → RUNNING → ready → stop. RUNNING is not ready: requests fail until `/v1/models` returns JSON.
* Call it from a local notebook with plain HTTP: `/v1/images/generations`, `/v1/audio/speech`. Based on [`01_simple_media_agent.ipynb`](https://github.com/DataTalksClub/build-media-agent-workshop/blob/main/01_simple_media_agent.ipynb) from the workshop.
* Save the asset locally
* Stop the Endpoint

Done when: you have the asset file on your laptop and the Endpoint is stopped.

**Option B: JupyterLab on a GPU (DevLabs)**

* Start [JupyterLab on Nebius](https://docs.nebius.com/applications/standalone/jupyterlab) on a GPU
* Load several models from Hugging Face with Diffusers and play with them: prompt, seed, steps, resolution
* Watch VRAM and time per asset as you change them
* Write down the run configuration so you can repeat it: prompt, seed, parameters, pinned model revision, GPU, library versions
* Stop JupyterLab

Done when: you have generated assets with two models and JupyterLab is stopped.

A seed makes a run repeatable, not identical everywhere. The same configuration can still give slightly different outputs on a different GPU, driver or PyTorch version, or with non-deterministic kernels. See [PyTorch: Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html).

Code is OpenAI-compatible, so the pattern ports elsewhere.

## 1.3 Containers for Generative AI

High level. Optimization (cold start, model cache, VRAM, `torch.compile`) is in [3.3](../03-scaling-reliability-cost#33-optimizing-cost-per-asset-cold-start-vram-and-speed).

* Template vs bring-your-own container: when the cookbook template is enough
* CUDA/PyTorch base images
* Weights vs image: what goes where, HF cache, pinned revisions
* Secrets: env at runtime, never in the image
* I/O: inputs and outputs through a bucket mount or S3
* Bad Dockerfile → production Dockerfile, side by side (reading, no lab)

## 1.4 Serving vs Batch Inference: the Same Generation for Different Use Cases

* Serving (Endpoint) when a human or an agent waits on the result. Batch (Job) when it produces an artifact you store.
* The same generation as a batch Job: from the CLI (`nebius ai job create`) or from Python
* Job lifecycle: QUEUED → RUNNING → COMPLETED or FAILED. Jobs stop on completion. Endpoints don't.
* Store Job outputs in Object Storage: create a bucket and access keys, then mount the bucket to the Job or write with boto3 (S3-compatible, `endpoint_url`)
* Run the 1.2 generation as a Job with the same configuration (prompt, seed, parameters, model revision). Compare the two assets: they match only if the GPU, runtime and deterministic settings also match.
* What bills: Endpoints while RUNNING, even when idle. Jobs while running. Disks and buckets while they exist.
* Set a `--timeout` on every Job
* What this means for scale: preview of [Module 3](../03-scaling-reliability-cost)

Done when: the Job is COMPLETED, its asset is in your bucket, and you can say whether it matches the 1.2 asset and why.

## 1.5 Measuring Startup, Load and Generation Costs

* Four measurements, kept separate:
  * Deployment startup (observed): create → RUNNING, from status transitions
  * Readiness wait (observed, Endpoint): RUNNING → first JSON from `/v1/models`. A proxy for model load: it also includes server start and weight download.
  * Model load (internal), where available: a timer around model loading in your own code (Job, JupyterLab) or a load time in server logs. Template Endpoints may not expose it.
  * Request latency: send → response, per asset, measured by the client. For a Job or JupyterLab, the generation timer in your code.
* Billable interval, defined separately: the time you pay for. An Endpoint bills while it runs, idle or not. A Job bills while it runs. Your timers explain where that time goes, but the bill is the billable interval × rate. See [Serverless AI pricing](https://docs.nebius.com/serverless/pricing-quotas).
* Cold vs warm: the first asset is not the typical asset
* Cost per asset:
  * Marginal compute cost, `c = rate × t_gen`: the cost of one more asset on a running, ready deployment
  * Fixed cost of one run, `F = rate × (t_start + t_ready)`. `t_ready` is the readiness wait for an Endpoint, or model load for a Job or JupyterLab.
  * Total run cost per asset for `N` assets: `F / N + c`
  * Assumes one shared startup per run, assets generated one after another, and no idle gaps. Idle time adds `rate × t_idle` to the run.
* Worked example: the reference Sana run, Endpoint vs Job at 1, 10 and 100 assets
* Practice: recompute the total run cost per asset for your own `N`, and add 10 minutes of idle time before you stop the Endpoint

Done when: you have `metrics.json` with all four measurements and your cost per asset at 1, 10 and 100 assets.

## Homework 1: GPU Media Worker

Generate one media asset with any model from the [table in 1.1](#11-landscape-modalities-and-model-families), measure it, and compare its cost with a per-call API.

* Notebook or `.py` script, your choice
* Report your model, generation time and cost per asset, and the creative idea behind your asset
* Find the per-call API price for the same model and compute your self-host breakeven in number of assets

Questions and submission: [homework.md](../cohorts/2026/homework/01-generative-media-workloads/homework.md).

## Capstone Proposal

Write a one-page proposal for your capstone: problem, records, two GPU media models and a budget estimate. Template: [projects](../projects).

# Community Notes

Did you take notes? You can share them here:

* Add your notes here!
