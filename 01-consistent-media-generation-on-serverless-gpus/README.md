<p align="center">
  <img src="../images/module-1-cover.png" alt="Module 1, Make it work: Consistent Media Generation on Serverless GPUs. Media model landscape, templates, containers and model cache, seeds, fitting VRAM, startup, load and generation time. You ship Homework 1: GPU Media Worker" width="100%">
</p>

# Module 1: Consistent Media Generation on Serverless GPUs

Welcome to Module 1 of Production AI Agents for Generative Media! These two weeks we make it work: run image, video and audio models on serverless GPUs, for serving and for batch, and measure what one asset costs.

> [!NOTE]
> Get an account, a project and GPU credits through the [Nebius AI Builder Program](https://dev.nebius.com/builders) before you start.

> [!WARNING]
> Make sure you stop or delete your resources (Endpoints, Jobs, JupyterLab) when you finish the module to avoid charges. An Endpoint bills while it's RUNNING, even when it serves no requests.

* Video Link (Coming Soon)
* Slides (Coming Soon)

<p align="center">
  <img src="../images/diagram-media-factory.png" alt="The Personalized Media Factory: a brief goes to an agent, GPU models generate image, voice or video, a check sends assets that aren't good enough back to revise and regenerate, and the factory scales from 1 to 1,000+ records with a cost per asset" width="100%">
</p>

## What you will learn

By the end of this module, you will be able to:

* Compare generative media modalities and model families (size, VRAM, latency, quality, licence) and explain why video is expensive
* Generate your first image, video or audio asset in minutes: through an Endpoint, or in JupyterLab on a GPU
* Explain what goes into a container for generative AI and what stays out of it (weights, secrets, outputs)
* Run the same generation for serving (Endpoint) and for batch inference (Job), and choose between them
* Measure startup, readiness and generation time, and turn them into marginal and total cost per asset

## Lessons

* [1.1 Landscape: Modalities and Model Families](01-landscape.md)
* [1.2 First Media Generation in a Few Minutes](02-first-generation.md)
* [1.3 Containers for Generative AI](03-containers.md)
* [1.4 Serving vs Batch Inference: the Same Generation for Different Use Cases](04-serving-vs-batch.md)
* [1.5 Measuring Startup, Load and Generation Costs](05-measuring-costs.md)

## Models used in this module

All with 1-click templates in the [Serverless AI Cookbook](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/README.md). The default path is [Sana](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-sana/README.md) on L40S; the others are there to explore.

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

## Homework 1: GPU Media Worker

Generate one media asset with any model from the [table above](#models-used-in-this-module), measure it, and compare its cost with a per-call API.

* Notebook or `.py` script, your choice
* Report your model, generation time and cost per asset, and the creative idea behind your asset
* Find the per-call API price for the same model and compute your self-host breakeven in number of assets

Questions and submission: [homework.md](../cohorts/2026/homework/01-consistent-media-generation-on-serverless-gpus/homework.md).

## Capstone Proposal

Write a one-page proposal for your capstone: problem, records, two GPU media models and a budget estimate. Template: [proposal-template.md](../projects/proposal-template.md) in [projects](../projects).

## Module structure

One page per lesson at the module root, all code flat in [code/](code/README.md):

```text
01-consistent-media-generation-on-serverless-gpus/
├── README.md                       # module overview and lessons index
├── 01-landscape.md                 # 1.1: modalities, model families, licences
├── 02-first-generation.md          # 1.2: Option A (Endpoint) & Option B (GPU DevLab)
├── 03-containers.md                # 1.3: bad vs production Dockerfile, line by line
├── 04-serving-vs-batch.md          # 1.4: Endpoint vs Job + Object Storage
├── 05-measuring-costs.md           # 1.5: startup, load, latency, calculate_cost()
└── code/
    ├── README.md                   # how to run each file, env vars
    ├── generation.json             # shared prompt, seed, pinned model commit, params
    ├── common.py                   # config, metrics.json, trace.jsonl, calculate_cost()
    ├── endpoint_quickstart.ipynb   # 1.2 Option A (default): laptop notebook → Endpoint
    ├── jupyterlab_models.ipynb     # 1.2 Option B (optional): GPU DevLab with Diffusers
    ├── Dockerfile.bad              # 1.3: what not to do
    ├── Dockerfile                  # 1.3: production worker container
    ├── generate.py                 # 1.4: sequential Diffusers worker for Job & DevLab
    ├── call_endpoint.py            # 1.2 & 1.4: ImageGenerator client for an Endpoint
    ├── measure_endpoint.py         # 1.5: record Endpoint startup, readiness & latency
    └── measure_job.py              # 1.5: enrich metrics.json with Job lifecycle timestamps
```

# Community Notes

Did you take notes? You can share them here:

* Add your notes here!
