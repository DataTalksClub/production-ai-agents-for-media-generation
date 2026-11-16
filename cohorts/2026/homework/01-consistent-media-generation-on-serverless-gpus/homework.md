## Homework 1: GPU Media Worker

> [!NOTE]
> You need a Nebius project with credits. See [Module 1](../../../../01-consistent-media-generation-on-serverless-gpus/README.md).
> Stop or delete every Endpoint, Job and DevLab when you're done. A forgotten Endpoint bills while it's RUNNING.

In this homework, you generate 10 media assets on a GPU in one run, as in [lesson 02](../../../../01-consistent-media-generation-on-serverless-gpus/02-first-generation.md), and measure their cost, as in [lesson 05](../../../../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md).

### Find a creative idea

Start with what you want to create, then pick the model. Any media works: image, video, speech, music or something else. A few ideas:

* A cartoon character or a comic panel
* A birthday or holiday gift card
* A product ad shot or a short product clip
* A children's storybook illustration
* A short looping video for social media
* A personal voice greeting or a podcast intro
* A jingle or background track

You can reuse the same idea in Homework 3, so pick one you like.

### Pick a model and how to run it

Any open model works. The [Serverless AI Cookbook templates](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/README.md) deploy image, video, speech and music models in a few clicks.

Run it one of three ways from Module 1:

* **Endpoint**, called from your laptop: [lesson 02, Option A](../../../../01-consistent-media-generation-on-serverless-gpus/02-first-generation.md#option-a-endpoint-and-notebook-on-your-laptop). Works with any template.
* **DevLab** (JupyterLab on a GPU): [lesson 02, Option B](../../../../01-consistent-media-generation-on-serverless-gpus/02-first-generation.md#option-b-jupyterlab-on-a-gpu).
* **Job**: [lesson 04, Path B](../../../../01-consistent-media-generation-on-serverless-gpus/04-serving-vs-batch.md#path-b-a-job-from-your-laptop). The lesson 04 worker runs Sana (image) or Kokoro-82M (speech).

The easiest path is Sana with the lesson 02 notebook: it records the times for you.

Run on a preemptible (spot) GPU when you can: it's cheaper and leaves more of your credits for the next modules. The cookbook templates and the lesson 04 Job are preemptible by default. A DevLab bills at the regular price.

### Generate and measure

1. Turn your idea into a prompt.
2. Generate **10 assets in one run**: one startup, one model load, ten generations. Use the same prompt, or ten variations of your idea.
    * Lesson 02 notebooks: change the step 4 loop from `range(3)` to `range(10)`.
    * Job: run with `--n 10` and only the steps of your model: `--step text --step illustration` for Sana, or `--step text --step greeting` for Kokoro. Upload your own 10-row CSV with the same columns as `kids.csv`, with your idea in `prompt` (image) or `greeting` (speech).
3. Save the assets. Then stop the Endpoint, or delete the DevLab.
4. Open [nb-measuring-costs.ipynb](../../../../01-consistent-media-generation-on-serverless-gpus/code/nb-measuring-costs.ipynb) and fill in section 5 with your run. Where each time comes from: [lesson 05, Measure a run of your own](../../../../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md#measure-a-run-of-your-own-homework).
5. Use the preemptible (spot) price per hour for your GPU and region at the time of the run: [spot prices](https://docs.nebius.com/signup-billing/spot-prices), or [Compute pricing](https://docs.nebius.com/compute/resources/pricing) for GPUs with a fixed preemptible price such as the L40S. See [lesson 05, Prices and assumptions](../../../../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md#prices-and-assumptions) for how to read them. On a DevLab, use the regular price.
6. Project the cost per asset to N = 1000 from your run: [lesson 05, From time to money](../../../../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md#from-time-to-money).

### Find a benchmark

Find a service that sells the same model, or the closest one, as a per-call API. Convert its price to USD per asset for your settings (resolution, duration or characters). Lesson 05 does it for one model: [The per-call API](../../../../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md#the-per-call-api-wan-22-a14b-on-fal).

### Question 1

What is the creative idea behind your asset? Why did you choose this topic? Two or three sentences.

### Question 2

Which model did you use?

### Question 3

Which modality?

- Image
- Video
- Speech
- Music
- Other

### Question 4

How did you run it?

- Endpoint
- DevLab
- Job

### Question 5

Which GPU did you run on?

- L40S
- H100
- H200
- RTX PRO 6000
- Other

### Question 6

What was your median generation time per media asset, in seconds?

### Question 7

What is your projected cost per asset at N = 1000, in USD?

### Question 8

Which benchmark service or API did you compare with? Text or link.

### Question 9

What does the benchmark charge per asset, in USD?

## Learning in Public

Share your asset and the idea behind it in a short post. Use `#[placeholder]`, tag DataTalksClub and Nebius, and submit the post link with your homework. Don't share credentials, Endpoint tokens or bucket keys.

See the [learning in public guide](https://datatalks.club/docs/courses/zoomcamp-logistics/learning-in-public/) for more ideas.

### Example post for LinkedIn

```text
Completed Module 1 of Production AI Agents for Generative Media!

I ran <model> on Nebius Serverless AI and generated this <image/video/audio> (attached).
It takes <Z> s to generate one asset (median of 10).
At 1000 assets, one costs about $<X>; <benchmark> charges $<Y>.

The idea: <what you wanted to create and why you chose this topic>.

My code: <LINK>

#[placeholder] @DataTalksClub @Nebius
```

## Submit the results

* Submit your results here: [placeholder]
* All questions are about your own run. Any honest answer counts.

## Clean up

> [!WARNING]
> 🧹 **Remove all running resources when you're done, even after a failed run.** Stop or delete every Endpoint, Job and DevLab you started. An Endpoint or DevLab keeps billing while it runs, even when idle, and a stopped DevLab still bills its disk until you delete it. Delete bucket data you no longer need. Check **Billing → Usage** the next day to confirm nothing is still billing.
