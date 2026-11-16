## Homework 1: GPU Media Worker

> [!NOTE]
> You need a Nebius project with credits. See [Module 1](../../../../01-consistent-media-generation-on-serverless-gpus/README.md).
> Stop or delete every Endpoint, Job and JupyterLab when you're done. A forgotten Endpoint bills while it's RUNNING.

In this homework, you generate 10 media assets on a GPU in one run, measure the run, and decide when self-hosting beats a per-call API.

### Find a creative idea

Start with what you want to create, then pick the model. A few ideas:

* A cartoon character or a comic panel
* A birthday or holiday gift card
* A product ad shot or a short product clip
* A children's storybook illustration
* A short looping video for social media
* A personal voice greeting or a podcast intro (speech)
* A jingle or background track (music)

You will reuse the same idea in Homework 3, so pick one you like.

### Setup

* Use a Jupyter notebook or a `.py` script.
* Pick **one** model from the list below. Any modality: image, video, speech or music.
* Run it any way from Module 1: an Endpoint called from your laptop, a Job, or a DevLab (JupyterLab on a GPU). The calculation below is the same for all three.

| Task | Model |
|---|---|
| Text-to-image | Sana, Qwen-Image, Z-Image-Turbo |
| Image editing | Qwen-Image-Edit-2511 |
| Text-to-video | Wan2.1-T2V-1.3B |
| Image-to-video | Wan2.2-I2V-A14B |
| Text-to-speech | Kokoro-82M, Chatterbox |
| Text-to-music | ACE-Step 1.5 |

Templates for all of them: [Serverless AI Cookbook](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/README.md).

### Generate and measure

See [05 - Measuring Startup, Load and Generation Costs](../../../../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md).

* Turn your creative idea into a prompt.
* Generate **10 assets in one run**: one startup, one model load, ten generations. Use the same prompt and parameters, or ten variations of your idea.
* Record how you ran it:
    * **Model**
    * **Ran as:** Endpoint, Job or DevLab
    * **GPU** and **region**
    * **Price per hour** for your run: the regular price, or for a preemptible run the spot price at the time of the run. Rates: [Nebius pricing](https://docs.nebius.com/compute/resources/pricing), [spot prices](https://docs.nebius.com/signup-billing/spot-prices). A spot price you read once is a snapshot, and spot can change during a run, so every cost below is a projection, not what you paid. What you paid is in **Billing → Usage**.
* Record, in seconds:
    * **Startup time `t_start`:** Job or DevLab: from create until it is RUNNING. Endpoint: from clicking Create until `/v1/models` first returns JSON, as one interval (`create_to_ready_s` in lesson 02's `metrics.json`).
    * **Ready time `t_ready`:** Job or DevLab: the model load timer in your code (`load_s`). Endpoint: `0`, because it's already in `t_start`. Don't add the notebook's polling time to the console's RUNNING time: polling starts when you run the cell, so the gap between them gets lost.
    * **Generation time `t_gen`:** per asset, for each of the 10. Endpoint: request latency from your client. Job or DevLab: the generation timer in your code. Use the median.
* Save the assets. If you ran a Job, they are in your bucket.

You need `t_start` and `t_ready` for the breakeven below. Keep your numbers: you will compare them with Homework 3.

### Find the API price

Find a service that sells the same model as a per-call API. If none does, use the closest equivalent and say which in Question 9. Write down the service, the price per asset for your settings (resolution, duration or characters), and the link to its pricing page.

### Compute the breakeven

Let:

* `rate` = your GPU price per second (price per hour / 3600)
* `t_start`, `t_ready`, `t_gen` = your measured times, in seconds
* `p` = the API price per asset

Then:

* Marginal compute cost per asset: `c = rate × t_gen`
* Fixed cost of one run (startup + ready): `F = rate × (t_start + t_ready)`
* Total run cost per asset for `N` assets: `F / N + c`. For your run, `N = 10`.
* Breakeven, in assets per run: `N = ceil(F / (p − c))`

The formulas assume one startup per run, assets generated one after another, and no idle time. If `p ≤ c`, self-hosting never breaks even on this model. Enter `0`.

### Question 1

What is the creative idea behind your asset? Why did you choose this topic? Two or three sentences.

### Question 2

Which model did you use?

- Sana
- Qwen-Image
- Z-Image-Turbo
- Qwen-Image-Edit-2511
- Wan2.1-T2V-1.3B
- Wan2.2-I2V-A14B
- Kokoro-82M
- Chatterbox
- ACE-Step 1.5

### Question 3

How did you run it?

- Endpoint, called from my laptop
- Job
- DevLab (JupyterLab on a GPU)

### Question 4

Which GPU did you run on?

- L40S
- H100
- H200
- RTX PRO 6000
- Other

### Question 5

Which region did you run in? For example `eu-north1`.

### Question 6

What was your generation time per asset (median of your 10), in seconds?

### Question 7

What is your projected marginal compute cost per asset, `c`, in USD?

### Question 8

What is your projected total cost per asset for your 10-asset run, `F / 10 + c`, in USD?

### Question 9

Which per-call API did you compare with, and what does it charge per asset? Format: `service, model, $price per asset, link`. Name the model if it isn't the one you ran.

### Question 10

What is your self-host breakeven `N`, in number of assets?

## Learning in Public

Share your asset and the idea behind it in a short post. Use `#[placeholder]`, tag DataTalksClub and Nebius, and submit the post link with your homework. Don't share credentials, Endpoint tokens or bucket keys.

See the [learning in public guide](https://datatalks.club/docs/courses/zoomcamp-logistics/learning-in-public/) for more ideas.

### Example post for LinkedIn

```text
Completed Module 1 of Production AI Agents for Generative Media!

I ran <model> on Nebius Serverless AI and generated this <image/video/audio> (attached).
It takes <Z> s to generate one asset (median of 10).

The idea: <what you wanted to create and why you chose this topic>.

My code: <LINK>

#[placeholder] @DataTalksClub @Nebius
```

## Submit the results

* Submit your results here: [placeholder]
* All questions are about your own run. Any honest answer counts.
