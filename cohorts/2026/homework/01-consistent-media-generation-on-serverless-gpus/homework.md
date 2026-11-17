## Homework 1: GPU Media Worker

> [!NOTE]
> You need a Nebius project with credits. See [Module 1](../../../../01-consistent-media-generation-on-serverless-gpus/README.md).
> Stop or delete every Endpoint, Job and DevLab when you're done. A forgotten Endpoint bills while it's RUNNING.

In this homework, you generate 10 media assets on a GPU in one run and work out their cost the way [lesson 05](../../../../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md) does.

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

Run it any way from Module 1: an Endpoint called from your laptop, a DevLab, or a Job.

The easiest path is Sana with the lesson 02 notebook: it records the times for you.

Run on a preemptible (spot) GPU when you can: it's cheaper and leaves more of your credits for the next modules. The cookbook templates and the lesson 04 Job are preemptible by default. A DevLab bills at the regular price.

### Generate and measure

1. Turn your idea into a prompt.
2. Generate **10 assets in one run**: one startup, one model load, ten generations. Use the same prompt, or ten variations of your idea.
3. Save the assets. Then stop the Endpoint, or delete the DevLab.
4. Record the startup, the model load and the time for each asset. Split them into the fixed cost per run and the cost per asset, as lesson 05 does.
5. Use the price per hour your run paid: the preemptible price for your GPU and region at the time of the run, or the regular price on a DevLab. See [Nebius prices](https://nebius.com/prices).
6. Project the cost per asset to N = 1000 from your run.

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

## Learning in Public

Share your asset and the idea behind it in a short post. Tag DataTalksClub and Nebius, and submit the post link with your homework. Don't share credentials, Endpoint tokens or bucket keys.

See the [learning in public guide](https://datatalks.club/docs/courses/zoomcamp-logistics/learning-in-public/) for more ideas.

### Example post for LinkedIn

```text
Completed Module 1 of Production AI Agents for Generative Media!

I ran <model> on Nebius Serverless AI and generated this <image/video/audio> (attached).
It takes <Z> s to generate one asset (median of 10).
At 1000 assets, one costs about $<X>.

The idea: <what you wanted to create and why you chose this topic>.

My code: <LINK>

@DataTalksClub @Nebius
```

## Submit the results

* Submit your results here: https://courses.datatalks.club/production-ai-agents-generative-media/homework/module1
* All questions are about your own run. Any honest answer counts.

## Clean up

> [!WARNING]
> 🧹 **Remove all running resources when you're done, even after a failed run.** Stop or delete every Endpoint, Job and DevLab you started. An Endpoint or DevLab keeps billing while it runs, even when idle, and a stopped DevLab still bills its disk until you delete it. Delete bucket data you no longer need. Check **Billing → Usage** the next day to confirm nothing is still billing.
