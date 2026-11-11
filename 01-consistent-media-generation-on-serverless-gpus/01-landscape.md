---
video_url: ""
prev_url: README.md
next_url: 02-first-generation.md
---
# 1.1 Landscape: Modalities and Model Families

Say a toy store wants to send every child on its mailing list a personalized holiday card: an illustration, a spoken greeting, maybe a short animated clip. Take Emma, a fictional 7-year-old who wishes for a telescope. Before writing any code, we have to choose which models make each asset, what they cost to run, and what their licences allow.

In this lesson we map the generative media modalities, see why video costs the most, compare open and closed models, and check what a licence really allows. You're done when you have written a five-line **model decision record** for your [Homework 1](../cohorts/2026/homework/01-consistent-media-generation-on-serverless-gpus/homework.md) idea: one asset you want to create (a card, a product shot, a voice greeting, a jingle). No idea yet? Use Emma's card.

Keep this template open while you read:

```text
1. Modality and model:      <task>, <model from the table at the end>
2. GPU:                     <the recommended GPU for that model>
3. Licence:                 <link to the model card>, commercial use: <yes / with conditions / no>
4. Cost driver (estimate):  <the closest row in the measured table and why this modality costs what it costs>
5. API to compare in HW1:   <a hosted API for the same or a similar model, with its pricing link>
```

## Modalities

Every generative media task maps an input (a prompt, a reference image, a script) to a media file:

| Task | Typical inputs | Output | What you inspect |
|---|---|---|---|
| **Text-to-image** | Prompt, size, seed | Image (`PNG`) | Subject, style, composition, no unwanted text |
| **Image editing** | Source image(s) + instruction, seed | Image (`PNG`) | The edit is applied; the rest is preserved |
| **Text-to-video** | Motion prompt, resolution, frames, fps, seed | Clip (`MP4`) | Smooth motion; objects persist |
| **Image-to-video** | Keyframe + motion prompt, duration, seed | Clip (`MP4`) | Identity kept from the first frame |
| **Text-to-speech** | Script, voice, speed | Audio (`MP3`/`WAV`) | Pronunciation, prosody, no clipping |
| **Text-to-music** | Style tags, lyrics or prompt, duration, seed | Audio (`MP3`/`WAV`) | Structure, tempo, no distortion |

For Emma's card, each part of the brief maps to a modality and a family of models:

```mermaid
flowchart LR
  Brief["Emma's card"] --> Modality{"Which asset?"}
  Modality -->|"Illustration"| Image["Image / edit<br/>e.g. Sana, Qwen-Image"]
  Modality -->|"Greeting"| Speech["Speech<br/>e.g. Kokoro, Chatterbox"]
  Modality -->|"Jingle"| Music["Music<br/>e.g. ACE-Step"]
  Modality -->|"Clip"| Video["Video<br/>e.g. Wan2.2"]
```

## What changes between modalities

Switching modality changes four numbers at once:

- **Latent size:** the model works on a compressed version of the media, called a latent. It's a 2D grid for an image, a sequence over time for audio, and a 3D volume (frames x height x width) for video.
- **Peak GPU memory (VRAM):** model weights plus the text encoder, the decoder and the working memory used during generation.
- **Load time:** fetching the weights and moving them onto the GPU, once per deployment.
- **Generation time** (`t_gen`): making one asset with the model already loaded.

Image, video and audio models now share one recipe: compress the media into a latent, cut it into tokens, let a transformer remove noise from those tokens step by step, then decode the result back into pixels or sound. What differs is how many tokens each asset needs (conceptual diagram):

```mermaid
flowchart LR
  I["Image<br/>1024x1024"] --> IL["2D latent<br/>~1,000 tokens (Sana)"]
  V["Video<br/>5 s, 121 frames"] --> VL["3D latent<br/>~12,000 tokens (Wan2.2-5B)"]
  A["Audio<br/>8 s track"] --> AL["latent over time"]
  IL --> T["Transformer removes noise,<br/>step by step"]
  VL --> T
  AL --> T
  T --> D["Decoder: back to<br/>pixels or sound"]
```

To see the real thing: the transformer backbone in [DiT, Figure 3](https://arxiv.org/abs/2212.09748); video turned into "spacetime patches" in [OpenAI's Sora report](https://openai.com/index/video-generation-models-as-world-simulators/); and a video autoencoder that compresses time as well as space in the [Wan report, Figure 5](https://arxiv.org/abs/2503.20314). <!-- c:M1-175 c:M1-176 c:M1-177 -->

We measured one reference model per modality on a single NVIDIA L40S GPU (48 GB VRAM) `[measured: 2026-11-02, gpu-l40s-a, 1gpu-8vcpu-32gb, eu-north1]`. Generation time depends on the GPU, the resolution, the number of steps and other settings, so another GPU or setting gives different numbers:
<!-- verify before release: link the 2026-11-02 run artifacts (metrics.json per row). -->
<!-- TODO(1.2): embed the reference outputs per row (image, speech, music, video) produced in 1.2. -->

| Modality | Model and settings | Peak VRAM | `t_gen` | Output |
|---|---|---|---|---|
| Text-to-speech | `hexgrad/Kokoro-82M` (20-word script) | `0.73 GB` | `1.88 s` | 7.6 s of speech, MP3 |
| Text-to-image | `Efficient-Large-Model/Sana_1600M_1024px_diffusers` (`1024x1024`, 20 steps) | `10.91 GB` | `4.42 s` | 1024x1024 PNG |
| Text-to-music | `ACE-Step/acestep-v15-xl-turbo-diffusers` (8 s motif) | `11.86 GB` | `2.88 s` | 8 s of music, MP3 |
| Text/image-to-video | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (`832x480`, 121 frames, 50 steps) | `27.89 GB` | `157.49 s` | 5 s clip, MP4 |

You'll generate assets like these yourself in 1.2. If your model isn't in this table, use the row for its modality as your estimate.

## Why video is expensive

On the same GPU, one 5-second clip from Wan2.2-TI2V-5B takes about **36 times** longer than one Sana image (`157.49 s` vs `4.42 s` `[calc]`). Three things drive the gap:

1. **Many more tokens.** Sana's autoencoder compresses each side of the image 32x, so a 1024x1024 image becomes about 1,000 tokens. Wan2.2-TI2V-5B compresses time 4x and each side 32x, so our 121-frame `832x480` clip becomes about 12,000 tokens `[calc: 31 x 26 x 15]`. Sana uses linear attention, whose cost grows with the token count; Wan uses full attention, whose cost grows with its square. [Sana](https://arxiv.org/abs/2410.10629), [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) <!-- c:M1-044 c:M1-045 c:M1-067 -->
2. **More steps.** Our image run used 20 denoising steps; the video run used 50.
3. **Parameters aren't the memory bill.** TI2V-5B is about 10 GB of weights in `bfloat16` `[calc]`, yet it peaked at 27.89 GB: the text encoder, working memory and the 3D decoder add the rest.

The larger [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) has two ~14B experts: 27B parameters in total, 14B active per step. Its official script at 720p needs at least 80 GB of VRAM, so it runs on an H100 rather than an L40S. Don't carry the 36x figure over to it; it's a different model on a different GPU. <!-- c:M1-062 c:M1-064 -->

This gap shapes how you build a pipeline: generate and check the still image first, animate only the images that pass, and make video for a sample of records rather than all of them.

Fewer steps is the other lever. [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) is distilled to 8 steps, at the cost of less variety; [Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) makes video in 4 steps, with artifacts on very large motion (both vendor-reported). Module 3 measures what that does to cost. <!-- c:M1-170 c:M1-171 -->

## Model families

- **Sana** (~1.6B, image): fast at 1024px; can't render complex legible text. [Card](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers) <!-- c:M1-046 c:M1-047 -->
- **Qwen-Image** (20B, image) and **Qwen-Image-Edit-2511** (edits from one or more input images): large multimodal diffusion transformers. [Card](https://huggingface.co/Qwen/Qwen-Image) <!-- c:M1-049 c:M1-050 -->
- **Z-Image-Turbo** (6B, image): distilled for 8-step generation. [Card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) <!-- c:M1-055 c:M1-056 -->
- **Wan2.1 / Wan2.2** (1.3B to 27B, video): text-to-video and image-to-video. [Wan2.2 card](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) <!-- c:M1-062 -->
- **Kokoro-82M** (speech): 54 preset voices in 8 languages; ships no voice cloning. [Card](https://huggingface.co/hexgrad/Kokoro-82M) <!-- c:M1-071 -->
- **Chatterbox** (0.5B, speech): clones a voice from a short reference clip and watermarks every output. Clone only voices you have consent to use. [Card](https://huggingface.co/ResembleAI/chatterbox) <!-- c:M1-073 c:M1-074 c:M1-121 -->
- **ACE-Step 1.5** (music): a language model plans the song and a diffusion transformer renders it. [Paper](https://arxiv.org/abs/2602.00744) <!-- c:M1-078 -->

## Open vs closed models

Closed models lead on image quality; open weights are close on video. [Artificial Analysis](https://artificialanalysis.ai/image/leaderboard/text-to-image) ranks models by blind pairwise preference votes (Elo), one board per task. Snapshot of 11 Nov 2026:

| Board | #1 (closed) | Best open weights | Elo gap `[calc]` |
|---|---|---|---|
| [Text-to-image](https://artificialanalysis.ai/image/leaderboard/text-to-image) | GPT Image 2.6 Sunburst (max), 1259 | Qwen-Image-2.1 (#23), 1046 | 213 |
| [Image editing](https://artificialanalysis.ai/image/leaderboard/editing) | GPT Image 2.6 Sunburst (max), 1209 | Fibo 2.0 (#26), 1086 | 123 |
| [Text-to-video, with audio](https://artificialanalysis.ai/video/leaderboard/text-to-video) | Wan 3.0, 1175 | MiniMax H3 (#3), 1161 | 14 |
| [Image-to-video, with audio](https://artificialanalysis.ai/video/leaderboard/image-to-video) | Seedance 2.5, 1161 | MiniMax H3 (#4), 1133 | 28 |
| [Text-to-speech, provider voices](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice) | Eleven v4, 1337 | Fish Audio Drama 3 (#8), 1274 | 63 |

<!-- c:M1-123 c:M1-124 c:M1-125 c:M1-127 c:M1-128 c:M1-130 c:M1-131 c:M1-133 c:M1-134 c:M1-135 c:M1-136 -->

Compare gaps only within a board: each has its own Elo scale, the speech board uses public votes, and the 14-point video gap is inside the confidence interval. Video boards now come with and without audio, because models such as MiniMax H3 and [LTX-2](https://huggingface.co/Lightricks/LTX-2) generate the soundtrack with the picture. "Open" needs a qualifier here: MiniMax H3's licence requires organisations in the EU, UK, US and South Korea to apply for permission. <!-- c:M1-167 c:M1-164 c:M1-166 c:M1-149 -->

**Closed APIs are a moving target.** Hosted models are retired on a schedule:

```mermaid
timeline
  title Hosted model shutdowns, 2026–27
  12 May 2026 : OpenAI removes dall-e-2 and dall-e-3
  24 Sep 2026 : OpenAI removes Sora 2 and the Videos API
  22 Oct 2026 : Veo 3.1 preview shut down in the Gemini API
  23 Oct 2026 : OpenAI shuts down gpt-image-1
  1 Dec 2026 : gpt-image-1.5 and gpt-image-1-mini (scheduled)
  4 Jan 2027 : fal's hosted Wan 2.2 A14B endpoint (scheduled)
```

Sources: [OpenAI deprecations](https://platform.openai.com/docs/deprecations), [Gemini API Veo](https://ai.google.dev/gemini-api/docs/veo) (Veo 3.1 GA remains on Google's enterprise platform), [fal](https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-video). <!-- c:M1-138 c:M1-139 c:M1-140 c:M1-141 c:M1-142 c:M1-143 -->

**When to just use an API.** If you need the best still images, make a few hundred assets a month, and have no one to run GPUs, a per-call API is the sensible choice. Self-hosting pays off when you need a model version you control (pin it and keep a copy of the weights), licence terms fixed to that model, a custom or fine-tuned model, or a volume where your cost per asset beats the API. You'll measure that breakeven in Homework 1.

## Licences

Open weights don't mean free for any commercial use, and the licence belongs to the exact model you deploy.

- **Sana** checkpoints are Apache-2.0. Their card adds the Gemma terms for the text encoder and says the model is "intended for research purposes only". Read both before commercial use. [Card](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers) <!-- c:M1-041 c:M1-042 c:M1-043 -->
- **Wan** cards add to Apache-2.0 that use "must not involve sharing any content that violates applicable laws, causes harm". [Card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) <!-- c:M1-059 c:M1-061 -->
- **Qwen-Image** is Apache-2.0; **Qwen-Image-2.1** is a different model with a research licence (below). [Card](https://huggingface.co/Qwen/Qwen-Image) <!-- c:M1-048 -->

Several strong "open" models of 2026 come with conditions. This is a summary, not legal advice; read the linked text:

| Model | Licence | Main conditions |
|---|---|---|
| [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) | Qwen Research License | No commercial use without a separate licence from Qwen |
| [FLUX.2 [dev]](https://huggingface.co/black-forest-labs/FLUX.2-dev) | FLUX Non-Commercial License | Weights non-commercial; outputs follow the licence's own terms |
| [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md) | FLUX Kommunity License | Output use for companies under US$5M revenue (with affiliates), with content filters or output review and AI disclosure; BFL's product page says product use needs a commercial licence, so confirm with BFL |
| [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | LTX-2.x Community License | No cost under US$10M revenue across the whole company group, subject to its use restrictions |
| [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md) | MiniMax H3 Community License | EU, UK, South Korea and US organisations must apply; MiniMax may authorise |

<!-- c:M1-144 c:M1-147 c:M1-145 c:M1-148 c:M1-149 -->

One more trap for serverless: Black Forest Labs' standard commercial weights licence doesn't allow exposing the model through an external-facing API endpoint. [BFL](https://bfl.ai/open-weights) <!-- c:M1-146 -->

## What "consistent" means in this module

"Consistent" covers five problems. A fixed seed solves only the first:

| Kind | What it means | What breaks it | See it | Where we handle it |
|---|---|---|---|---|
| **Reproducibility** | Same settings and seed give the same output, within a tolerance | Another PyTorch release, GPU or attention backend; reusing a seeded generator | [Diffusers guide](https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds) | 1.2: same seed, same image; 1.4: compare runs |
| **Identity** | The same character or product in a new scene | A seed doesn't carry identity; models paste the reference instead of re-rendering it | [ConsiStory comparison](https://research.nvidia.com/labs/par/projects/consistory/), [OpenS2V copy-paste](https://pku-yuangroup.github.io/OpenS2V-Nexus/) | Reference-image editing (Qwen-Image-Edit-2511); Module 2 judge loop |
| **Temporal** | No flicker within a clip, no drift when a clip is extended | Unstable frames; errors that pile up as a clip is extended | [FramePack drift](https://lllyasviel.github.io/frame_pack_gitpage/), [VBench flicker](https://vchitect.github.io/VBench-project/) | Short clips; Module 4 evaluation |
| **Cross-asset** | The card, the clip and the next shot show the same Emma | Each asset made in isolation | [HoloCine multi-shot](https://holo-cine.github.io/) | Image first, then image-to-video (Module 2) |
| **Voice** | The same voice in every clip | A changed voice; a reference clip in another language | [Chatterbox demos](https://resemble-ai.github.io/chatterbox_demopage/) | Kokoro preset voices; one fixed reference clip for Chatterbox |

<!-- c:M1-096 c:M1-098 c:M1-099 c:M1-101 c:M1-102 c:M1-106 c:M1-107 c:M1-113 c:M1-117 c:M1-121 c:M1-122 c:M1-178 c:M1-179 c:M1-180 c:M1-181 c:M1-182 c:M1-183 -->

Watch the metrics too: a model that pastes the reference scores high on identity similarity while looking wrong. [OpenS2V-Eval paper](https://arxiv.org/abs/2505.20292)

## Projects you can try

We picked projects you can try or reproduce in a few minutes: open source or a free tier, close to the workloads in this course, and on open-weights models where possible.

- **Edit an image or clone a voice in the browser.** Hugging Face Spaces run [Qwen-Image-Edit-2511](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2511), [Kokoro](https://huggingface.co/spaces/hexgrad/Kokoro-TTS) and [Chatterbox](https://huggingface.co/spaces/ResembleAI/Chatterbox) for free, with GPU quotas. Try a product photo edit and a greeting in two voices. <!-- c:M1-187 -->
- **Generate video with open weights.** ComfyUI's official [Wan2.2 workflows](https://docs.comfy.org/tutorials/video/wan/wan2_2) run locally or on Comfy Cloud; Wan2.2 is Apache-2.0. <!-- c:M1-186 -->
- **Dub a video.** [pyVideoTrans](https://github.com/jianchang512/pyvideotrans) (GPL-3.0) transcribes, translates and re-voices a clip with open speech models, from a desktop app, a CLI or Docker. <!-- c:M1-188 -->
- **Script to finished video, as a service.** [Crreo](https://www.crreo.ai/) turns a script into a video with generated visuals, voice and music, using its own Kine and Kalo models. The free plan makes watermarked clips up to 30 seconds. Compare its per-asset price with your own in Homework 1. <!-- c:M1-184 c:M1-185 -->

## Models used in this module

We run these models on **Nebius Serverless AI**. The [Serverless AI Cookbook](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/README.md) has quick-start templates: each model link below opens the Nebius console with the model, container and recommended GPU already filled in. We'll deploy one in the next lesson. <!-- c:M1-174 c:M1-079 c:M1-081 c:M1-082 -->

| Task | Model (opens a pre-filled deploy form) | Recommended GPU | Licence |
|---|---|---|---|
| Text-to-image *(default)* | [Sana][deploy-sana] | L40S | [Apache-2.0 + Gemma terms](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers) |
| Text-to-image | [Qwen-Image][deploy-qwen-image] | H100 | [Apache-2.0](https://huggingface.co/Qwen/Qwen-Image) |
| Text-to-image | [Z-Image-Turbo][deploy-z-image] | H100 | [Apache-2.0](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |
| Image editing | [Qwen-Image-Edit-2511][deploy-qwen-edit] | H100 | [Apache-2.0](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) |
| Text-to-video | [Wan2.1-T2V-1.3B][deploy-wan21] | H100 | [Apache-2.0](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) |
| Image-to-video | [Wan2.2-I2V-A14B][deploy-wan22] | H100 | [Apache-2.0](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) |
| Text-to-speech | [Kokoro-82M][deploy-kokoro] | L40S | [Apache-2.0](https://huggingface.co/hexgrad/Kokoro-82M) |
| Text-to-speech | [Chatterbox][deploy-chatterbox] | H100 | [MIT](https://huggingface.co/ResembleAI/chatterbox) |
| Text-to-music | [ACE-Step 1.5][deploy-ace-step] | H100 | [MIT](https://huggingface.co/ACE-Step/Ace-Step1.5) |

<!-- c:M1-041 c:M1-048 c:M1-052 c:M1-054 c:M1-059 c:M1-061 c:M1-069 c:M1-072 c:M1-075 c:M1-076 -->

Templates deploy as **preemptible** by default: cheaper, but the GPU can be interrupted. Prices in eu-north1, 11 Nov 2026 ([Nebius pricing](https://docs.nebius.com/compute/resources/pricing)): an L40S template costs `$0.7492/h` preemptible or `$1.5484/h` on-demand `[calc: GPU + 8 vCPU + 32 GiB]`; an H100 costs `$4.50` per GPU-hour on-demand, while preemptible H100 is spot-priced from `$0.79` and can exceed on-demand. <!-- c:M1-085 c:M1-086 c:M1-087 c:M1-088 c:M1-089 -->

Our measured music run used the larger `ACE-Step/acestep-v15-xl-turbo-diffusers` checkpoint, also [MIT](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers).

[deploy-sana]: https://console.nebius.com/serverless/endpoint/create?image=cr.eu-north1.nebius.cloud%2Fe00gw2b7v3pxetvpy7%2Fsana-serve%3A660347b&targetPort=8000&platform=gpu-l40s-a&preset=1gpu-8vcpu-32gb&diskSize=500GiB&preemptible=true&auth=true
[deploy-qwen-image]: https://console.nebius.com/serverless/endpoint/create?image=vllm%2Fvllm-omni%3Av0.24.0&command=vllm%20serve%20Qwen%2FQwen-Image%20--omni%20--host%200.0.0.0%20--port%208000&targetPort=8000&platform=gpu-h100-sxm&preset=1gpu-16vcpu-200gb&diskSize=500GiB&preemptible=true&auth=true
[deploy-z-image]: https://console.nebius.com/serverless/endpoint/create?image=vllm%2Fvllm-omni%3Av0.24.0&command=vllm%20serve%20Tongyi-MAI%2FZ-Image-Turbo%20--omni%20--host%200.0.0.0%20--port%208000&targetPort=8000&platform=gpu-h100-sxm&preset=1gpu-16vcpu-200gb&diskSize=500GiB&preemptible=true&auth=true
[deploy-qwen-edit]: https://console.nebius.com/serverless/endpoint/create?image=vllm%2Fvllm-omni%3Av0.24.0&command=vllm%20serve%20Qwen%2FQwen-Image-Edit-2511%20--omni%20--host%200.0.0.0%20--port%208000&targetPort=8000&platform=gpu-h100-sxm&preset=1gpu-16vcpu-200gb&diskSize=500GiB&preemptible=true&auth=true
[deploy-wan21]: https://console.nebius.com/serverless/endpoint/create?image=vllm%2Fvllm-omni%3Av0.24.0&command=vllm%20serve%20Wan-AI%2FWan2.1-T2V-1.3B-Diffusers%20--omni%20--host%200.0.0.0%20--port%208000&targetPort=8000&platform=gpu-h100-sxm&preset=1gpu-16vcpu-200gb&diskSize=500GiB&preemptible=true&auth=true
[deploy-wan22]: https://console.nebius.com/serverless/endpoint/create?image=vllm%2Fvllm-omni%3Av0.24.0&command=vllm%20serve%20Wan-AI%2FWan2.2-I2V-A14B-Diffusers%20--omni%20--host%200.0.0.0%20--port%208000&targetPort=8000&platform=gpu-h100-sxm&preset=1gpu-16vcpu-200gb&diskSize=500GiB&preemptible=true&auth=true
[deploy-kokoro]: https://console.nebius.com/serverless/endpoint/create?image=cr.eu-north1.nebius.cloud%2Fe00gw2b7v3pxetvpy7%2Fkokoro-serve%3A660347b&targetPort=8000&platform=gpu-l40s-a&preset=1gpu-8vcpu-32gb&diskSize=500GiB&preemptible=true&auth=true
[deploy-chatterbox]: https://console.nebius.com/serverless/endpoint/create?image=cr.eu-north1.nebius.cloud%2Fe00gw2b7v3pxetvpy7%2Fchatterbox-serve%3A660347b&targetPort=8000&platform=gpu-h100-sxm&preset=1gpu-16vcpu-200gb&diskSize=500GiB&preemptible=true&auth=true
[deploy-ace-step]: https://console.nebius.com/serverless/endpoint/create?image=cr.eu-north1.nebius.cloud%2Fe00gw2b7v3pxetvpy7%2Facestep-serve%3Ad315ae1&targetPort=8000&platform=gpu-h100-sxm&preset=1gpu-16vcpu-200gb&diskSize=500GiB&preemptible=true&auth=true

## Your model decision record

For Emma's card:

```text
1. Modality and model:      text-to-image, Sana
2. GPU:                     L40S
3. Licence:                 https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers
                            Apache-2.0 + Gemma terms; card says "research purposes only": check before commercial use
4. Cost driver (estimate):  an image takes seconds on one GPU; a short video clip takes minutes
5. API to compare in HW1:   a hosted text-to-image API, e.g. Z-Image Turbo, listed at $5.00 per 1,000 images on
                            https://artificialanalysis.ai/image/leaderboard/text-to-image
```

Line 4 is an estimate from the measured table. We'll measure your real numbers in [1.5](05-measuring-costs.md) and check the estimate. If Emma's brief needed the clip, line 1 would become Wan2.2-I2V-A14B and line 2 an H100. That's the trade-off to defend. <!-- c:M1-126 -->

## Common failures and recovery

| Symptom | Cause | Fix |
|---|---|---|
| Licence terms aren't what you expected | You read the family, not the exact model card | Open the card of the exact model you deploy; read its licence |
| Out of memory | Model plus working memory exceed the GPU | Use the recommended GPU; lower resolution or frame count |
| Your hosted model is retired | The provider deprecated it | Keep a self-hosted fallback |
| A model repo is deleted, gated or relicensed | Hugging Face repos can change | Keep a copy of the weights in your own storage |
| Same seed, different character in a new scene | A seed only repeats one prompt | Use reference-image editing (1.2, Module 2) |

Next, generate your first media asset in [1.2 First Media Generation in a Few Minutes](02-first-generation.md).
