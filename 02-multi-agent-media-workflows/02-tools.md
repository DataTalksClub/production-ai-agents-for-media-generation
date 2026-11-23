---
video_url: ""
prev_url: 01-agent-loop.md
next_url: 03-gpu-tools.md
---
# 2.2 Tools and Structured Outputs

An agent that makes ad keyframes will, sooner or later, call its image tool with a size the model doesn't support, wait less than the image takes, or ask for the same image twice. Each of those can cost GPU time for nothing. In this lesson we design the image tool so none of them does: the signature becomes the schema the model sees, retries are split into three kinds, a timeout is set against the real generation time, and a spec hash reuses a finished image instead of paying for it again. You're done when the model fixes an invalid call after one validation retry, and a second identical `generate_image` call is a cache hit with no GPU time and no cost.

Before you start: complete the **Setup** section of the [module README](../02-multi-agent-media-workflows/README.md) once.

Code: [`code/nb-tools.ipynb`](code/nb-tools.ipynb), one cell per demo. The tool itself lives in the shared [`code/tools.py`](code/tools.py); lessons 2.3 to 2.5 and Homework 2 import it. `MODE = "fake"` runs everything offline with a scripted model and a placeholder image backend; `"fake-gpu"` uses a real LLM for the validation and plan demos. No GPU in this lesson.

## The tool signature is the schema

The model never sees your Python. It sees a JSON schema built from the signature and the docstring, resent every turn:

```python
Size = Literal["832x480", "480x832", "1024x1024"]
Prompt = Annotated[str, Field(min_length=10, max_length=600)]
```

```python
async def generate_image(
    ctx: RunContext[ToolDeps],
    prompt: Prompt,
    seed: int = SANA.seed,
    size: Size = "832x480",
) -> AssetRef:
    """Generate one keyframe image and return a reference to the PNG.

    Args:
        prompt: One scene, described visually. No on-screen text.
        seed: Change it to get a different image for the same prompt.
        size: WIDTHxHEIGHT in pixels. 832x480 matches the video frame.
    """
```

The first notebook cell prints the schema the model receives: `size` arrives as an `enum` of three values, `prompt` with its length bounds, each argument with the sentence from the docstring. The docstring is a prompt; write it for the model. Constraints in the types are checked by PydanticAI before the function runs, so a bad argument never reaches the GPU.

## Three kinds of retry

"Retry on error" pays twice when the error happened after the GPU started. So `tools.py` keeps three kinds apart:

| Kind | Example | What we do | What it costs |
| --- | --- | --- | --- |
| Validation retry | `size="1000x1000"` | Error goes back to the model; it fixes the call | One more LLM turn, no GPU |
| Transport retry | Connect timeout, `502`/`503` while the Endpoint loads its weights | Back off and retry, at most 3 times | Nothing: the request never reached the model |
| Uncertain submission | Read timeout, dropped connection after sending | Never resend blindly | Maybe a GPU run you can't see. Lesson 2.4 handles it with a ticket |

The validation retry, with a scripted model that gets the size wrong first:

```python
model = (
    scripted_model(
        call("generate_image", prompt=brief.shot, size="1000x1000"),  # not an allowed size
        call("generate_image", prompt=brief.shot, size="832x480"),
        "Keyframe ready.",
    )
    if MODE == "fake"
    else token_factory_model()
)
```

```text
model_request  generate_image   cache_hit=False t_gen_s=None cost=0.0
model_request  generate_image   cache_hit=False t_gen_s=None cost=0.0
tool_call      generate_image   cache_hit=False t_gen_s=0.2 cost=0.0
model_request  text             cache_hit=False t_gen_s=None cost=0.0
GPU requests received: 1
```

Two model requests, one tool call, one GPU request: the bad call never ran. With `MODE = "fake-gpu"` the real model gets "Use size 1000x1000" in the prompt; in our run it needed [measure] retries to fix the call.

## Tool timeouts

PydanticAI treats a [tool timeout](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-timeout) as a retryable failure, counted toward the tool's retries. For a GPU tool that's a trap: the client stops waiting, the GPU keeps working, and the retry sends a second request. The demo makes a "GPU" that takes 3 s per image and tries two timeouts:

```python
for timeout in (1, 10):
    deps = new_deps(sleep_s=3)  # this "GPU" takes 3 s per image
    agent = Agent(
        scripted_model(*[call("generate_image", prompt=brief.shot)] * 2, "Keyframe ready."),
        deps_type=ToolDeps,
        tools=[Tool(generate_image, timeout=timeout, max_retries=1)],
    )
```

```text
Gave up: Tool 'generate_image' exceeded max retries count of 1.
timeout=1 s -> GPU requests: 2
timeout=10 s -> GPU requests: 1
```

Two strategies, by how long the tool runs:

- **Short tools** (a Sana image, seconds): set the timeout above the measured p95 generation time. Sana on the course L40S: p95 [measure: 2.3], so the course uses [proposed: 2 × p95].
- **Long tools** (a video, minutes): don't wait inside the tool at all. Submit, return, and check back later. That's lesson 2.4.

## Cache keys: reuse a completed result

The model, a retry or a rerun of your script will call the tool with the same arguments again. The spec hash names everything that changes the output:

```python
    spec = {
        "model": model,
        "model_revision": model_revision,
        "image": image,  # container image of the Endpoint, as in Module 1 config
        "prompt": prompt,
        "seed": seed,
        "params": params,
        "inputs": [file_sha256(p) for p in inputs],
        "adapter": adapter,
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
```

The image file is named after it. Before any GPU call, the tool checks whether that file exists:

```python
    if store.exists(name):  # same spec as a finished generation: reuse it
        return AssetRef(cache_hit=True, **ref)
```

Call the tool twice with the same spec:

```text
model_request  generate_image   cache_hit=False t_gen_s=None cost=0.0
tool_call      generate_image   cache_hit=False t_gen_s=0.2 cost=0.0
model_request  generate_image   cache_hit=False t_gen_s=None cost=0.0
tool_call      generate_image   cache_hit=True  t_gen_s=None cost=0.0
model_request  text             cache_hit=False t_gen_s=None cost=0.0
GPU requests received: 1
```

The second `tool_call` is a cache hit: `cost_usd` 0, no `t_gen_s`. If you did Module 1, this is `request_id` and `cached_image` from `image.py`, extended with the full hash and the input files.

**Exercise:** the notebook flips one byte of a keyframe and hashes both files as inputs of a video spec. Explain why the hashes differ although the file name and every argument are the same. That's why the video tool in 2.4 hashes the keyframe's bytes, not its name.

What the cache doesn't cover: a request that is still running (that's a ticket, 2.4), a crash in the middle of a generation, and two processes asking for the same spec at once (Module 3).

## Return references, not bytes

The tool returns an `AssetRef`: a URI, the spec hash, the size, the workload and the generation time. The PNG goes to the run folder, and to your bucket when `BUCKET` is set in `code/.env`. The model gets a line of text; the bytes never enter the conversation, so they never cost tokens on every later turn. PydanticAI's own [image generation](https://pydantic.dev/articles/image-generation) does the opposite: the image comes back as `BinaryImage` and stays in the history, which is handy for one image and costly for many. In this module, only the judge (2.5) loads the bytes, once, in its own call.

## Structured outputs for LLM steps

The plan step turns the brief into two prompts. Free text would need parsing; a typed output can't come back malformed:

```python
class Plan(BaseModel):
    """The planner's output: one prompt per generation step."""

    image_prompt: str = Field(description="Prompt for one keyframe image that shows the shot")
    motion_prompt: str = Field(description="What moves in the 5-second clip made from it")
```

```python
planner = Agent(
    model,
    output_type=Plan,
    instructions="Turn the ad brief into a keyframe prompt and a motion prompt.",
)
plan = (await planner.run(brief.model_dump_json())).output
```

Lesson 2.5 reuses the same idea for the judge's verdict, with one twist: it parses the reply itself, so you can see what happens when it doesn't parse.

## Notes

- A budget check doesn't belong inside the tool. Lesson 2.5 checks the GPU budget before each paid generation, in the loop.
- Transport retries in `SanaEndpoint.generate()` cover only failures where the request never reached the model. Which errors the Sana template returns while loading: [verify: T4].

Next: [2.3 GPU Workloads as Agentic Tools](03-gpu-tools.md), where the placeholder backend becomes a real Endpoint.

<!-- nav:start -->
[← Previous: 2.1 The Agent Loop: Deep Dive](01-agent-loop.md) · [Next: 2.3 GPU Workloads as Agentic Tools →](03-gpu-tools.md)
<!-- nav:end -->
