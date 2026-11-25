---
prev_url: ../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md
next_url: 02-tools.md
---
# 2.1 The Agent Loop: Deep Dive

Say you have a one-paragraph brief for a short video ad: a desk lamp, a person working from home, dusk. You want an agent that reads the brief and makes the first keyframe image. In this lesson we build that agent with [PydanticAI](https://pydantic.dev/docs/ai/) on Nebius Token Factory, then open up the loop the framework runs for us: who decides the next step, what goes over the wire each turn, and where it stops. You're done when the agent produces one image from one record, `events.jsonl` shows every model request and tool call in order, and you can say whether your Homework 2 brief needs an agent, a workflow or a hybrid.

Before you start: complete the **Setup** section of the [module README](../02-multi-agent-media-workflows/README.md) once.

Code: [`code/nb-agent-loop.ipynb`](code/nb-agent-loop.ipynb). Cell 0 sets `MODE`:

- `"fake"`: you play the LLM with a Python function, and the image tool draws a placeholder. No network, no credits.
- `"fake-gpu"`: a real LLM on Token Factory, placeholder image tool. Tokens only, a few cents [measure].

The image tool stays a placeholder until lesson 2.3. Here we care about the loop, not the pixels.

## Start with the finished result

The notebook opens with a saved run of the module's finished pipeline: the brief, a keyframe, a 5-second video, its trace and the cost of each step. It needs no credits. The rest of the module builds it one piece at a time; this lesson builds the first piece, an agent that calls one tool.

[placeholder: the saved reference run (`assets/reference-run/`) lands after the Homework 2 reference run. Until then the cell prints "No saved reference run in assets/ yet."]

## Agent, workflow or hybrid

Three ways to wire the same steps:

- **Workflow:** code decides the next step. `plan() → keyframe() → video()`, always in that order.
- **Agent:** the LLM decides the next step. It gets tools and picks which one to call, with which arguments, and when to stop.
- **Hybrid:** code fixes the stages, and the LLM decides inside a stage.

This module builds a hybrid: plan → keyframe → judge → video. The stages are fixed in code, because a video costs far more than a keyframe and we never want the model to skip the judge. Inside the keyframe stage, the model picks the tool call. In the common vocabulary, the fixed stages are prompt chaining, and judge → regenerate is evaluator-optimizer.

**Exercise:** in the trace at the end of this lesson, mark each decision as made by code or by the model. Find the tool call the model chose: that's the part that makes the example agentic.

## A minimal agent

### The tool

A tool is a typed Python function. PydanticAI reads its signature and docstring and shows them to the model as a schema. `ctx.deps` carries whatever the tool needs at run time; here it's the tracer that writes `events.jsonl`.

```python
async def generate_image(ctx: RunContext[Tracer], prompt: str, seed: int = 7) -> str:
    """Generate one keyframe image for the ad and return its file path."""
    path = ctx.deps.folder / f"keyframe-seed{seed}.png"
    path.write_bytes(placeholder_png(prompt, 832, 480, seed))
    ctx.deps.emit("tool_call", "generate_image", call_id=ctx.tool_call_id, asset_ref=str(path))
    return str(path)
```

### Be the LLM

Before any real model, we replace it with a function. [`FunctionModel`](https://pydantic.dev/docs/ai/api/models/function/) calls it with the conversation so far and the tools on offer, and it returns the next message. Ours asks for one tool call, then reports the result:

```python
def be_the_llm(messages: list, info: AgentInfo) -> ModelResponse:
    """A hand-written model: call generate_image, then report the result."""
    print(f"Model sees {len(messages)} messages, tools: {[t.name for t in info.function_tools]}")
    last = messages[-1].parts[-1]
    if last.part_kind == "tool-return":
        return ModelResponse(parts=[TextPart(f"Keyframe ready: {last.content}")])
    return ModelResponse(parts=[ToolCallPart("generate_image", {"prompt": brief.shot, "seed": 7})])
```

Run it and the agent calls the model twice: once with 1 message, once with 3 (prompt, tool call, tool result). The loop is plain code. The model only decides what comes next: a tool call keeps the loop going, text ends it.

**Exercise:** before running, predict the second message the function returns. Then change `seed` in the tool call and find the difference in the file name.

### Swap in the real LLM

One line changes: the model.

```python
if MODE == "fake-gpu":
    agent = make_agent(token_factory_model())  # model ID pinned in models.py
    result = await agent.run(prompt, deps=tracer)
    print(result.output)
```

`token_factory_model()` in `tools.py` points PydanticAI's OpenAI model at Token Factory's base URL. It reads `NEBIUS_API_KEY` from `code/.env` and asks for it the first time. The printout has the same shape as before; now the LLM chooses the prompt and the seed.

### What goes over the wire

With `MODE = "fake-gpu"`, one cell records the raw HTTP requests and prints the second one. Two things to notice:

- `tools` is sent on **every** turn: the schema of `generate_image`, built from the signature and docstring.
- `messages` holds the **whole** history: the brief, the model's tool call, the tool result. Each turn resends everything before it, so input tokens grow with every step. In our run: request 1 [measure] input tokens, request 2 [measure].

## Step through a run with agent.iter()

`agent.run()` returns the answer and hides how many requests it made. `agent.iter()` yields each node of the run as it happens:

```python
async with agent.iter(prompt, deps=tracer) as run:
    async for node in run:
        print(type(node).__name__)
print(run.result.output)
```

```text
UserPromptNode
ModelRequestNode
CallToolsNode
ModelRequestNode
CallToolsNode
End
```

`ModelRequestNode` costs tokens. `CallToolsNode` runs the tools the model asked for, and a tool may cost GPU time. Count both and you know what a run costs before you look at a bill.

## Stop conditions

A model that never stops asking for images:

```python
def never_stops(messages: list, info: AgentInfo) -> ModelResponse:
    """Ask for another image every turn, with a new seed."""
    return ModelResponse(
        parts=[ToolCallPart("generate_image", {"prompt": brief.shot, "seed": len(messages)})]
    )


try:
    await make_agent(FunctionModel(never_stops)).run(
        prompt, deps=tracer, usage_limits=UsageLimits(tool_calls_limit=3)
    )
except UsageLimitExceeded as exc:
    print("Stopped:", exc)
```

Limits live in code, not in the prompt. [`UsageLimits`](https://pydantic.dev/docs/ai/core-concepts/agent/) has `request_limit`, `tool_calls_limit`, token limits and `cost_limit`; a run over a limit raises `UsageLimitExceeded`.

| Limit | Counts | Misses |
| --- | --- | --- |
| `request_limit` | model requests | GPU spend |
| `tool_calls_limit` | tool calls; checked before the tools run, so none of a parallel batch runs if it would go over | GPU spend |
| `output_tokens_limit` and the other token limits | LLM tokens | GPU spend |
| `cost_limit` | LLM cost the framework estimates, best effort | GPU spend; [verify: whether it's computed for Token Factory models] |

Every row misses the same thing: a GPU tool call adds nothing to the framework's usage. The image or video it starts can cost more than all the tokens of the run. Lesson 2.5 adds a GPU budget guard of our own.

## Where state lives

The conversation lives in memory and dies with the process. Save it to the run folder, and a new process can continue it:

```python
path = save_messages(tracer.folder, result.all_messages())
print(path, path.stat().st_size, "bytes")

restored = load_messages(tracer.folder)
again = await agent.run("Make one more with seed 8.", message_history=restored, deps=tracer)
print(again.output)
```

`messages.json` is only the conversation. Lesson 2.4 adds what else has to survive a crash: which GPU request is still running, and where.

## Picking an LLM for tool calling

The course default is pinned in `code/models.py`. We picked it by measuring tool calling on Token Factory, not from a model card:

| Model | Valid tool calls | Tokens per run | Latency |
| --- | --- | --- | --- |
| [measure: author bench, 10 runs each on 3 candidates] | | | |

To try another model, change `LLM` in `models.py`. Token Factory's [function calling docs](https://docs.tokenfactory.nebius.com/ai-models-inference/function-calling) describe the API; check the model list for which models support tools.

## Trace events

Printed nodes vanish when the notebook closes. The trace stays. `run_traced()` in `tracing.py` drives `agent.iter()` and writes one `model_request` event per model response; the tool writes its own `tool_call` event. Both carry the tool call ID, so you can link every tool call to the request that asked for it:

```python
tracer = Tracer(new_run_id())
await run_traced(agent, prompt, tracer=tracer, deps=tracer)
```

```text
model_request  generate_image   call_id=None parent=- tokens_in=[measure]
tool_call      generate_image   call_id=call_... parent=3f2a91c0 tokens_in=None
model_request  text             call_id=None parent=- tokens_in=[measure]
```

Steps = model requests + tool calls: 3 here. `check_trace.py` checks a trace against the field contract and prints its totals:

```bash
uv run python check_trace.py runs/02-multi-agent-media-workflows/<run_id>/events.jsonl
```

Every lesson in this module writes the same events, and Homework 2 asks for them.

## Notes

- `run_traced()` also saves `messages.json` after every model response, so a crash loses at most the step in flight.
- Notebook cells call `await agent.run(...)`. Jupyter runs top-level `await`; `run_sync` inside a notebook needs `nest_asyncio` ([troubleshooting](https://pydantic.dev/docs/ai/overview/troubleshooting/)).

Next: [2.2 Tools and Structured Outputs](02-tools.md), where the tool gets typed arguments, retries and a cache.

<!-- nav:start -->
[← Previous: 05 - Measuring Startup, Load and Generation Costs](../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md) · [Next: 2.2 Tools and Structured Outputs →](02-tools.md)
<!-- nav:end -->
