# Module 2: Agents That Use GPUs

Welcome to Module 2 of Production AI Agents for Generative Media! This week we make it intelligent: give an agent GPU workloads as tools, keep the agent on CPU, and build a planner, generator and reviewer loop that regenerates an asset instead of accepting the first output.

> [!NOTE]
> This module builds on Module 1: the Endpoints, Jobs, Object Storage bucket and cost terms from [1.4](../01-generative-media-workloads#14-serving-vs-batch-inference-the-same-generation-for-different-use-cases) and [1.5](../01-generative-media-workloads#15-measuring-startup-load-and-generation-costs). The agent's LLM runs on [Nebius Token Factory](https://docs.tokenfactory.nebius.com/quickstart), an OpenAI-compatible API.

> [!WARNING]
> Make sure you stop or delete your resources (Endpoints, Jobs) when you finish the module to avoid charges. An Endpoint bills while it's RUNNING, even when the agent isn't calling it. Video is the most expensive asset: keep the fast path to two videos or fewer.

* Video Link (Coming Soon)
* Slides (Coming Soon)

## Learning Objectives

By the end of this module, you will be able to:

* Explain the loop an agent framework runs for you (model request → tool calls → tool results → output) and step through it with PydanticAI
* Choose between agent, workflow and hybrid architectures for a media task
* Design tools with typed inputs and outputs, validation, retries, timeouts and idempotency
* Wrap GPU workloads as agent tools: inference on an Endpoint, batch inference or fine-tuning as a Job, called directly or over MCP
* Handle long-running tools: submit, poll, resume, without paying for the same GPU work twice
* Build a planner, generator and reviewer loop with attempt caps, gates and metrics, and compare one agent with role prompts against several agents

Default path for the whole module: one [PydanticAI](https://pydantic.dev/docs/ai/) agent on Token Factory, the [Sana](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-sana/README.md) Endpoint from Module 1 for images, [Wan2.2-I2V-A14B](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-wan22-i2v-a14b/README.md) for video. Every model request, tool call and check writes one event to `events.jsonl` in the run folder. Every lesson script also runs with `--fake`: fully offline, with a scripted model and placeholder assets, so you can read the code path without credits.

## 2.1 The Agent Loop: Deep Dive

PydanticAI from the first line. We don't write the loop by hand; we open it up and look at every step.

* Agent, workflow or hybrid, at a high level:
  * Workflow (fixed pipeline): the same steps every time. Cheaper, easier to test.
  * Agent: decides the steps at runtime. Needed when the brief varies.
  * Hybrid: a workflow skeleton (plan → keyframe → judge → video) with the agent at decision points. This module builds a hybrid: the 2.5 judge loop is the decision point.
  * When a framework pays off and when it gets in the way. Graph frameworks ([pydantic-graph](https://pydantic.dev/docs/ai/api/pydantic_graph/graph_builder/), LangGraph) as pointers.
* A minimal agent: `Agent`, a model on Token Factory through the [OpenAI provider](https://pydantic.dev/docs/ai/models/openai/) with a custom `base_url`, two tools (`write_prompt`, `generate_image`) and a typed `output_type`
* What `agent.run()` does: user prompt → model request → tool calls → tool results → model request … until the output validates
* Step through the same run with [`agent.iter()`](https://pydantic.dev/docs/ai/agents/#iterating-over-an-agents-graph) and print every node: `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, `End`
* Stop conditions: a valid `output_type`, [`UsageLimits`](https://pydantic.dev/docs/ai/api/pydantic-ai/usage/) (`request_limit`, `tool_calls_limit`, token limits) and your own budget check
* Where state lives: message history, `deps`, run usage. What survives a restart: only what you persist (message history, tool results, asset references). [Durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) is the framework answer; this module persists by hand
* Picking an LLM for tool calling: run the same task several times per model and count valid tool calls before you choose
* Trace: one event per model request and per tool call, linked by the tool call ID. Graph nodes are printed for reading, not counted as steps, and carry no cost.

Done when: the agent produces one image from one record, `events.jsonl` shows every model request and tool call in order, and you can say whether your Homework 2 brief needs an agent, a workflow or a hybrid.

## 2.2 Tools and Structured Outputs

* The tool signature is the schema: type hints and Pydantic models become the JSON schema the model sees; the docstring becomes the description
* Three kinds of retry. Keep them apart:
  * Validation retry: the arguments are wrong. The error goes back to the model so it can fix the call. No GPU work happened. Cap retries per tool.
  * Transport retry: the request never reached the GPU (connection error, Endpoint not ready). Retry with backoff.
  * Uncertain submission: the request may have been accepted, but you didn't get the answer (timeout after send). Don't resend; reconcile first (2.4).
* A PydanticAI [tool timeout](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-timeout) counts as a retryable failure: the model is told the call timed out and can call the tool again. So a GPU tool that waits for a long generation can be called twice. Long work goes through submit and poll instead (2.4).
* Idempotency key: a hash of everything that changes the output: model, pinned revision, container image, prompt, seed, parameters, the content hash of every input file (such as the keyframe for image-to-video) and any adapter revision. If the asset already exists in the bucket, return it.
* Return references (bucket URL and metadata), not bytes
* Structured outputs for LLM steps: a Pydantic model as `output_type`

Done when: the model fixes an invalid call after one validation retry, and a second identical `generate_image` call is a cache hit in the trace with no GPU time and no cost.

## 2.3 GPU Workloads as Agentic Tools

* Control plane vs compute/data plane: the agent runs on CPU; GPU work runs as an Endpoint or a Job; inputs and outputs move through Object Storage. The agent itself never needs a GPU.
* A GPU tool is not only a model call. Three kinds:
  * Inference on a running Endpoint: `generate_image`. Seconds, synchronous.
  * Batch inference as a Job: many assets in one run, as in [1.4](../01-generative-media-workloads#14-serving-vs-batch-inference-the-same-generation-for-different-use-cases)
  * Fine-tuning as a Job: the tool submits training and returns the adapter's bucket path when the Job completes. Built in [3.6](../03-scaling-reliability-cost#36-fine-tune-a-model-on-a-new-style-optional-advanced).
* How the agent calls a tool: a plain Python function (default), an OpenAI-compatible HTTP call, or an [MCP](https://modelcontextprotocol.io/) server when several clients share the same tools. PydanticAI connects to MCP servers as a [client](https://pydantic.dev/docs/ai/mcp/client/). MCP is optional.
* Credentials: the tool holds Endpoint tokens and bucket keys. The model never sees them.
* Lifecycle per tool: readiness wait before the first call, a timeout on every call and every Job, teardown at the end of the run
* Default path: the Sana Endpoint as `generate_image`. Also try: the same tools behind an MCP server, used from your agent and from an IDE agent.

Done when: your agent calls a GPU tool, and each trace event records the workload type (Endpoint or Job), its ID and the generation time.

## 2.4 Long-running Tools: Submit, Poll, Resume

* A video or a fine-tuning Job takes minutes to hours. A blocking tool call stalls the agent and hits the tool timeout, which triggers a retry (2.2).
* Pattern: `submit_video(spec) -> ticket` and `get_status(ticket) -> state`. The tool returns as soon as the GPU accepts the request.
* Default path: the Wan2.2-I2V-A14B Endpoint through the asynchronous [Videos API](https://docs.vllm.ai/projects/vllm-omni/en/stable/serving/videos_api/) of vLLM-Omni, which the template runs:
  * `POST /v1/videos` creates a video request and returns its video ID with status `queued`
  * `GET /v1/videos/{video_id}` returns the status: `queued`, then `in_progress`, then `completed` (or a failure)
  * `GET /v1/videos/{video_id}/content` downloads the video. Copy it to your bucket, then `DELETE /v1/videos/{video_id}` to remove it from the Endpoint.
  * The template's own example uses `/v1/videos/sync`, which blocks until the video is ready. Don't use it from an agent tool.
* Two IDs, not one: the Endpoint ID (the deployment) and the video ID (one request on it). The video ID only means something on that Endpoint. The template creates a preemptible Endpoint; if it restarts, the video ID may be gone.
* Also try: the same tool as a Nebius Job. Then the Job ID is both deployment and request, status follows the Job lifecycle (QUEUED → RUNNING → COMPLETED or FAILED), the Job writes to the bucket, and it stops by itself.
* The ticket: stored in the bucket next to the run, one per spec hash. It holds the backend, the Endpoint or Job ID, the video ID, the state and the asset reference.
  * `intent`: written before you submit
  * `pending`: the video ID is saved, right after the submit returns
  * `completed` or `failed`: terminal
  * `unknown`: you can't tell whether the request exists (a crash between submit and save, or a video ID the Endpoint no longer knows)
* Reconcile before you retry: on `intent` or `unknown`, look for the request on the Endpoint first. Resubmit only when it isn't there, and mark the trace event as a possible duplicate.
* What this guarantees: once the video ID is saved, a retry or a restart never starts a second video. A crash in the gap between submit and save can start one extra video; the trace shows it.
* Restart boundaries to test: kill before submit, after submit but before the video ID is saved, while pending, and after completed but before the copy to the bucket
* In PydanticAI: plain submit and poll tools (default), or [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/#external-tool-execution) that end the run with pending calls and resume it once the results arrive
* Polling with backoff vs callbacks and events. Polling is enough here: no public callback URL needed.

Done when: you kill the agent while the video is pending, resume it with the run ID, and the trace shows one video generation; and you can say what your trace shows after a kill between submit and save.

## 2.5 Planner, Generator, Reviewer: the Judge Loop

* Roles:
  * Planner (LLM): turns the brief into a shot description and prompts; decides what to regenerate and how
  * Generator (GPU tools): keyframe image, video
  * Reviewer (VLM judge): scores the keyframe against the brief and returns a verdict
* Judge output, typed: a verdict, a score, reasons and a fix (new seed or prompt change)
  * `PASS`: the keyframe may go to video
  * `RETRY`: generate again with the fix. Counts as a generation attempt.
  * `WEAK`: stop. Keep the keyframe and the reasons for diagnosis. A `WEAK` keyframe never passes the video gate.
* A judge reply that doesn't parse is not a verdict: ask the judge again, with its own small cap. That's a judge call, not a new generation. If it still doesn't parse, stop with that reason. It never counts as `PASS`.
* Attempts: a cap per asset type. Cheap assets get more attempts than expensive ones. At the cap without `PASS`, stop with a reason; no video.
* Gates:
  * Deterministic checks first: the file exists, resolution, duration, frame count
  * No video until its keyframe is `PASS`: spend the cheap GPU before the expensive one
* Three kinds of check, each for what it can see:
  * Deterministic checks: file, size, duration, frames
  * Image judgement: the VLM judges the keyframe against the brief
  * Human playback: you watch the final video. The judge saw a still frame; motion and artifacts are yours to check.
* Prove recovery with a fixture: a keyframe made to break the brief on purpose (the cup missing, readable text), labelled with the verdict you expect. The judge must return `RETRY` on it, and the loop must recover to `PASS`. Your live run may pass on the first attempt; that's a valid result.
* Metrics per run: attempts per asset, accept rate (keyframes with `PASS` ÷ keyframes judged), generation time spent on rejected keyframes, judge cost, cost per accepted video
* Multi-agent or one agent with role prompts? Build both on the same brief and compare:
  * One agent: one PydanticAI agent whose instructions cover all three roles
  * Multi-agent: a planner agent that hands review to a separate reviewer agent through a tool ([agent delegation](https://pydantic.dev/docs/ai/guides/multi-agent-applications/))
  * Measure for each: steps (model requests and tool calls), tokens, generation time, cost and accept rate

Done when: the judge returns `RETRY` on the fixture and the loop recovers to `PASS`, your live run shows a verdict for every keyframe, and you have one table with steps and cost for both designs.

## Homework 2: Media Agent

Build a media agent for a short coffee ad video: plan → keyframe → judge → regenerate → video, with a full trace of every step.

* The agent judges the keyframe and regenerates on `RETRY` instead of accepting the first output. Show recovery on a fixture keyframe; your live run may pass on the first attempt.
* Video runs as a long-running tool with a ticket
* Tools called directly or over MCP, your choice
* Report the cost per video, split by step, and whether you chose an agent, a workflow or a hybrid for this brief, and why

Questions and submission: [homework.md](../cohorts/2026/homework/02-agents-that-use-gpus/homework.md).

# Community Notes

Did you take notes? You can share them here:

* Add your notes here!
