---
video_url: "https://youtu.be/uipMy9uKWtM"
prev_url: 01-environment-preparation.md
next_url: 03-gpu-tools.md
---

# 2.2 Creating the Media Agent

Now let's create an agent.

For that, install [PydanticAI](https://pydantic.dev/docs/ai/):

```bash
uv add pydantic-ai
```

I'll use Nebius Token Factory for the agent, but you can use any other LLM provider supported by PydanticAI.

Create an API key in [Token Factory](https://docs.tokenfactory.nebius.com/quickstart) and add it to `.env`:

```bash
NEBIUS_API_KEY=...
```

Token Factory is OpenAI-compatible, so we use the OpenAI model class from PydanticAI and point it to Token Factory. See more examples in the [Token Factory cookbook](https://github.com/nebius/token-factory-cookbook/blob/main/agents/README.md).

We continue in the same notebook as in the previous lesson, so `sana_generator` and `wan_generator` are already there.

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    model_name='zai-org/GLM-5.3-Flash',
    provider=OpenAIProvider(
        base_url='https://api.tokenfactory.nebius.com/v1/',
        api_key=os.environ['NEBIUS_API_KEY']
    )
)
```

Now the agent. We give it the prompt and the two tools from the previous lesson:

```python
from pydantic_ai import Agent

media_instructions = """
You help user create a video advertisement based on the idea they give you. 

First generate an image and then use the image to generate a video. The 
video should be 5 seconds with 16 fps unless user asks for a different 
configuration.
""".strip()

media_agent = Agent(
    model=model,
    tools=[
        sana_generator.generate_image_tool,
        wan_generator.generate_video_tool
    ],
    instructions=media_instructions
)
```

Let's run it:

```python
result = await media_agent.run('a desk lamp, a person working from home, dusk')
print(result.output)
```

We get:

```
Your ad is ready! Here's what I created:

**The scene:** A warm, cinematic dusk shot — a sleek desk lamp casting amber light over a person working from home on their laptop, with deep blue twilight glowing through the window behind them.

**The video (5s @ 16fps):**
- A slow cinematic push-in toward the glowing lamp
- The person's hands typing gently on the laptop
- Subtle light flicker from the lamp with dust particles drifting through the beam
- Dusk deepening outside the window for that calm, focused evening-work mood

**Files:**
- 🖼️ Image: `out\quickstart\qwen_20261126-184233_b1k.png`
- 🎬 Video: `out\quickstart\wan_20261126-184315_s4z.mp4`

Want a different vibe (brighter, more minimalist, a specific lamp style) or a different camera movement? I can regenerate either the image or the video.
```

It worked, but we don't see what happened inside. Which tools did the agent call, and with which arguments?

For that, we can pass a callback that prints every tool call:

```python
from pydantic_ai.messages import FunctionToolCallEvent

class NamedCallback:

    def __init__(self, agent):
        self.agent_name = agent.name

    async def print_function_calls(self, ctx, event):
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await self.print_function_calls(ctx, sub)
            return

        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            print(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)
```

Let's run the agent again with it:

```python
callback = NamedCallback(media_agent)

result = await media_agent.run(
    'a desk lamp, a person working from home, dusk',
    event_stream_handler=callback
)
```

Now we see what's inside:

```
TOOL CALL (media_agent): generate_image_tool({"prompt": "Cinematic advertisement still: a modern minimalist desk lamp glowing warm amber as the hero of the scene, on a tidy wooden desk where a person works from home at dusk. Soft blue-purple twilight light through a large window behind them, laptop open, coffee mug steaming, notebook and plant on desk. Shallow depth of field, warm-cool color contrast, moody product photography, photorealistic, 35mm lens, cozy atmosphere"})
TOOL CALL (media_agent): generate_video_tool({"duration": 5, "fps": 16, "image_path": "out\\quickstart\\qwen_20261126-184847_ezg.png", "prompt": "Slow cinematic camera push-in toward the glowing desk lamp. The person types gently on the laptop, steam rises softly from the coffee mug, warm lamp light subtly pulses, dusk light outside the window slowly deepens. Calm, cozy, premium advertisement mood"})
```

Now we see the loop: the agent first calls `generate_image_tool`, gets the path to the image, and then passes this path to `generate_video_tool`.

Note that it didn't use our short idea as is. It wrote a much more detailed prompt for each model. And it set `duration` and `fps` from the instructions.

## Checking the image

There's a problem. Let's look at the image from this run (`qwen_20261126-184847_ezg.png`): there's a desk lamp, a window, a laptop, a mug - but no person. The image model ignored that part of the prompt. And the agent didn't notice: it got a path to the image, not the image itself, so it went on and made a video from it.

We want the agent to look at the image before generating video.

The model we use (`zai-org/GLM-5.3-Flash`) can read images, so we only need to give the image to it. For that, we create a tool that returns the image. In PydanticAI, we can do it with `ToolReturn`: `return_value` is the result of the tool call, and everything in `content` is sent to the model too - including images:

```python
from pathlib import Path
from pydantic_ai import BinaryContent, ToolReturn

def view_image(path: str) -> ToolReturn:
    """
    Look at an image. Use it to check a generated image before making a video from it.
    """
    return ToolReturn(
        return_value=f"Image {path} is attached",
        content=[BinaryContent(data=Path(path).read_bytes(), media_type="image/png")],
    )
```

Now we add this tool to the agent and update the instructions:

```python
media_instructions = """
You help user create a video advertisement based on the idea they give you. 

First generate an image and then use the image to generate a video. The 
video should be 5 seconds with 16 fps unless user asks for a different 
configuration.

Before generating the video, make sure that the image contains what you expect.
If it doesn't, regenerate it with a different prompt.
""".strip()

media_agent = Agent(
    model=model,
    tools=[
        sana_generator.generate_image_tool,
        wan_generator.generate_video_tool,
        view_image,
    ],
    instructions=media_instructions
)
```

Let's run it:

```python
result = await media_agent.run(
    'a desk lamp, a person working from home, dusk',
    event_stream_handler=callback
)
```

```
TOOL CALL (media_agent): generate_image_tool({"prompt": "A cozy home office at dusk, warm light from a stylish modern desk lamp glowing softly on a wooden desk, a person seen from behind working on a laptop, large window showing deep blue twilight sky with city lights beginning to twinkle, cinematic warm and moody lighting, shallow depth of field, photorealistic, advertising photography style"})
TOOL CALL (media_agent): view_image({"path": "out\\quickstart\\qwen_20261126-190433_dyr.png"})
TOOL CALL (media_agent): generate_video_tool({"duration": 5, "fps": 16, "image_path": "out\\quickstart\\qwen_20261126-190433_dyr.png", "prompt": "Slow cinematic dolly-in toward a person working at a home office desk at dusk. The desk lamps glow warmly, the person gently types on the laptop, city lights twinkle and clouds drift slowly outside the window, subtle warm light flicker, calm and cozy atmosphere"})
```

Now the agent calls `view_image` after generating the image and only then generates the video. This time the image was good, so it went to the video right away. If something was missing, it would generate a new image first.

Note that we don't recreate `callback`: it only needs the name of the agent, and it's the same.

<!-- nav:start -->
[← Previous: 2.1 Environment Preparation](01-environment-preparation.md) · [Next: 2.3 GPU Workloads as Agentic Tools →](03-gpu-tools.md)
<!-- nav:end -->
