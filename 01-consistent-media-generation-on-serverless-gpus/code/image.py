"""Generate images through an OpenAI-style image Endpoint (such as the Sana template) from your laptop."""
from __future__ import annotations

import base64
import json
import time
import urllib.error
from pathlib import Path

import common


DEFAULT_MODEL = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"


class ImageGenerator:
    """Client for an OpenAI-style image Endpoint: POST /v1/images/generations, save the PNG."""

    def __init__(self, endpoint: str, token: str | None, model: str = DEFAULT_MODEL):
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.model = model
        self.last_s: float | None = None

    def generate_image(
        self,
        prompt: str,
        *,
        out: str | Path = "runs/first-image.png",
        seed: int | None = 7,
        size: str = "512x512",
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        negative_prompt: str = "",
        timeout: int = 300,
    ) -> Path:
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
        }
        if seed is not None:  # no seed: the server picks a random one, so every call differs
            request_data["seed"] = seed
        start = time.perf_counter()
        try:
            payload = common.request(
                self.endpoint,
                self.token,
                "/v1/images/generations",
                request_data,
                timeout=timeout,
            )
        except urllib.error.HTTPError as exc:
            # Surface the server's message (e.g. 400 "this endpoint serves ...").
            detail = exc.read(500).decode(errors="replace") if exc.fp else ""
            raise RuntimeError(f"HTTP {exc.code} from /v1/images/generations: {detail}") from exc
        self.last_s = time.perf_counter() - start
        print(f"Generated in {self.last_s:.1f} s")
        png = base64.b64decode(payload["data"][0]["b64_json"], validate=True)
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Expected a PNG response")
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        return path.resolve()


def request_id(**request):
    """Short stable id of a generation request (model, prompt, seed, settings): same request, same id."""
    import hashlib

    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:8]


def image_comparison(paths, height=320):
    """Build one horizontal row, caption each image with its file name, and return JPEG bytes for the notebook."""
    import io

    from PIL import Image, ImageDraw

    pad, caption_h = 12, 28
    tiles = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        tiles.append(img.resize((round(img.width * height / img.height), height)))
    width = sum(t.width for t in tiles) + pad * (len(tiles) + 1)
    sheet = Image.new("RGB", (width, height + caption_h + 2 * pad), "white")
    draw = ImageDraw.Draw(sheet)
    x = pad
    for path, tile in zip(paths, tiles):
        sheet.paste(tile, (x, pad))
        draw.text((x, pad + height + 6), Path(path).name, fill="black")
        x += tile.width + pad
    preview = io.BytesIO()
    sheet.save(preview, format="JPEG", quality=80)
    return preview.getvalue()
