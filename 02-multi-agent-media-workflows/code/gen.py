import base64
import io
import random
import string
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image


def base64_decode(b64: str) -> Image.Image:
    png = base64.b64decode(b64)
    return Image.open(io.BytesIO(png))


def save_image(image: Image, filename: str) -> str:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def save_video(video: bytes, filename: str) -> str:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(video)
    return str(path)


def generate_random_filename(prefix: str = "qwen", ext: str = "png"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=3))
    return f"out/quickstart/{prefix}_{ts}_{rand}.{ext}"


class ImageGenerator:

    def __init__(self, endpoint: str, model: str, token: str | None = None):
        self.endpoint = endpoint
        self.model = model
        self.token = token

    def generate_image(self, prompt: str) -> Image.Image:
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "size": "1024x1024",
            "seed": 7,
            "response_format": "b64_json",
            "num_inference_steps": 20,
        }

        response = requests.post(
            f"{self.endpoint}/v1/images/generations",
            headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
            json=request_data,
        )
        response.raise_for_status()
        response_json = response.json()

        b64 = response_json["data"][0]["b64_json"]
        return base64_decode(b64)

    def generate_image_tool(self, prompt: str) -> str:
        """
        Generate an illustration and save it, return the path.
        """
        png = self.generate_image(prompt)
        filename = generate_random_filename()
        return save_image(png, filename)


def video_size(image_path: str, area: int = 640 * 480) -> str:
    # Keep the image's aspect ratio at about the same pixel count as 640x480.
    # Wan needs width and height divisible by 16.
    width, height = Image.open(image_path).size
    scale = (area / (width * height)) ** 0.5
    w = round(width * scale / 16) * 16
    h = round(height * scale / 16) * 16
    return f"{w}x{h}"


class VideoGenerator:

    def __init__(self, endpoint: str, model: str, token: str | None = None):
        self.endpoint = endpoint
        self.model = model
        self.token = token

    def generate_video(self, prompt: str, image_path: str, duration: float, fps: int) -> bytes:
        # Wan needs num_frames = 4k + 1 (e.g. 49, 81)
        num_frames = round(duration * fps / 4) * 4 + 1
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "size": video_size(image_path),
            "num_frames": str(num_frames),
            "fps": str(fps),
            "num_inference_steps": "20",
            "guidance_scale": "1.0",
            "guidance_scale_2": "1.0",
            "flow_shift": "12.0",
            "boundary_ratio": "0.875",
            "seed": "7",
        }

        image_bytes = Path(image_path).read_bytes()

        response = requests.post(
            f"{self.endpoint}/v1/videos/sync",
            headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
            data=request_data,
            files={"input_reference": ("frame.png", image_bytes, "image/png")},
            timeout=1800,
        )
        response.raise_for_status()

        return response.content

    def generate_video_tool(
        self, prompt: str, image_path: str, duration: float = 5.0, fps: int = 16
    ) -> str:
        """
        Animate an image into a short clip and save it, return the path.
        The image becomes the first frame of the video.
        duration is the clip length in seconds (keep it at 5 or less),
        fps is frames per second (16 by default).
        """
        mp4 = self.generate_video(prompt, image_path, duration, fps)
        filename = generate_random_filename(prefix="wan", ext="mp4")
        return save_video(mp4, filename)
