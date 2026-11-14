"""Lesson 04: Santa's gift cards as a batch pipeline: one GPU, six steps, N kids.

    kid profile -> text -> illustration -> card -> clip -> greeting -> mix

Every step is a plain function, so the notebook (nb-santa-cards.ipynb) can run it for one kid.
run() works step by step, not kid by kid: it loads one model, makes that step's file for every
kid, saves it to <out>/cards/<id>/, frees the GPU and moves on. Steps hand over through these
files only, and each file is written once, so <out> can be a mounted bucket.

    python code/batch_inference.py --n 10 --out runs/batch_inference/job-001
    python code/batch_inference.py --kids /data/inputs/kids.csv --n 10 --out /data/runs/batch_inference/job-001

In a Job, /data is your bucket: the kids' profiles come from it and the cards go to it.

The clip step runs Wan2.2 image-to-video, the largest model here: the full pipeline needs an H100.
The pipeline is deterministic: nothing judges or retries. Resuming after a preempted GPU is
Module 3. Adapted from https://github.com/mnrozhkov/santa-scale-challenge.
"""

from __future__ import annotations

import csv
import gc
import shutil
import statistics
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, Field, computed_field
from rich.console import Console

from common import load_pipeline
from models import KOKORO, SANA, WAN_I2V

if TYPE_CHECKING:
    from diffusers import SanaPipeline, WanImageToVideoPipeline
    from kokoro import KPipeline

ASSETS = Path(__file__).resolve().parent.parent / "assets"  # data and media of this module
KIDS_CSV = ASSETS / "kids.csv"  # id,name,age,wishlist,wish,greeting,prompt: 200 kids, Emma first
FONTS = ASSETS / "fonts"
DEFAULT_TRACK = ASSETS / "4_jingle.mp3"

CARD_SIZE = (1024, 1400)
MUSIC_VOLUME = 0.2  # the music sits under the voice
MOTION = (
    "Gentle storybook motion: a slow camera drift across the illustration, softly falling snow "
    "and warm twinkling lights. No people, no faces, no text, no letters, no watermarks."
)

# Job logs aren't a terminal: soft_wrap keeps each log line on one line.
console = Console(soft_wrap=True, markup=False)


def log(message: str) -> None:
    """Print one timestamped line; in a Job it shows up in `nebius ai job logs`."""
    console.print(f"{time.strftime('%H:%M:%S')} {message}")


# --- records ------------------------------------------------------------------------------------


class Step(StrEnum):
    """The pipeline's steps, in the order run() does them."""

    TEXT = "text"
    ILLUSTRATION = "illustration"
    CARD = "card"
    CLIP = "clip"
    GREETING = "greeting"
    MIX = "mix"


Mood = Literal["warm", "playful", "dreamy", "adventurous", "cozy", "magical", "joyful", "calm"]


class KidProfile(BaseModel):
    """One row of kids.csv. The wish, greeting and illustration prompt are already written."""

    id: str
    name: str
    age: int = Field(ge=1, le=18)
    wishlist: str
    wish: str = Field(min_length=1)
    greeting: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class CardText(BaseModel):
    """Everything the later steps need for one kid; saved as cards/<id>/card.json."""

    kid: KidProfile
    gifts: list[str] = Field(min_length=1, max_length=5)
    wish: str = Field(min_length=1)
    mood: Mood = "warm"
    greeting: str
    image_prompt: str


class StepMetrics(BaseModel):
    """Timings for one pipeline step across the batch."""

    load_s: float | None = None
    latency_s: dict[str, float] = Field(default_factory=dict)

    @computed_field
    @property
    def median_s(self) -> float | None:
        return round(statistics.median(self.latency_s.values()), 3) if self.latency_s else None

    @computed_field
    @property
    def total_s(self) -> float:
        return round((self.load_s or 0.0) + sum(self.latency_s.values()), 3)


class RunMetrics(BaseModel):
    """Whole-run timings written to <out>/metrics_batch.json."""

    n: int
    kids_csv: str
    gpu: str | None = None
    started_at: str
    finished_at: str
    runtime_s: float
    steps: dict[Step, StepMetrics]


def load_kids(n: int, path: Path = KIDS_CSV) -> list[KidProfile]:
    """The first n kid profiles from the CSV."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [KidProfile.model_validate(row) for row in rows[:n]]


def read_card(folder: Path) -> CardText:
    """The card text that step 1 saved in a kid's folder."""
    return CardText.model_validate_json((folder / "card.json").read_text())


# --- 1. text: gifts, wish, mood and the spoken greeting -----------------------------------------


def write_text(kid: KidProfile) -> CardText:
    """Copy the wish, greeting and illustration prompt from the kid's row into card.json."""
    return CardText(
        kid=kid,
        gifts=[kid.wishlist],
        wish=kid.wish.strip(),
        mood="warm",
        greeting=kid.greeting.strip(),
        image_prompt=kid.prompt.strip(),
    )


# --- 2. illustration: Sana ----------------------------------------------------------------------


def load_image_model() -> SanaPipeline:
    """Sana on the GPU, pinned to one Hub commit (models.SANA)."""
    return load_pipeline(SANA)


def draw_illustration(pipe: SanaPipeline, prompt: str) -> Image.Image:
    """One illustration with the lesson's Sana settings and seed."""
    import torch  # GPU-only package: imported where it's needed

    generator = torch.Generator(device="cuda").manual_seed(SANA.seed)
    return pipe(prompt=prompt, **SANA.params, generator=generator).images[0]


# --- 3. card: Pillow, no GPU --------------------------------------------------------------------


def render_card(illustration: Image.Image, card: CardText) -> Image.Image:
    """A 1024x1400 card: the illustration on top, then the kid's name and the wish."""
    image = Image.new("RGB", CARD_SIZE, "#fffaf2")
    image.paste(ImageOps.fit(illustration.convert("RGB"), (944, 720)), (40, 40))
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(str(FONTS / "Inter-Bold.ttf"), 64)
    body = ImageFont.truetype(str(FONTS / "Inter-Regular.ttf"), 38)
    headline = f"Merry Christmas, {card.kid.name}!"
    draw.text((512, 850), headline, font=title, fill="#b30000", anchor="mm")

    lines, line = [], ""  # wrap the wish to 880 px
    for word in card.wish.split():
        if draw.textlength(f"{line} {word}", font=body) > 880:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    draw.multiline_text(
        (512, 940), "\n".join(lines), font=body, fill="#2c3e50",
        anchor="ma", align="center", spacing=14,
    )
    return image


@contextmanager
def staged(out: Path) -> Iterator[Path]:
    """Write to local disk first, then copy once to `out` (bucket mounts reject random-write seeks)."""
    with tempfile.NamedTemporaryFile(suffix=out.suffix) as tmp:
        path = Path(tmp.name)
        yield path
        shutil.copyfile(path, out)


# --- 4. clip: Wan2.2 image-to-video -------------------------------------------------------------


def load_video_model() -> WanImageToVideoPipeline:
    """Wan2.2 image-to-video (models.WAN_I2V) on the GPU. The first call downloads the weights."""
    import torch  # GPU-only packages: imported where they're needed
    from diffusers import WanImageToVideoPipeline

    pipe = WanImageToVideoPipeline.from_pretrained(WAN_I2V.model, torch_dtype=torch.bfloat16)
    return pipe.to("cuda")


def animate(pipe: WanImageToVideoPipeline, illustration: Path, out: Path) -> None:
    """Animate the illustration and save a silent MP4."""
    import torch  # GPU-only packages: imported where they're needed
    from diffusers.utils import export_to_video

    with Image.open(illustration) as still:
        first_frame = ImageOps.fit(still.convert("RGB"), WAN_I2V.size)
    generator = torch.Generator(device="cuda").manual_seed(WAN_I2V.seed)
    frames = pipe(image=first_frame, prompt=MOTION, **WAN_I2V.params, generator=generator).frames[0]
    with staged(out) as tmp:
        export_to_video(frames, str(tmp), fps=WAN_I2V.fps)  # fps must be explicit: the default is 10


# --- 5. greeting: Kokoro text-to-speech ---------------------------------------------------------


def load_voice() -> KPipeline:
    """Kokoro text-to-speech for American English (models.KOKORO)."""
    from kokoro import KPipeline  # installed in the worker image and the DevLab, not on a laptop

    return KPipeline(lang_code=KOKORO.lang_code, repo_id=KOKORO.model)


def speak(voice: KPipeline, text: str, out: Path) -> float:
    """Say the text, save a 16-bit mono WAV, return its length in seconds."""
    import numpy as np  # comes with the GPU stack

    audio = np.concatenate([np.asarray(chunk) for _, _, chunk in voice(text, voice=KOKORO.voice)])
    with staged(out) as tmp, wave.open(str(tmp), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(KOKORO.sample_rate)
        f.writeframes((audio.clip(-1, 1) * 32767).astype("<i2").tobytes())
    return len(audio) / KOKORO.sample_rate


# --- 6. mix: ffmpeg, no GPU ---------------------------------------------------------------------


def pick_music(mood: str) -> Path:
    """assets/<mood>.mp3 if you add one; otherwise the bundled jingle."""
    track = ASSETS / f"{mood}.mp3"
    return track if track.is_file() else DEFAULT_TRACK


def card_seconds(greeting: Path) -> float:
    """How long the card plays: the clip's length, or the greeting plus half a second if longer."""
    with wave.open(str(greeting)) as f:
        greeting_s = f.getnframes() / f.getframerate()
    return max(WAN_I2V.duration_s, greeting_s + 0.5)


def mix(clip: Path, greeting: Path, music: Path, out: Path, seconds: float) -> None:
    """Clip + spoken greeting + quiet background music -> one MP4 that lasts `seconds`.

    The clip and the music loop, so the card can last longer than the clip.
    """
    import imageio_ffmpeg  # its own ffmpeg binary, the one export_to_video uses

    audio = f"[2:a]volume={MUSIC_VOLUME}[m];[1:a][m]amix=inputs=2:duration=longest:normalize=0[a]"
    with staged(out) as tmp:
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(clip),
            "-i", str(greeting),
            "-stream_loop", "-1", "-i", str(music),
            "-filter_complex", audio,
            "-map", "0:v", "-map", "[a]", "-t", f"{seconds:.2f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(tmp),
        ]
        subprocess.run(command, check=True)


# --- the batch run ------------------------------------------------------------------------------


def free_gpu() -> None:
    """Release the memory of a model you've deleted, before the next model loads."""
    import torch  # GPU-only package: imported where it's needed

    gc.collect()
    torch.cuda.empty_cache()


def gpu_name() -> str | None:
    """The CUDA device name when a GPU is present, otherwise None."""
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        return None


def load_model(step: Step, loader: Callable[[], Any], stats: StepMetrics) -> Any:
    """Load a step's model once, record load_s, and log how long it took."""
    start = time.perf_counter()
    model = loader()
    elapsed = time.perf_counter() - start
    stats.load_s = round(elapsed, 3)
    log(f"{step} model loaded in {elapsed:.1f} s")
    return model


@contextmanager
def progress(step: Step, i: int, total: int, kid_id: str, stats: StepMetrics) -> Iterator[None]:
    """Record the step's latency for this kid and log one line for `nebius ai job logs`."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    stats.latency_s[kid_id] = round(elapsed, 3)
    log(f"{step} {i}/{total} {kid_id} {elapsed:.1f} s")


def run(
    n: int = 10,
    out: Path = Path("runs/santa"),
    steps: Sequence[Step] = tuple(Step),
    kids_csv: Path = KIDS_CSV,
) -> Path:
    """Make cards for the first n kids in kids_csv, one step at a time.

    Writes <out>/metrics_batch.json and returns the folder that holds cards/<id>/.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()
    out = Path(out)
    kids = load_kids(n, kids_csv)
    cards = out / "cards"
    folders = [cards / kid.id for kid in kids]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    steps_metrics: dict[Step, StepMetrics] = {Step(s): StepMetrics() for s in steps}
    log(f"{len(kids)} kids from {kids_csv}, steps: {', '.join(steps)}, out: {cards}")

    if Step.TEXT in steps:
        stats = steps_metrics[Step.TEXT]
        for i, (kid, folder) in enumerate(zip(kids, folders), 1):
            with progress(Step.TEXT, i, len(kids), kid.id, stats):
                card = write_text(kid)
                (folder / "card.json").write_text(card.model_dump_json(indent=2))

    if Step.ILLUSTRATION in steps:
        stats = steps_metrics[Step.ILLUSTRATION]
        pipe = load_model(Step.ILLUSTRATION, load_image_model, stats)
        for i, folder in enumerate(folders, 1):
            with progress(Step.ILLUSTRATION, i, len(folders), folder.name, stats):
                illustration = draw_illustration(pipe, read_card(folder).image_prompt)
                illustration.save(folder / "illustration.png")
        del pipe
        free_gpu()

    if Step.CARD in steps:
        stats = steps_metrics[Step.CARD]
        for i, folder in enumerate(folders, 1):
            with progress(Step.CARD, i, len(folders), folder.name, stats):
                with Image.open(folder / "illustration.png") as illustration:
                    render_card(illustration, read_card(folder)).save(folder / "card.png")

    if Step.CLIP in steps:
        stats = steps_metrics[Step.CLIP]
        pipe = load_model(Step.CLIP, load_video_model, stats)
        for i, folder in enumerate(folders, 1):
            with progress(Step.CLIP, i, len(folders), folder.name, stats):
                animate(pipe, folder / "illustration.png", folder / "clip.mp4")
        del pipe
        free_gpu()

    if Step.GREETING in steps:
        stats = steps_metrics[Step.GREETING]
        voice = load_model(Step.GREETING, load_voice, stats)
        for i, folder in enumerate(folders, 1):
            with progress(Step.GREETING, i, len(folders), folder.name, stats):
                speak(voice, read_card(folder).greeting, folder / "greeting.wav")
        del voice
        free_gpu()

    if Step.MIX in steps:
        stats = steps_metrics[Step.MIX]
        for i, folder in enumerate(folders, 1):
            with progress(Step.MIX, i, len(folders), folder.name, stats):
                greeting = folder / "greeting.wav"
                music = pick_music(read_card(folder).mood)
                seconds = card_seconds(greeting)
                mix(folder / "clip.mp4", greeting, music, folder / "card.mp4", seconds)

    metrics = RunMetrics(
        n=len(kids),
        kids_csv=str(kids_csv),
        gpu=gpu_name(),
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        runtime_s=round(time.perf_counter() - run_start, 3),
        steps=steps_metrics,
    )
    metrics_path = out / "metrics_batch.json"
    metrics_path.write_text(metrics.model_dump_json(indent=2) + "\n")
    log(f"finished: {cards} (metrics: {metrics_path})")
    return cards


# --- command line -------------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    kids: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False,
            help="The kids' profiles (CSV). In a Job, a path on the bucket mount.",
        ),
    ] = KIDS_CSV,
    n: Annotated[int, typer.Option(min=1, max=200, help="How many kids from the CSV.")] = 10,
    out: Annotated[
        Path, typer.Option(help="Output folder. In a Job, a path on the bucket mount.")
    ] = Path("runs/santa"),
    step: Annotated[
        list[Step] | None,
        typer.Option(help="Run only these steps (repeat the option). Default: all six."),
    ] = None,
) -> None:
    """Make Santa's gift cards for the first N kids, one step at a time."""
    run(n, out, step or tuple(Step), kids)


if __name__ == "__main__":
    app()
