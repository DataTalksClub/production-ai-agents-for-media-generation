"""Compress one emma_card.ipynb run into small lesson assets (maintainer tool).

    uv run python make_lesson_assets.py --run runs/emma-card/<ts> --out assets/01-landscape

Needs ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

KB = 1024
FOLDER_BUDGET = 3 * KB * KB
GIF_BUDGET = 1 * KB * KB
# asset name in the run manifest -> (output file, size budget)
OUTPUTS = {
    "illustration": ("emma-illustration.webp", 120 * KB),
    "new-scene": ("emma-new-scene.webp", 120 * KB),
    "clip": ("emma-clip.mp4", int(1.2 * KB * KB)),
    "greeting": ("emma-greeting.mp3", 80 * KB),
    "jingle": ("emma-jingle.mp3", 150 * KB),
}


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def duration_s(path):
    """Media duration from ffmpeg's banner; None for images."""
    info = subprocess.run(["ffmpeg", "-i", str(path)], capture_output=True, text=True).stderr
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
    if not match:
        return None
    h, m, s = match.groups()
    return round(int(h) * 3600 + int(m) * 60 + float(s), 2)


def webp(src, dest):
    img = Image.open(src).convert("RGB")
    img = img.resize((768, round(img.height * 768 / img.width)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=80)
    dest.write_bytes(buf.getvalue())


def gif(src, dest):
    for fps, width in ((8, 416), (6, 416), (6, 320)):
        ffmpeg(
            "-i", str(src),
            "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
            str(dest),
        )
        if dest.stat().st_size <= GIF_BUDGET:
            return
    print(f"warning: {dest.name} is still over 1 MB at 6 fps, 320 px")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="runs/emma-card/<ts>")
    parser.add_argument("--out", required=True, type=Path, help="assets/01-landscape")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH. Install it (macOS: brew install ffmpeg) and run again.")
    records = {r["asset"]: r for r in json.loads((args.run / "manifest.json").read_text())}
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for name, (out_name, _) in OUTPUTS.items():
        record = records.get(name)
        if record is None:
            print(f"skip {name}: not in this run")
            continue
        src = args.run / record["file"]
        dest = args.out / out_name
        if name in ("illustration", "new-scene"):
            webp(src, dest)
        elif name == "clip":
            ffmpeg(
                "-i", str(src), "-an", "-c:v", "libx264", "-vf", "scale=832:480",
                "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest),
            )
            gif(src, args.out / "emma-clip.gif")
        elif name == "greeting":
            ffmpeg("-i", str(src), "-ac", "1", "-b:a", "48k", str(dest))
        else:
            ffmpeg("-i", str(src), "-ac", "2", "-b:a", "96k", str(dest))
        outputs = [dest] + ([args.out / "emma-clip.gif"] if name == "clip" else [])
        manifest.append(
            {
                **record,
                "outputs": [
                    {"file": p.name, "bytes": p.stat().st_size, "duration_s": duration_s(p)}
                    for p in outputs
                ],
            }
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    budgets = dict(OUTPUTS.values())
    budgets["emma-clip.gif"] = GIF_BUDGET
    total = 0
    print(f"\n{'file':<24}{'KB':>8}{'budget KB':>11}")
    for path in sorted(args.out.iterdir()):
        size = path.stat().st_size
        total += size
        budget = budgets.get(path.name)
        flag = " over" if budget and size > budget else ""
        print(f"{path.name:<24}{size / KB:>8.0f}{(budget or 0) / KB:>11.0f}{flag}")
    print(f"{'total':<24}{total / KB:>8.0f}{FOLDER_BUDGET / KB:>11.0f}")
    if total > FOLDER_BUDGET:
        sys.exit("Folder is over the 3 MB budget.")


if __name__ == "__main__":
    main()
