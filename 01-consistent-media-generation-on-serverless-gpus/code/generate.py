"""One Sana pipeline, repeated sequential generations, local or mounted output."""
import argparse
import base64
import importlib.metadata
import os
import subprocess
import time
from common import DEFAULT_CONFIG, asset_record, config, emit, metrics, output_dir, positive, write_json

# A tiny PNG fixture, not an AI-generated image. No GPU measurement is fabricated.
FAKE_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgaPgPAAIDAYAkYfWXAAAAAElFTkSuQmCC")

def load_pipeline(cfg):
    import torch
    from diffusers import SanaPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --fake locally or choose a GPU runtime")
    pipe = SanaPipeline.from_pretrained(cfg["model"], revision=cfg["model_revision"],
        variant=cfg["variant"], torch_dtype=torch.float16).to("cuda")
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.bfloat16)
    return pipe

def run(config_path=DEFAULT_CONFIG, n=1, out=None, fake=False, mode="job"):
    cfg = config(config_path)
    cfg["image"] = os.environ.get("WORKER_IMAGE")
    folder = output_dir(out)
    m = metrics(cfg, mode, fake)
    emit(folder, "run.started", fake=fake)
    try:
        if not fake:
            import torch
            start = time.perf_counter()
            pipe = load_pipeline(cfg)
            torch.cuda.synchronize()
            m["load_s"] = time.perf_counter()-start
            m["env"].update(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                torch=torch.__version__, diffusers=importlib.metadata.version("diffusers"),
                preset=os.environ.get("GPU_PRESET"))
            try:
                m["env"]["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                    "--format=csv,noheader"], text=True, timeout=10).strip()
            except (OSError, subprocess.SubprocessError):
                pass
        emit(folder, "model.loaded", duration_s=m["load_s"])
        for index in range(n):
            path = folder / f"asset-{index:03d}.png"
            if fake:
                path.write_bytes(FAKE_PNG)
            else:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                image = pipe(prompt=cfg["prompt"], **cfg["params"],
                    generator=torch.Generator(device="cuda").manual_seed(cfg["seed"])).images[0]
                torch.cuda.synchronize()
                m["latency_s"].append(time.perf_counter()-start)
                m["env"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                image.save(path)
            m["assets"].append(asset_record(path))
            emit(folder, "asset.saved", file=path.name, fake=fake)
        m["status"] = "completed"
    except Exception as exc:
        m["status"] = "failed"
        emit(folder, "run.failed", error_type=type(exc).__name__)
        raise
    finally:
        emit(folder, "run.finished", status=m["status"])
        write_json(folder / "metrics.json", m)
        print(folder / "metrics.json")
    return folder

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--n", type=positive, default=1)
    p.add_argument("--out")
    p.add_argument("--mode", choices=["job", "jupyterlab"], default="job")
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()
    run(a.config, a.n, a.out, a.fake, a.mode)
