# Module 1 Code Guide

Run shell commands from the **repository root**, unless a command specifies otherwise. You need Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync --extra notebooks
```

## Files in `code/`

| File | Lesson | Purpose |
|---|---|---|
| `generation.json` | 1.2–1.5 | Shared prompt (Emma's holiday telescope card), pinned Sana commit, seed (`7`), and generation parameters |
| `endpoint_quickstart.ipynb` | 1.2 Option A *(Default)* | Laptop notebook that calls a deployed Sana Endpoint using `ImageGenerator`, records `metrics.json`, and computes costs |
| `jupyterlab_models.ipynb` | 1.2 Option B *(Optional)* | GPU DevLab notebook for participants who choose to run Diffusers directly on a GPU VM and inspect VRAM |
| `call_endpoint.py` | 1.2, 1.4 | Standalone CLI script and `ImageGenerator` client for an OpenAI-compatible image Endpoint |
| `generate.py` | 1.2, 1.4 | Standalone GPU worker that loads the pinned Sana checkpoint once and runs `--n` sequential generations |
| `Dockerfile.bad`, `Dockerfile` | 1.3 | Broken vs production GPU worker container definitions |
| `measure_endpoint.py` | 1.5 | Runs `call_endpoint.py` and attaches observed Endpoint creation, `RUNNING`, and stop timestamps |
| `measure_job.py` | 1.5 | Attaches observed lifecycle timestamps (`created_at`, `running_at`, `finished_at`) to a `metrics.json` file |
| `common.py` | 1.2–1.5 | Shared configuration loader, `metrics.json` / `trace.jsonl` writers, and the `calculate_cost()` Python function |

## Environment Variables

See `.env.example` for the full list of variable names. Export variables in your shell before running the scripts or starting Jupyter Lab; never commit tokens or credentials to Git.

| Variable | Used By | Description |
|---|---|---|
| `BASE_URL` | `call_endpoint.py`, `measure_endpoint.py`, `endpoint_quickstart.ipynb` | HTTPS origin of your Endpoint (without `/v1`) |
| `TOKEN` | `call_endpoint.py`, `measure_endpoint.py`, `endpoint_quickstart.ipynb` | Bearer token configured when creating the Endpoint |
| `ENDPOINT_IMAGE` | `call_endpoint.py` | Optional container image reference from the Endpoint deployment |
| `QWEN_ENDPOINT_URL`, `QWEN_ENDPOINT_TOKEN` | `endpoint_quickstart.ipynb` | Optional second Endpoint for the `Qwen/Qwen-Image` comparison cell |
| `WORKER_IMAGE` | `generate.py` | Pushed worker image digest (`registry/sana-worker@sha256:...`) |
| `BUCKET` | `nebius ai job create` | Name of your private Object Storage bucket |
| `RUN_ID` | `nebius ai job create` | Unique run identifier (for example `emma-20261116T100000Z`) |

## 1.2 Option A (Default): Laptop + Endpoint

Deploy the [Sana template](https://github.com/nebius/serverless-ai-cookbook/tree/main/templates/endpoint-sana) in the Nebius console with token authentication enabled. Then run either the notebook or the CLI client:

```bash
export BASE_URL="https://<your-endpoint-host>"
read -rs TOKEN
export TOKEN

# Run from CLI:
uv run python 01-consistent-media-generation-on-serverless-gpus/code/call_endpoint.py --n 3

# Or open endpoint_quickstart.ipynb in Jupyter Lab:
uv run jupyter lab
```

Stop or delete the Endpoint in the console as soon as your run completes.

## 1.2 Option B (Optional): GPU DevLab

If you chose to start a GPU DevLab (JupyterLab on an L40S VM) instead of using an Endpoint from your laptop, clone this repository inside the DevLab and open `jupyterlab_models.ipynb`, or run `generate.py` directly on the GPU:

```bash
uv run python 01-consistent-media-generation-on-serverless-gpus/code/generate.py --n 3 --mode jupyterlab
```

Download your outputs from `runs/01-consistent-media-generation-on-serverless-gpus/` and stop or delete the DevLab in the console when finished.

## 1.4 Build and Submit a Batch Job

To run `generate.py` as a finite Serverless GPU Job writing to Object Storage:

```bash
export WORKER_TAG="cr.<region>.nebius.cloud/<your-registry-id>/sana-worker:module1"
docker buildx build --platform linux/amd64 \
  -t "$WORKER_TAG" --push \
  01-consistent-media-generation-on-serverless-gpus/code

docker buildx imagetools inspect "$WORKER_TAG"
export WORKER_IMAGE="cr.<region>.nebius.cloud/<your-registry-id>/sana-worker@sha256:<your-digest>"
export BUCKET="<your-bucket-name>"
export RUN_ID="emma-$(date -u +%Y%m%dT%H%M%SZ)"

# Validate the request first with --dry-run, then remove --dry-run to submit:
nebius ai job create \
  --name "m1-${RUN_ID}" \
  --image "$WORKER_IMAGE" \
  --platform gpu-l40s-a \
  --preset 1gpu-8vcpu-32gb \
  --on-demand \
  --timeout 1h \
  --restart-policy never \
  --volume "s3://${BUCKET}:/outputs:rw" \
  --env "WORKER_IMAGE=${WORKER_IMAGE}" \
  --env GPU_PRESET=1gpu-8vcpu-32gb \
  --container-command python \
  --args "/app/generate.py --config /app/generation.json --n 3 --out /outputs/runs/01-consistent-media-generation-on-serverless-gpus/${RUN_ID}" \
  --dry-run
```

## 1.5 Enrich Timestamps and Calculate Cost in Python

Attach your observed UTC timestamps (`YYYY-MM-DDTHH:MM:SSZ`) to `metrics.json`:

```bash
uv run python 01-consistent-media-generation-on-serverless-gpus/code/measure_job.py \
  "$METRICS_PATH" \
  --created-at "$CREATED_AT" \
  --running-at "$RUNNING_AT" \
  --finished-at "$FINISHED_AT"
```

Then call `calculate_cost` directly from Python or a notebook cell:

```python
import json
from pathlib import Path
from common import calculate_cost

m = json.loads(Path("runs/01-consistent-media-generation-on-serverless-gpus/<run-id>/metrics.json").read_text())
ready_s = m["ready_wait_s"] if m["mode"] == "endpoint" else m["load_s"]
report = calculate_cost(
    startup_s=m["startup_s"],
    ready_s=ready_s,
    latency_s=m["latency_s"],
    price_per_hour=1.5484,  # L40S 1gpu-8vcpu-32gb on-demand rate
    counts=(1, 10, 100),
    idle_s=0.0,
    api_price=0.04,
)
print(json.dumps(report, indent=2))
```
