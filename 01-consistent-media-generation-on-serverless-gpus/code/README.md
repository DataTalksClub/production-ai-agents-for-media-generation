# Module 1 code

Run shell commands from the **repository root**, unless a command says otherwise. Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/) are the local prerequisites. CPU clients, cost calculations and offline checks use only the Python standard library. GPU libraries are imported only on the live worker path.

## Automated Python checks

Fake mode is reserved for automated `.py` tests. It is not part of the learner workflow or the notebooks.

```bash
uv sync
uv run python 01-consistent-media-generation-on-serverless-gpus/code/smoke_test.py --fake
```

The test runner invokes the Python entry points with `--fake` to check file handling and arithmetic without network access or a GPU. Fixture metrics are labelled `fake: true` and cannot be used for live cost calculations. Notebooks are checked for syntax and structure only; executing them requires their real Endpoint or GPU runtime.

Every generation gets a fresh run directory. `--out` can choose the directory explicitly, but it must not already exist.

| File | Lesson | Entry point / purpose |
|---|---|---|
| `generation.json` | 1.2–1.5 | Shared prompt, model commit, seed and parameters |
| `endpoint_quickstart.ipynb` | 1.2 A | Laptop walkthrough of `call_endpoint.run` |
| `jupyterlab_models.ipynb` | 1.2 B | GPU walkthrough of `generate.run`; Sana reference path |
| `call_endpoint.py` | 1.2, 1.4 | Readiness probe, image request, local output |
| `generate.py` | 1.2, 1.4 | Pinned Sana loading and repeated generation |
| `Dockerfile.bad`, `Dockerfile` | 1.3 | Reading exercise; corrected worker can be built |
| `submit_job.sh` | 1.4 | Explicit Job resource request and bucket mount |
| `measure_endpoint.py` | 1.5 | Same Endpoint client, accepting observed lifecycle times |
| `measure_job.py` | 1.5 | Add lifecycle evidence to a downloaded metrics file |
| `cost.py` | 1.5 | Marginal/fixed costs, N projections and breakeven |
| `common.py` | all code | Local config, JSON, trace and timestamp helpers |
| `smoke_test.py` | validation | Automated Python and failure-path checks; notebook syntax checks |

## Endpoint setup

Deploy the official [Sana template](https://github.com/nebius/serverless-ai-cookbook/tree/main/templates/endpoint-sana) in the console; keep token authentication enabled. These scripts do not create resources. Record lifecycle timestamps yourself and stop/delete the Endpoint afterward, even if the client fails. See [1.2](../02-first-generation.md) for the full sequence.

| Variable | Used by | Meaning |
|---|---|---|
| `BASE_URL` | Endpoint client | HTTPS origin, **without** `/v1` |
| `TOKEN` | Endpoint client | Endpoint bearer token, never written to outputs |
| `ENDPOINT_IMAGE` | Endpoint client | Image reference copied from deployment metadata |
| `WORKER_IMAGE` | Job wrapper/worker | Pushed worker digest; recorded with configuration |
| `BUCKET` | Job wrapper | Existing private bucket; credentials in AWS default profile |
| `RUN_ID` | Job wrapper | Unique safe name, e.g. `fox-20261116T090000Z` |
| `JOB_TIMEOUT` | Job wrapper | Explicit platform timeout, default `1h` |
| `GPU_PRESET` | Worker | Optional environment metadata; wrapper sets it |
| `HF_TOKEN` | GPU Hub loader | Optional private runtime token, not needed for default public weights |

`.env.example` documents names only; the scripts do not auto-load it. Set values in your shell or runtime secret configuration. Do not paste secrets into notebooks, JSON config or shell commands committed to Git.

```bash
uv run python 01-consistent-media-generation-on-serverless-gpus/code/call_endpoint.py --n 3
```

The client supports the inspected Sana image API, not every OpenAI-shaped image server. A different template might use different extension fields or return URLs rather than base64 PNGs. Use that template's documentation before adapting it. Nonretryable HTTP errors fail promptly; readiness retries are bounded. Generation requests are not automatically retried.

## Notebooks

```bash
uv sync --extra notebooks
uv run jupyter lab
```

Open either notebook in this directory. They can locate `code/` when started from the repository root, module directory or code directory. Configure the Endpoint credentials or GPU runtime, then run the notebook top to bottom. Both notebooks perform real generation and end with cleanup instructions.

For GPU JupyterLab, use a CUDA PyTorch runtime compatible with the pinned versions in the root `gpu` extra. The notebook has an explicit installation cell for the remaining libraries. Do not install a CPU wheel over a working CUDA runtime. The tested-local status does not imply that the live CUDA environment has been validated; see [VALIDATION.md](../VALIDATION.md).

The Option B filename follows the module's planned layout. This version implements one Sana pipeline; optional Qwen/Wan/audio exploration uses separate template recipes rather than pretending their pipelines are interchangeable.

## Worker image

Build from the module's **code directory** as context. The Dockerfile uses the inspected cookbook's runtime versions; live build/run verification remains pending. Use a registry your Nebius project can pull from. Follow [Container Registry setup](https://docs.nebius.com/container-registry/quickstart) for authentication and image push.

```bash
export WORKER_TAG='YOUR_REGISTRY/sana-worker:module1'
docker buildx build --platform linux/amd64 \
  -t "$WORKER_TAG" --push \
  01-consistent-media-generation-on-serverless-gpus/code
# Inspect the pushed manifest and copy its digest:
docker buildx imagetools inspect "$WORKER_TAG"
export WORKER_IMAGE='YOUR_REGISTRY/sana-worker@sha256:YOUR_DIGEST'
```

The GPU image provides PyTorch; it does not install another torch wheel. Direct GPU execution outside the container can use `uv sync --extra gpu` on an appropriate Linux GPU host, but validate the installed CUDA build before loading. CPU/macOS learners should use the default dependencies and the remote Endpoint instead.

The model commit is pinned in `generation.json`. The root `uv.lock` records resolved Python dependencies. The Dockerfile uses separate explicit versions, so record the pushed image digest and actual runtime versions in the run report. No instructor worker image or measured GPU result is supplied yet.

## Job and storage

Configure a private Object Storage bucket and its access keys in an AWS-compatible default profile. See [storage quickstart](https://docs.nebius.com/object-storage/quickstart), [AWS CLI setup](https://docs.nebius.com/object-storage/interfaces/aws-cli) and [Job flags](https://docs.nebius.com/cli/reference/ai/job/create).

Set `WORKER_IMAGE`, `BUCKET` and a new `RUN_ID`. Run `submit_job.sh --dry-run` to validate, then run it without that flag to submit. The default request uses on-demand L40S; costs depend on your actual project, region and runtime. To change the prompt, edit `generation.json` and rebuild the worker so the submitted image contains the new config.

The Job writes to `/outputs/runs/01-consistent-media-generation-on-serverless-gpus/<RUN_ID>/`, which maps to your mounted bucket. Capture the Job ID, inspect its status and logs, verify artifacts on completion, and cancel if it stalls or exceeds your own spending limit. The configured timeout is a failure bound and is not a monetary cap. CLI documentation and cookbook examples disagree on the shortest valid duration; this wrapper uses the documented `1h` minimum. Confirm the installed CLI before launch.

## Measurements and costs

`metrics.json` contains:

- `startup_s`, `ready_wait_s`, `load_s`, `latency_s`, `billable_s` using the definitions in [1.5](../05-measuring-costs.md).
- `mode`, `fake`, `status` and `billing_basis`, so simulated work and wall-time proxies cannot be mistaken for real invoice evidence.
- `config`, with exact generation settings. The Endpoint uses `requested_model_revision` and `model_revision: null` because the stock server does not pin/report the actual revision.
- `env`, with runtime details where readable; unknown Endpoint hardware/software remains null.
- `assets`, with relative names and SHA-256 digests.

Pass `--created-at` and `--running-at` to `measure_endpoint.py` when starting the readiness probe promptly after observing RUNNING. Later, attach all three lifecycle observations to either an Endpoint or Job file:

```bash
uv run python 01-consistent-media-generation-on-serverless-gpus/code/measure_job.py \
  "$METRICS_PATH" --created-at "$CREATED_AT" --running-at "$RUNNING_AT" --finished-at "$FINISHED_AT"
uv run python 01-consistent-media-generation-on-serverless-gpus/code/cost.py \
  "$METRICS_PATH" --price-per-hour "$PRICE_PER_HOUR" --n 1 10 100 --api-price "$API_PRICE_PER_ASSET"
```

Use ISO timestamps with timezones (for example, the format `YYYY-MM-DDTHH:MM:SSZ`). Complete [run-report.md](../run-report.md) with sources and observation limits. The enrichment is manual timestamp processing, not an automated resource lifecycle controller. It cannot recover events you did not record.

`trace.jsonl` uses the local draft event contract: `schema_version`, `timestamp` (UTC ISO), `run_id`, `event`, plus event details. Events include run start/finish/failure, model/API ready, asset saved, lifecycle recorded and cost calculated. No trace backend is required. A future shared course trace schema will need an explicit compatibility review.
