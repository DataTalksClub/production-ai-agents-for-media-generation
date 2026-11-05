# Validation and publication status

This is authored lesson content plus a locally testable code pack. It is **not yet a live-validated course release**. No paid resource was launched while authoring it. No GPU latency, memory, price or credit-use result is invented.

## Local validation

Run from the repository root:

```bash
uv run python 01-consistent-media-generation-on-serverless-gpus/code/smoke_test.py --fake
bash -n 01-consistent-media-generation-on-serverless-gpus/code/submit_job.sh
```

The smoke suite runs every Python entry point in fake mode, verifies PNG/metadata output, checks notebook Python syntax without executing GPU calls, and checks cost and timestamp edge cases plus readiness failures. It requires no network, credentials or GPU. The root GitHub Actions workflow runs these checks with Python 3.12.

Author checks on 2026-11-05: **8 offline tests passed** with Python 3.12; both notebooks passed syntax and nbformat validation; real notebook execution remains pending; all **6 Mermaid diagrams parsed**; **43 local Markdown links resolved** against the staged module and repository; shell syntax passed; `uv lock` and `uv sync --locked --extra notebooks` succeeded. Required storage/registry setup URLs were checked and corrected to canonical pages. An offline pass does not validate Docker builds, CUDA compatibility, provider flags, quota, storage permissions, actual Endpoint behavior or GPU performance.

## Required live dry run

- [ ] Build the worker for linux/amd64; record resolved base/image digest and dependency inventory.
- [ ] Validate Python/runtime compatibility and CUDA availability on the target L40S.
- [ ] Deploy the default Sana template with authentication; record image, GPU, region and timestamps.
- [ ] Observe startup/readiness, generate three assets, save evidence, stop/delete the Endpoint.
- [ ] Run the GPU notebook live; inspect output and timings, download artifacts and stop/delete Devlab.
- [ ] Validate the Job timeout against the installed CLI and the learner's spending limit.
- [ ] Verify the default S3 profile/bucket mount, run the Job, download PNGs/metrics/traces and confirm terminal state.
- [ ] Compare Endpoint vs Job configuration and images; document unknown Endpoint revision/runtime values.
- [ ] Supply a measured Sana comparison at N=1, 10, 100 (projections, not 100 required generations).
- [ ] Verify a representative optional modality separately before describing it as a supported code path.
- [ ] Have another person complete HW1 from a clean checkout; record time, spend and obstacles.
- [ ] Check all resource cleanup states and remaining storage charges.
- [ ] Confirm technical review, video links, homework form/hashtag and capstone submission instructions.

## Open evidence and scope choices

- Measured first-asset time, GPU peak memory, load/generation duration and spend: **[measure]**.
- Instructor-published worker digest and end-to-end bucket mount: **[verify: live build and Job]**.
- GPU notebook library combination: **[verify: live CUDA environment]**.
- Stock Endpoint resolved model commit: unknown; no client-side revision guarantee is made.
- CLI timeout discrepancy: cookbook 15m vs reference minimum 1h; **[verify: installed CLI]**.
- The proposed code contract refers to a future shared mediafactory event/metrics schema. This module makes its local schema explicit; **[verify: compatibility when that package exists]**.
- Model catalog is inherited from the module README; only Sana is implemented and inspected in depth. Optional model/licence/template entries need a freeze-day check.
- Duration and credit limits in the design are targets, not measured results. No spend cap is enforced by these examples.
- Lifecycle measurements use explicit observer timestamps. They do not provision resources or pretend to have measured startup before the client began.
- Proposal worksheet is module-local to avoid changing unrelated project instructions; it links to the existing project requirements.
- There is one root `pyproject.toml`; GPU direct dependencies and container versions are explicit. The root `uv.lock` resolves all extras. The GPU image still needs a recorded digest and live validation.
