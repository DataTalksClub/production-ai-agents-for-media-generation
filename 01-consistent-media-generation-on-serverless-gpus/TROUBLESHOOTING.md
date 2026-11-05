# Common failures

| Symptom | Check | Recovery |
|---|---|---|
| uv uses a different Python | `uv run python --version` | Install/select Python 3.12; root project requires it. |
| 401/403 from Endpoint | Endpoint token, auth configuration, correct URL | Set the correct token; do not disable auth to hide the error. |
| Temporary 502/503 | Resource state and container logs | Let bounded readiness polling run; on timeout inspect download/load, then stop the resource. |
| Readiness succeeds but model mismatch | `/v1/models` model ID vs generation.json | Use the Sana Endpoint or adapt the client's request to the intended server. |
| Generation times out | Endpoint logs, resource still running | Inspect before retrying; a timeout does not prove work was cancelled. |
| CUDA unavailable | GPU allocation and torch CUDA build | Select a GPU runtime with CUDA PyTorch; use `--fake` on CPU. |
| Black images | Component precision, pinned model/runtime | Keep Sana text encoder/VAE in bf16 as in the inspected recipe; confirm live compatibility. |
| Out of memory | Other GPU processes and generation settings | Stop other work or lower one setting in a copied config; record the change rather than comparing it as identical. |
| Quota/capacity failure | Project, region, Compute quotas | Cancel pending attempts, resolve quota/capacity, then retry one run. |
| Container image pull fails | Image digest, architecture and registry permissions | Verify the pushed manifest and access from the project. |
| Job has no bucket output | Mount, AWS default profile, permission, worker logs | Fix storage configuration; a local container path alone is not persistence. |
| Output directory already exists | Reused RUN_ID or --out | Choose a new run ID; preserve old evidence. |
| Job rejects short timeout | Installed CLI help vs cookbook example | Use the valid explicit duration; actively cancel to meet your budget. |
| Cost tool rejects metrics | fake flag, missing timestamps, missing latency | Collect live measurements; do not fill unknowns with zero. |
| Notebook cannot import worker | Notebook working directory | Start from root, module or code directory as documented. |
| Bucket/file remains after cleanup | Storage lifecycle is separate | Download needed evidence, then delete unwanted objects/resources explicitly. |

After any failed live run, check remote state and stop/delete Endpoints or Devlabs, or cancel active Jobs. Python exceptions do not perform cloud cleanup.
