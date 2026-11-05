"""Offline unit tests for Module 1 code and notebooks using mocks and fixtures (no --fake flags in lesson code)."""
import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE = REPO_ROOT / "01-consistent-media-generation-on-serverless-gpus" / "code"
sys.path.insert(0, str(CODE))

import call_endpoint
from call_endpoint import ImageGenerator, wait_ready
from common import DEFAULT_CONFIG, calculate_cost, config, interval, write_json
import generate
import measure_endpoint
import measure_job
from measure_job import enrich

# Minimal 1x1 PNG test fixture used only inside unit tests.
SAMPLE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgaPgPAAIDAYAkYfWXAAAAAElFTkSuQmCC"
)


class Module1Tests(unittest.TestCase):
    def test_notebook_syntax(self):
        for path in CODE.glob("*.ipynb"):
            for cell in json.loads(path.read_text())["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), str(path), "exec")

    def test_generate_worker_with_mocked_torch_and_pipeline(self):
        fake_image = MagicMock()
        fake_image.save.side_effect = lambda p: Path(p).write_bytes(SAMPLE_PNG)
        fake_pipe = MagicMock(return_value=types.SimpleNamespace(images=[fake_image]))

        fake_cuda = MagicMock()
        fake_cuda.is_available.return_value = True
        fake_cuda.get_device_name.return_value = "NVIDIA L40S"
        fake_cuda.max_memory_allocated.return_value = 10_908_000_000
        fake_torch = types.SimpleNamespace(
            cuda=fake_cuda,
            version=types.SimpleNamespace(cuda="13.0"),
            __version__="2.13.0",
            float16="float16",
            bfloat16="bfloat16",
            Generator=lambda device="cuda": MagicMock(),
        )

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.object(generate, "load_pipeline", return_value=fake_pipe),
            patch("importlib.metadata.version", return_value="0.39.0"),
            redirect_stdout(io.StringIO()),
        ):
            out = Path(temporary) / "job_run"
            generate.run(DEFAULT_CONFIG, n=3, out=out, mode="job")
            m = json.loads((out / "metrics.json").read_text())
            self.assertEqual(m["status"], "completed")
            self.assertEqual(len(m["assets"]), 3)
            self.assertEqual(len(m["latency_s"]), 3)
            self.assertIsNotNone(m["load_s"])
            self.assertEqual(m["env"]["gpu"], "NVIDIA L40S")
            self.assertAlmostEqual(m["env"]["peak_vram_gb"], 10.908)
            self.assertTrue((out / "trace.jsonl").exists())

    def test_measure_endpoint_and_measure_job_cli(self):
        cfg = config(DEFAULT_CONFIG)
        payload = {"data": [{"b64_json": base64.b64encode(SAMPLE_PNG).decode()}]}
        responses = [{"data": [{"id": cfg["model"]}]}, payload, payload]
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                os.environ,
                {"BASE_URL": "https://example.test", "TOKEN": "secret-token"},
            ),
            patch("call_endpoint.request", side_effect=responses),
            redirect_stdout(io.StringIO()),
        ):
            out = Path(temporary) / "ep_run"
            with patch.object(
                sys,
                "argv",
                [
                    "measure_endpoint.py",
                    "--out",
                    str(out),
                    "--n",
                    "2",
                    "--created-at",
                    "2026-01-01T10:00:00Z",
                    "--running-at",
                    "2026-01-01T10:03:00Z",
                ],
            ):
                measure_endpoint.main()
            m = json.loads((out / "metrics.json").read_text())
            self.assertEqual(m["startup_s"], 180.0)
            self.assertIsNotNone(m["ready_wait_s"])
            self.assertEqual(len(m["latency_s"]), 2)

            # Also verify measure_job.py CLI on a metrics.json file
            job_metrics = out / "job_metrics.json"
            write_json(job_metrics, {"status": "completed", "mode": "job"})
            with patch.object(
                sys,
                "argv",
                [
                    "measure_job.py",
                    str(job_metrics),
                    "--created-at",
                    "2026-11-16T10:00:00Z",
                    "--running-at",
                    "2026-11-16T10:03:45Z",
                    "--finished-at",
                    "2026-11-16T10:04:30Z",
                ],
            ):
                measure_job.main()
            jm = json.loads(job_metrics.read_text())
            self.assertEqual(jm["startup_s"], 225.0)
            self.assertEqual(jm["billable_s"], 270.0)

    def test_cost_function_boundaries_and_idle(self):
        r = calculate_cost(60, 30, [10, 12, 11], 1.0, [1, 10, 100], api_price=0.01)
        self.assertAlmostEqual(r["marginal_compute_usd"], 11 / 3600)
        self.assertAlmostEqual(r["fixed_run_usd"], 90 / 3600)
        self.assertEqual(r["breakeven_assets"], 4)
        idle = calculate_cost(60, 30, [10, 12, 11], 1.0, [10], idle_s=600)
        self.assertAlmostEqual(
            idle["projected_compute_usd_per_asset"]["10"]
            - r["projected_compute_usd_per_asset"]["10"],
            600 / 3600 / 10,
        )
        self.assertEqual(
            calculate_cost(60, 30, [10, 12, 11], 1.0, [1], api_price=0.0)["breakeven_assets"], 0
        )
        sana = calculate_cost(
            startup_s=225.17,
            ready_s=39.89,
            latency_s=[4.418, 4.465, 4.371],
            price_per_hour=1.5484,
            counts=(1, 10, 100),
            api_price=0.04,
        )
        self.assertEqual(sana["breakeven_assets"], 3)
        with self.assertRaises(ValueError):
            calculate_cost(None, 30, [10], 1.0, [1])
        with self.assertRaises(ValueError):
            calculate_cost(60, 30, [], 1.0, [1])
        with self.assertRaises(ValueError):
            calculate_cost(60, 30, [10], float("nan"), [1])

    def test_lifecycle(self):
        self.assertEqual(interval("2026-01-01T01:00:00+01:00", "2026-01-01T00:01:00Z"), 60)
        for end in ("2025-01-01T00:00:00Z", "2026-01-01T00:01:00"):
            with self.assertRaises(ValueError):
                interval("2026-01-01T00:00:00Z", end)
        with self.assertRaises(ValueError):
            enrich({}, "2026-01-01T00:00:00Z", "2026-01-01T00:02:00Z", "2026-01-01T00:01:00Z")

    def test_readiness_retry_auth_and_deadline(self):
        transient = urllib.error.HTTPError(
            "https://example.test", 502, "starting", {}, io.BytesIO()
        )
        with (
            patch(
                "call_endpoint.request",
                side_effect=[transient, {"data": [{"id": "sana"}]}],
            ) as req,
            patch("call_endpoint.time.sleep"),
        ):
            self.assertEqual(wait_ready("https://example.test", "token")["data"][0]["id"], "sana")
            self.assertEqual(req.call_count, 2)
        auth = urllib.error.HTTPError(
            "https://example.test", 401, "unauthorized", {}, io.BytesIO()
        )
        with patch("call_endpoint.request", side_effect=auth) as req:
            with self.assertRaises(urllib.error.HTTPError):
                wait_ready("https://example.test", "token")
            self.assertEqual(req.call_count, 1)
        with patch("call_endpoint.time.monotonic", side_effect=[0, 601]):
            with self.assertRaises(TimeoutError):
                wait_ready("https://example.test", "token", 600)

    def test_live_client_contract_with_mock_transport(self):
        cfg = config(DEFAULT_CONFIG)
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                os.environ,
                {"BASE_URL": "https://example.test", "TOKEN": "secret-test-token"},
            ),
        ):
            out = Path(temporary) / "client"
            payload = {"data": [{"b64_json": base64.b64encode(SAMPLE_PNG).decode()}]}
            responses = [{"data": [{"id": cfg["model"]}]}, payload, payload, payload]
            with (
                patch("call_endpoint.request", side_effect=responses) as req,
                redirect_stdout(io.StringIO()),
            ):
                call_endpoint.run(DEFAULT_CONFIG, 3, out)
            self.assertEqual(req.call_count, 4)
            for call in req.call_args_list[1:]:
                body = call.args[3]
                self.assertEqual(body["seed"], cfg["seed"])
                self.assertEqual(body["n"], 1)
                self.assertEqual(body["size"], "1024x1024")
            m = json.loads((out / "metrics.json").read_text())
            self.assertIsNone(m["config"]["model_revision"])
            self.assertIsNone(m["ready_wait_s"])
            self.assertEqual(len(m["latency_s"]), 3)
            self.assertNotIn(
                "secret-test-token",
                (out / "metrics.json").read_text() + (out / "trace.jsonl").read_text(),
            )

    def test_image_generator_class(self):
        cfg = config(DEFAULT_CONFIG)
        payload = {"data": [{"b64_json": base64.b64encode(SAMPLE_PNG).decode()}]}
        with patch("call_endpoint.request", return_value=payload) as req:
            gen = ImageGenerator("https://example.test", "token", cfg["model"])
            png = gen.generate_image(cfg["prompt"], seed=7)
            self.assertEqual(png, SAMPLE_PNG)
            self.assertEqual(req.call_count, 1)

    def test_client_failure_leaves_evidence(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"BASE_URL": "https://example.test", "TOKEN": "test"}),
        ):
            out = Path(temporary) / "failed"
            with (
                patch("call_endpoint.wait_ready", side_effect=TimeoutError("unavailable")),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(TimeoutError):
                    call_endpoint.run(DEFAULT_CONFIG, 1, out)
            self.assertEqual(json.loads((out / "metrics.json").read_text())["status"], "failed")
            self.assertIn("run.failed", (out / "trace.jsonl").read_text())

    def test_config_pin(self):
        self.assertEqual(len(config(DEFAULT_CONFIG)["model_revision"]), 40)

    def test_module_markdown_links(self):
        module_dir = REPO_ROOT / "01-consistent-media-generation-on-serverless-gpus"
        pattern = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)")
        for md_path in list(module_dir.glob("*.md")) + [module_dir / "code" / "README.md"]:
            text = md_path.read_text()
            for target in pattern.findall(text):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                resolved = (md_path.parent / target).resolve()
                self.assertTrue(
                    resolved.exists(),
                    f"Broken relative link in {md_path.name}: {target} -> {resolved}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
