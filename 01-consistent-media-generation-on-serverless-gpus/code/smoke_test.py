"""Offline checks for the lesson's real entry points and error boundaries."""
import argparse
import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zlib

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from call_endpoint import wait_ready
from common import config, DEFAULT_CONFIG, interval
from cost import calculate
from generate import FAKE_PNG
from measure_job import enrich

class LessonChecks(unittest.TestCase):
    def test_python_entrypoints_and_notebook_syntax(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            for name in ('common.py','generate.py','call_endpoint.py','measure_endpoint.py','measure_job.py','cost.py'):
                command = [sys.executable, str(CODE/name), '--fake']
                if name in ('generate.py','call_endpoint.py','measure_endpoint.py'):
                    command += ['--out', str(tmp/name.removesuffix('.py')), '--n', '3']
                subprocess.run(command, cwd=tmp, check=True, capture_output=True, text=True, timeout=30)
            for name in ('generate','call_endpoint','measure_endpoint'):
                m = json.loads((tmp/name/'metrics.json').read_text())
                self.assertTrue(m['fake'])
                self.assertEqual(m['status'], 'completed')
                self.assertEqual(len(m['assets']), 3)
                self.assertEqual(m['latency_s'], [])
                self.assertIsNone(m['load_s'])
                self.assertTrue((tmp/name/'trace.jsonl').exists())
                with self.assertRaises(ValueError):
                    calculate(m, 1, [1])
            subprocess.run(['bash', '-n', str(CODE/'submit_job.sh')], check=True, capture_output=True)
            # Live notebooks are compiled, never executed by the offline suite.
            for path in CODE.glob('*.ipynb'):
                for cell in json.loads(path.read_text())['cells']:
                    if cell['cell_type'] == 'code':
                        compile(''.join(cell['source']), str(path), 'exec')

    def test_fake_png_crc(self):
        self.assertEqual(FAKE_PNG[:8], b'\x89PNG\r\n\x1a\n')
        pos = 8
        while pos < len(FAKE_PNG):
            length = struct.unpack('>I', FAKE_PNG[pos:pos+4])[0]
            block = FAKE_PNG[pos+4:pos+8+length]
            crc = struct.unpack('>I', FAKE_PNG[pos+8+length:pos+12+length])[0]
            self.assertEqual(zlib.crc32(block), crc)
            pos += 12+length

    def test_cost_boundaries_and_idle(self):
        m = {'status':'completed','mode':'job','startup_s':60,'load_s':30,'latency_s':[10,12,11]}
        r = calculate(m, 1, [1,10,100], api_price=0.01)
        self.assertAlmostEqual(r['marginal_compute_usd'],11/3600)
        self.assertAlmostEqual(r['fixed_run_usd'],90/3600)
        self.assertEqual(r['breakeven_assets'],4)
        idle = calculate(m, 1, [10], idle_s=600)
        self.assertAlmostEqual(idle['projected_compute_usd_per_asset']['10']-r['projected_compute_usd_per_asset']['10'],600/3600/10)
        self.assertEqual(calculate(m,1,[1],api_price=0)['breakeven_assets'],0)
        with self.assertRaises(ValueError):
            calculate({**m,'startup_s':None},1,[1])
        with self.assertRaises(ValueError):
            calculate({**m,'status':'failed'},1,[1])
        with self.assertRaises(ValueError):
            calculate(m,float('nan'),[1])

    def test_lifecycle(self):
        self.assertEqual(interval('2026-01-01T01:00:00+01:00','2026-01-01T00:01:00Z'),60)
        for end in ('2025-01-01T00:00:00Z','2026-01-01T00:01:00'):
            with self.assertRaises(ValueError):
                interval('2026-01-01T00:00:00Z',end)
        with self.assertRaises(ValueError):
            enrich({},'2026-01-01T00:00:00Z','2026-01-01T00:02:00Z','2026-01-01T00:01:00Z')

    def test_readiness_retry_auth_and_deadline(self):
        transient = urllib.error.HTTPError('https://example.test',502,'starting',{},io.BytesIO())
        with patch('call_endpoint.request', side_effect=[transient, {'data':[{'id':'sana'}]}]) as req, patch('call_endpoint.time.sleep'):
            self.assertEqual(wait_ready('https://example.test','token')['data'][0]['id'],'sana')
            self.assertEqual(req.call_count,2)
        auth = urllib.error.HTTPError('https://example.test',401,'unauthorized',{},io.BytesIO())
        with patch('call_endpoint.request', side_effect=auth) as req:
            with self.assertRaises(urllib.error.HTTPError):
                wait_ready('https://example.test','token')
            self.assertEqual(req.call_count,1)
        with patch('call_endpoint.time.monotonic',side_effect=[0,601]):
            with self.assertRaises(TimeoutError):
                wait_ready('https://example.test','token',600)

    def test_live_client_contract_with_mock_transport(self):
        from call_endpoint import run
        cfg = config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"BASE_URL":"https://example.test", "TOKEN":"secret-test-token"}):
            out = Path(temporary)/"client"
            payload = {"data":[{"b64_json":base64.b64encode(FAKE_PNG).decode()}]}
            responses = [{"data":[{"id":cfg["model"]}]},payload,payload,payload]
            with patch('call_endpoint.request',side_effect=responses) as req, redirect_stdout(io.StringIO()):
                run(DEFAULT_CONFIG, 3, out, False)
            self.assertEqual(req.call_count,4)
            for call in req.call_args_list[1:]:
                body=call.args[3]
                self.assertEqual(body['seed'],cfg['seed'])
                self.assertEqual(body['n'],1)
                self.assertEqual(body['size'],'1024x1024')
            m=json.loads((out/'metrics.json').read_text())
            self.assertIsNone(m['config']['model_revision'])
            self.assertIsNone(m['ready_wait_s'])
            self.assertEqual(len(m['latency_s']),3)
            self.assertNotIn('secret-test-token',(out/'metrics.json').read_text()+(out/'trace.jsonl').read_text())

    def test_client_failure_leaves_evidence(self):
        from call_endpoint import run
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"BASE_URL":"https://example.test", "TOKEN":"test"}):
            out=Path(temporary)/'failed'
            with patch('call_endpoint.wait_ready',side_effect=TimeoutError('unavailable')), redirect_stdout(io.StringIO()):
                with self.assertRaises(TimeoutError):
                    run(DEFAULT_CONFIG,1,out,False)
            self.assertEqual(json.loads((out/'metrics.json').read_text())['status'],'failed')
            self.assertIn('run.failed',(out/'trace.jsonl').read_text())

    def test_config_pin(self):
        self.assertEqual(len(config(DEFAULT_CONFIG)['model_revision']),40)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fake', action='store_true', required=True)
    p.parse_args()
    unittest.main(argv=[sys.argv[0]], verbosity=2)
