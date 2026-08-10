from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock


HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None

if HAS_FASTAPI:
    from fastapi.testclient import TestClient


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class ApiTests(unittest.TestCase):
    def _create_client(self, settings_text: str | None = None) -> TestClient:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                settings_text
                or (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "stub",\n'
                    '    "models": {\n'
                    '      "stub-tts": {"backend": "stub", "enabled": true, "target_inflight": 2},\n'
                    '      "disabled-tts": {"backend": "stub", "enabled": false}\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            previous = os.environ.get("TTS_POOL_SETTINGS_PATH")
            os.environ["TTS_POOL_SETTINGS_PATH"] = str(settings_path)
            try:
                sys.modules.pop("app.main", None)
                main = importlib.import_module("app.main")
                app = main.create_app(settings_path)
            finally:
                if previous is None:
                    os.environ.pop("TTS_POOL_SETTINGS_PATH", None)
                else:
                    os.environ["TTS_POOL_SETTINGS_PATH"] = previous
        return TestClient(app)

    def test_json_response_mode_returns_audio_response_envelope(self) -> None:
        client = self._create_client()

        response = client.post(
            "/v1/responses",
            json={
                "model": "stub-tts",
                "input": "Hello world",
                "language": "English",
                "stream": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "tts_response")
        self.assertEqual(payload["model"], "stub-tts")
        self.assertEqual(payload["audio"]["mime_type"], "audio/wav")
        self.assertEqual(payload["audio"]["sample_rate_hz"], 16000)
        self.assertGreater(len(base64.b64decode(payload["audio"]["data_base64"])), 44)
        self.assertIn("engine_queue_wait_ms", payload["metrics"])
        self.assertIn("backend_synthesis_wall_ms", payload["metrics"])
        self.assertIn("engine_total_wall_ms", payload["metrics"])
        self.assertIn("pool_total_wall_ms", payload["metrics"])
        self.assertEqual(payload["metadata"]["engine"], "stub")

    def test_streaming_mode_is_explicitly_unsupported(self) -> None:
        client = self._create_client()

        response = client.post(
            "/v1/responses",
            json={
                "model": "stub-tts",
                "input": "Hello world",
                "language": "English",
                "stream": True,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "stream_unsupported")

    def test_models_endpoint_returns_loaded_models(self) -> None:
        client = self._create_client()

        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": ["stub-tts"]})

    def test_admin_models_endpoint_returns_config_runtime_and_capabilities(self) -> None:
        client = self._create_client()

        response = client.get("/v1/admin/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["models"]), 2)

        enabled_model = payload["models"][0]
        self.assertEqual(enabled_model["name"], "stub-tts")
        self.assertEqual(enabled_model["resolved_backend"], "stub")
        self.assertTrue(enabled_model["configured_enabled"])
        self.assertEqual(enabled_model["runtime_state"], "loaded")
        self.assertTrue(enabled_model["is_loaded"])
        self.assertEqual(enabled_model["configured_target_inflight"], 2)
        self.assertEqual(enabled_model["runtime_inflight"], 0)
        self.assertTrue(enabled_model["accepting_new_requests"])
        self.assertEqual(enabled_model["fairness"]["keys"], [])
        self.assertEqual(enabled_model["fairness"]["rejected_per_key_limit"], 0)
        self.assertEqual(enabled_model["fairness"]["rejected_executor_limit"], 0)
        self.assertEqual(enabled_model["capabilities"]["output_formats"], ["wav"])
        self.assertFalse(enabled_model["capabilities"]["streaming"])
        self.assertEqual(enabled_model["definition"]["target_inflight"], 2)

        disabled_model = payload["models"][1]
        self.assertEqual(disabled_model["name"], "disabled-tts")
        self.assertFalse(disabled_model["configured_enabled"])
        self.assertEqual(disabled_model["runtime_state"], "unloaded")
        self.assertFalse(disabled_model["is_loaded"])

    def test_admin_models_reports_observed_vram_load_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                "{\n"
                '  "engine": {\n'
                '    "backend": "stub",\n'
                '    "models": {\n'
                '      "stub-tts": {"backend": "stub", "enabled": true, "target_inflight": 2}\n'
                "    }\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )

            from app.config import load_settings
            from app.engine.router import TTSRouterEngine

            settings = load_settings(settings_path)
            with mock.patch("app.engine.router.query_primary_gpu_used_mib", side_effect=[100, 123]):
                engine = TTSRouterEngine(settings)
            try:
                enabled_model = engine.admin_models_payload()["models"][0]
            finally:
                engine.close()

        self.assertEqual(enabled_model["name"], "stub-tts")
        self.assertEqual(enabled_model["vram_estimate_mib"], 23)
        self.assertEqual(enabled_model["vram_estimate_source"], "observed_load_delta")

    def test_load_and_unload_disabled_model(self) -> None:
        client = self._create_client()

        load_response = client.post("/v1/admin/models/disabled-tts/load")
        self.assertEqual(load_response.status_code, 200)
        self.assertEqual(load_response.json()["runtime_state"], "loaded")
        self.assertEqual(client.get("/v1/models").json(), {"models": ["disabled-tts", "stub-tts"]})

        unload_response = client.post("/v1/admin/models/disabled-tts/unload")
        self.assertEqual(unload_response.status_code, 200)
        self.assertEqual(unload_response.json()["runtime_state"], "unloaded")
        self.assertEqual(client.get("/v1/models").json(), {"models": ["stub-tts"]})

    def test_unloaded_model_response_is_rejected(self) -> None:
        client = self._create_client()

        response = client.post(
            "/v1/responses",
            json={
                "model": "disabled-tts",
                "input": "Hello world",
                "language": "English",
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "model_not_loaded")

    def test_admin_gpu_memory_endpoint_returns_envelope(self) -> None:
        client = self._create_client()

        response = client.get("/v1/admin/gpu-memory")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("gpus", payload)
        self.assertIn("models", payload)
        self.assertIn("error", payload)

    def test_admission_errors_are_returned_as_429(self) -> None:
        import app.main as main
        from app.engine import RequestAdmissionError

        class RejectingEngine:
            code = ""

            def complete(self, request):
                del request
                raise RequestAdmissionError(
                    status_code=429,
                    code=self.code,
                    message="pending queue is full",
                )

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(main, "build_engine", return_value=RejectingEngine()):
                app = main.create_app(settings_path)

        with TestClient(app) as client:
            for code in ("fairness_key_queue_full", "executor_queue_full"):
                with self.subTest(code=code):
                    RejectingEngine.code = code
                    response = client.post(
                        "/v1/responses",
                        json={
                            "model": "stub-tts",
                            "input": "Hello",
                            "language": "English",
                        },
                    )

                    self.assertEqual(response.status_code, 429)
                    self.assertEqual(response.json()["detail"]["code"], code)

    def test_inference_log_contains_normalized_fairness_key(self) -> None:
        import app.main as main
        from app.schemas import ResponseRequest

        request = ResponseRequest(
            model="stub-tts",
            input="Hello",
            language="English",
            fairness_key="  opaque-principal  ",
        )
        with mock.patch.object(main.LOGGER, "info") as info:
            main._log_inference("ttsresp_test", request, {"wall_ms": 1.0})

        payload = json.loads(info.call_args.args[1])
        self.assertEqual(payload["fairness_key"], "opaque-principal")
