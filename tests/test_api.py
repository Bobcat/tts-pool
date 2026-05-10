from __future__ import annotations

import base64
import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
import unittest


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
        self.assertEqual(enabled_model["capabilities"]["output_formats"], ["wav"])
        self.assertFalse(enabled_model["capabilities"]["streaming"])
        self.assertEqual(enabled_model["definition"]["target_inflight"], 2)

        disabled_model = payload["models"][1]
        self.assertEqual(disabled_model["name"], "disabled-tts")
        self.assertFalse(disabled_model["configured_enabled"])
        self.assertEqual(disabled_model["runtime_state"], "unloaded")
        self.assertFalse(disabled_model["is_loaded"])

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
