from __future__ import annotations

import importlib
import importlib.util
import json
import os
import socket
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

    def test_http_synthesis_route_is_removed(self) -> None:
        client = self._create_client()

        response = client.post(
            "/v1/responses",
            json={
                "model": "stub-tts",
                "input": "Hello world",
                "language": "English",
            },
        )

        self.assertEqual(response.status_code, 404)

    def test_models_endpoint_returns_loaded_models(self) -> None:
        client = self._create_client()

        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": ["stub-tts"]})

    def test_fastapi_lifespan_starts_and_stops_grpc_server(self) -> None:
        import grpc
        from grpc_health.v1 import health_pb2
        from grpc_health.v1 import health_pb2_grpc

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            grpc_port = probe.getsockname()[1]
        client = self._create_client(
            json.dumps(
                {
                    "service": {
                        "grpc": {
                            "enabled": True,
                            "host": "127.0.0.1",
                            "port": grpc_port,
                            "shutdown_grace_s": 0.1,
                        }
                    },
                    "engine": {
                        "backend": "stub",
                        "models": {
                            "stub-tts": {
                                "backend": "stub",
                                "enabled": True,
                            }
                        },
                    },
                }
            )
        )

        with client:
            self.assertEqual(client.app.state.grpc_server.bound_port, grpc_port)
            with grpc.insecure_channel(f"127.0.0.1:{grpc_port}") as channel:
                response = health_pb2_grpc.HealthStub(channel).Check(
                    health_pb2.HealthCheckRequest(service="tts.v1.TTSService"),
                    timeout=1.0,
                )
            self.assertEqual(response.status, health_pb2.HealthCheckResponse.SERVING)

        self.assertIsNone(client.app.state.grpc_server.bound_port)

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

    def test_admin_gpu_memory_endpoint_returns_envelope(self) -> None:
        client = self._create_client()

        response = client.get("/v1/admin/gpu-memory")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("gpus", payload)
        self.assertIn("models", payload)
        self.assertIn("error", payload)
