from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
import unittest

from app.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_fairness_defaults_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text("{}\n", encoding="utf-8")

            fairness = load_settings(settings_path).engine.fairness

        self.assertEqual(fairness.default_weight, 1.0)
        self.assertEqual(fairness.weights, {})
        self.assertEqual(fairness.soft_max_inflight_per_key, 1)
        self.assertEqual(fairness.max_pending_per_key, 4)
        self.assertEqual(fairness.max_pending_per_executor, 8)
        self.assertEqual(fairness.idle_state_ttl_s, 300.0)

    def test_fairness_settings_and_weights_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "engine": {
                            "fairness": {
                                "default_weight": 1.5,
                                "weights": {"  priority-workload  ": 2.0},
                                "soft_max_inflight_per_key": 2,
                                "max_pending_per_key": 3,
                                "max_pending_per_executor": 6,
                                "idle_state_ttl_s": 90,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            fairness = load_settings(settings_path).engine.fairness

        self.assertEqual(fairness.default_weight, 1.5)
        self.assertEqual(fairness.weights, {"priority-workload": 2.0})
        self.assertEqual(fairness.soft_max_inflight_per_key, 2)
        self.assertEqual(fairness.max_pending_per_key, 3)
        self.assertEqual(fairness.max_pending_per_executor, 6)
        self.assertEqual(fairness.idle_state_ttl_s, 90.0)

    def test_invalid_fairness_settings_are_rejected(self) -> None:
        invalid_payloads = (
            {"default_weight": 0},
            {"default_weight": "inf"},
            {"weights": []},
            {"weights": {" ": 1.0}},
            {"weights": {"a": -1.0}},
            {"soft_max_inflight_per_key": 0},
            {"max_pending_per_key": 1.5},
            {"max_pending_per_executor": -1},
            {"idle_state_ttl_s": "nan"},
        )
        for fairness_payload in invalid_payloads:
            with self.subTest(fairness_payload=fairness_payload):
                with tempfile.TemporaryDirectory() as tmpdir:
                    settings_path = Path(tmpdir) / "settings.json"
                    settings_path.write_text(
                        json.dumps({"engine": {"fairness": fairness_payload}}),
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        load_settings(settings_path)

    def test_load_settings_merges_local_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            local_path = Path(tmpdir) / "local.json"
            settings_path.write_text(
                (
                    "{\n"
                    '  "service": {"port": 8020},\n'
                    '  "engine": {\n'
                    '    "backend": "stub",\n'
                    '    "models": {\n'
                    '      "stub-tts": {"backend": "stub", "enabled": true, "target_inflight": 1}\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            local_path.write_text(
                (
                    "{\n"
                    '  "service": {"port": 8123},\n'
                    '  "engine": {\n'
                    '    "models": {\n'
                    '      "stub-tts": {"target_inflight": 3}\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            settings = load_settings(settings_path)

        self.assertEqual(settings.service.port, 8123)
        self.assertEqual(settings.engine.models["stub-tts"].target_inflight, 3)

    def test_settings_path_env_var_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                (
                    "{\n"
                    '  "service": {"port": 8099},\n'
                    '  "engine": {"models": {"stub-tts": {"backend": "stub"}}}\n'
                    "}\n"
                ),
                encoding="utf-8",
            )
            previous = os.environ.get("TTS_POOL_SETTINGS_PATH")
            os.environ["TTS_POOL_SETTINGS_PATH"] = str(settings_path)
            try:
                settings = load_settings()
            finally:
                if previous is None:
                    os.environ.pop("TTS_POOL_SETTINGS_PATH", None)
                else:
                    os.environ["TTS_POOL_SETTINGS_PATH"] = previous

        self.assertEqual(settings.service.port, 8099)
        self.assertIn("stub-tts", settings.engine.models)

    def test_voxcpm2_warmup_cases_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "models": {\n'
                    '      "voxcpm2": {\n'
                    '        "backend": "voxcpm2",\n'
                    '        "voxcpm2_warmup_enabled": true,\n'
                    '        "voxcpm2_warmup_cases": [\n'
                    '          {\n'
                    '            "name": "medium_ref_6",\n'
                    '            "text": "Warm this request shape.",\n'
                    '            "reference_audio": true,\n'
                    '            "reference_duration_s": 4.0,\n'
                    '            "cfg_value": 1.5,\n'
                    '            "inference_timesteps": 6\n'
                    "          }\n"
                    "        ]\n"
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            settings = load_settings(settings_path)

        model_settings = settings.engine.models["voxcpm2"]
        self.assertTrue(model_settings.voxcpm2_warmup_enabled)
        self.assertEqual(len(model_settings.voxcpm2_warmup_cases), 1)
        warmup_case = model_settings.voxcpm2_warmup_cases[0]
        self.assertEqual(warmup_case.name, "medium_ref_6")
        self.assertTrue(warmup_case.reference_audio)
        self.assertEqual(warmup_case.cfg_value, 1.5)
        self.assertEqual(warmup_case.inference_timesteps, 6)

    def test_kokoro_warmup_settings_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "models": {\n'
                    '      "kokoro": {\n'
                    '        "backend": "kokoro",\n'
                    '        "kokoro_warmup_enabled": true,\n'
                    '        "kokoro_warmup_languages": ["English", "Chinese", "English"]\n'
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            settings = load_settings(settings_path)

        model_settings = settings.engine.models["kokoro"]
        self.assertTrue(model_settings.kokoro_warmup_enabled)
        self.assertEqual(model_settings.kokoro_warmup_languages, ("English", "Chinese"))

    def test_nanovllm_model_settings_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "models": {\n'
                    '      "nanovllm_voxcpm": {\n'
                    '        "backend": "nanovllm_voxcpm",\n'
                    '        "target_inflight": 4,\n'
                    '        "nanovllm_model_id": "/models/VoxCPM2",\n'
                    '        "nanovllm_devices": [0, 1],\n'
                    '        "nanovllm_max_num_seqs": 8,\n'
                    '        "nanovllm_max_num_batched_tokens": 2048,\n'
                    '        "nanovllm_max_model_len": 1024,\n'
                    '        "nanovllm_gpu_memory_utilization": 0.75,\n'
                    '        "nanovllm_inference_timesteps": 6,\n'
                    '        "nanovllm_max_generate_length": 128,\n'
                    '        "nanovllm_temperature": 0.8,\n'
                    '        "nanovllm_cfg_value": 1.5,\n'
                    '        "nanovllm_reference_max_duration_s": 4.0,\n'
                    '        "nanovllm_warmup_enabled": true,\n'
                    '        "nanovllm_warmup_cases": [\n'
                    '          {\n'
                    '            "name": "short_ref_voice",\n'
                    '            "text": "Warm this NanoVLLM shape.",\n'
                    '            "reference_audio": true,\n'
                    '            "reference_duration_s": 2.0,\n'
                    '            "cfg_value": 1.25,\n'
                    '            "temperature": 0.01,\n'
                    '            "max_generate_length": 64\n'
                    "          }\n"
                    "        ]\n"
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            settings = load_settings(settings_path)

        model_settings = settings.engine.models["nanovllm_voxcpm"]
        self.assertEqual(model_settings.backend, "nanovllm_voxcpm")
        self.assertEqual(model_settings.nanovllm_model_id, "/models/VoxCPM2")
        self.assertEqual(model_settings.nanovllm_devices, (0, 1))
        self.assertEqual(model_settings.nanovllm_max_num_seqs, 8)
        self.assertEqual(model_settings.nanovllm_max_num_batched_tokens, 2048)
        self.assertEqual(model_settings.nanovllm_max_model_len, 1024)
        self.assertEqual(model_settings.nanovllm_gpu_memory_utilization, 0.75)
        self.assertEqual(model_settings.nanovllm_inference_timesteps, 6)
        self.assertEqual(model_settings.nanovllm_max_generate_length, 128)
        self.assertEqual(model_settings.nanovllm_temperature, 0.8)
        self.assertEqual(model_settings.nanovllm_cfg_value, 1.5)
        self.assertEqual(model_settings.nanovllm_reference_max_duration_s, 4.0)
        self.assertTrue(model_settings.nanovllm_warmup_enabled)
        self.assertEqual(len(model_settings.nanovllm_warmup_cases), 1)
        warmup_case = model_settings.nanovllm_warmup_cases[0]
        self.assertEqual(warmup_case.name, "short_ref_voice")
        self.assertTrue(warmup_case.reference_audio)
        self.assertEqual(warmup_case.reference_duration_s, 2.0)
        self.assertEqual(warmup_case.cfg_value, 1.25)
        self.assertEqual(warmup_case.temperature, 0.01)
        self.assertEqual(warmup_case.max_generate_length, 64)
