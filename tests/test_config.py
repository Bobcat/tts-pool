from __future__ import annotations

import os
import tempfile
from pathlib import Path
import unittest

from app.config import load_settings


class ConfigTests(unittest.TestCase):
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
                    '            "reference_audio_match": "voice_and_pace",\n'
                    '            "reference_duration_s": 4.0,\n'
                    '            "voice_preset": "configured",\n'
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
        self.assertEqual(warmup_case.reference_audio_match, "voice_and_pace")
        self.assertEqual(warmup_case.cfg_value, 1.5)
        self.assertEqual(warmup_case.inference_timesteps, 6)
