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
