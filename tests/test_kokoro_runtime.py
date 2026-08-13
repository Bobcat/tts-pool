from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from app.config import ModelSettings
from app.engine.kokoro import KokoroTTSRuntime
from app.schemas import ReferenceAudio
from app.schemas import ResponseRequest
from app.schemas import VoiceSpec


class FakeModel:
    device = "cpu"


class FakeKokoroResult:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio
        self.graphemes = "hello"
        self.phonemes = "HH AH L OW"


class FakePipeline:
    def __init__(self) -> None:
        self.voices: dict[str, bool] = {}
        self.calls: list[dict[str, object]] = []

    def __call__(self, text: str, *, voice: str, speed: float):
        self.calls.append({"text": text, "voice": voice, "speed": speed})
        self.voices[voice] = True
        return [FakeKokoroResult(np.zeros(2400, dtype=np.float32))]


class KokoroRuntimeTests(unittest.TestCase):
    def test_synthesize_uses_direct_kokoro_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir)
            voices_dir = model_root / "voices"
            voices_dir.mkdir()
            (voices_dir / "af_heart.pt").write_bytes(b"voice")
            pipeline = FakePipeline()
            runtime = KokoroTTSRuntime(
                model_name="kokoro",
                model_settings=ModelSettings(
                    backend="kokoro",
                    model_path=str(model_root),
                    kokoro_speed=1.0,
                    voice_presets={"English": "af_heart"},
                ),
            )
            runtime._model = FakeModel()
            runtime._pipelines["a"] = pipeline

            result = runtime.synthesize(
                ResponseRequest(
                    model="kokoro",
                    input="Hello world.",
                    language="English",
                    voice=VoiceSpec(instructions="Ignored for Kokoro."),
                    generation={"kokoro": {"speed": 1.25}},
                )
            )

        self.assertEqual(result.sample_rate_hz, 24000)
        self.assertEqual(result.duration_ms, 100)
        self.assertGreater(len(result.audio), 44)
        self.assertEqual(result.metadata["engine"], "kokoro")
        self.assertEqual(result.metadata["voice"], "af_heart")
        self.assertEqual(result.metadata["language_code"], "a")
        self.assertEqual(result.metadata["speed"], 1.25)
        self.assertTrue(result.metadata["voice_instructions_ignored"])
        self.assertEqual(pipeline.calls[0]["text"], "Hello world.")
        self.assertEqual(pipeline.calls[0]["speed"], 1.25)
        self.assertIn("kokoro_total_wall_ms", result.metrics)
        self.assertIn("kokoro_wav_encode_ms", result.metrics)

    def test_reference_audio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir)
            (model_root / "voices").mkdir()
            runtime = KokoroTTSRuntime(
                model_name="kokoro",
                model_settings=ModelSettings(backend="kokoro", model_path=str(model_root)),
            )

            with self.assertRaisesRegex(ValueError, "reference_audio"):
                runtime.synthesize(
                    ResponseRequest(
                        model="kokoro",
                        input="Hello world.",
                        language="English",
                        voice=VoiceSpec(
                            reference_audio=ReferenceAudio(
                                mime_type="audio/wav",
                                data=b"not used",
                            )
                        ),
                    )
                )

    def test_load_warms_configured_languages_and_voices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir)
            voices_dir = model_root / "voices"
            voices_dir.mkdir()
            (voices_dir / "af_heart.pt").write_bytes(b"voice")
            (voices_dir / "bf_emma.pt").write_bytes(b"voice")
            english_pipeline = FakePipeline()
            british_pipeline = FakePipeline()
            runtime = KokoroTTSRuntime(
                model_name="kokoro",
                model_settings=ModelSettings(
                    backend="kokoro",
                    model_path=str(model_root),
                    kokoro_speed=1.0,
                    kokoro_warmup_enabled=True,
                    voice_presets={
                        "English": "af_heart",
                        "British English": "bf_emma",
                    },
                ),
            )
            runtime._model = FakeModel()
            runtime._pipelines["a"] = english_pipeline
            runtime._pipelines["b"] = british_pipeline

            runtime.load()

        self.assertEqual(english_pipeline.calls[0]["text"], "Ready.")
        self.assertEqual(english_pipeline.calls[0]["voice"], str(voices_dir / "af_heart.pt"))
        self.assertEqual(british_pipeline.calls[0]["text"], "Ready.")
        self.assertEqual(british_pipeline.calls[0]["voice"], str(voices_dir / "bf_emma.pt"))

    def test_load_respects_kokoro_warmup_language_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir)
            voices_dir = model_root / "voices"
            voices_dir.mkdir()
            (voices_dir / "af_heart.pt").write_bytes(b"voice")
            (voices_dir / "bf_emma.pt").write_bytes(b"voice")
            english_pipeline = FakePipeline()
            british_pipeline = FakePipeline()
            runtime = KokoroTTSRuntime(
                model_name="kokoro",
                model_settings=ModelSettings(
                    backend="kokoro",
                    model_path=str(model_root),
                    kokoro_speed=1.0,
                    kokoro_warmup_enabled=True,
                    kokoro_warmup_languages=("British English",),
                    voice_presets={
                        "English": "af_heart",
                        "British English": "bf_emma",
                    },
                ),
            )
            runtime._model = FakeModel()
            runtime._pipelines["a"] = english_pipeline
            runtime._pipelines["b"] = british_pipeline

            runtime.load()

        self.assertEqual(english_pipeline.calls, [])
        self.assertEqual(british_pipeline.calls[0]["voice"], str(voices_dir / "bf_emma.pt"))


if __name__ == "__main__":
    unittest.main()
