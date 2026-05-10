from __future__ import annotations

import base64
import io
import unittest
import wave
from unittest import mock

import numpy as np

from app.config import ModelSettings
from app.engine.voxcpm2 import VoxCPM2TTSRuntime
from app.schemas import ReferenceAudio
from app.schemas import ResponseRequest
from app.schemas import VoiceSpec


class FakeTTSModel:
    sample_rate = 16000
    device = "cuda"


class FakeVoxCPM:
    def __init__(self) -> None:
        self.tts_model = FakeTTSModel()
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return np.zeros(1600, dtype=np.float32)


class VoxCPM2ReferenceTests(unittest.TestCase):
    def test_reference_audio_is_clipped_to_request_max_duration(self) -> None:
        model = FakeVoxCPM()
        runtime = VoxCPM2TTSRuntime(
            model_name="voxcpm2",
            model_settings=ModelSettings(
                backend="voxcpm2",
                voxcpm2_reference_max_duration_s=8.0,
            ),
        )
        runtime._model = model
        reference = ReferenceAudio(
            mime_type="audio/wav",
            data_base64=base64.b64encode(_silent_wav(seconds=2.0)).decode("ascii"),
            max_duration_s=1.0,
        )

        result = runtime.synthesize(
            ResponseRequest(
                model="voxcpm2",
                input="Hallo wereld",
                language="Dutch",
                voice=VoiceSpec(preset="configured", reference_audio=reference),
            )
        )

        self.assertEqual(result.metadata["reference_source_duration_ms"], 2000)
        self.assertTrue(result.metadata["reference_clipped"])
        self.assertLessEqual(result.metadata["reference_duration_ms"], 1000)
        self.assertIsNotNone(model.generate_kwargs["reference_wav_path"])
        self.assertIn("Match the speaking pace", model.generate_kwargs["text"])

    def test_unknown_voice_preset_is_rejected(self) -> None:
        runtime = VoxCPM2TTSRuntime(
            model_name="voxcpm2",
            model_settings=ModelSettings(backend="voxcpm2"),
        )
        runtime._model = FakeVoxCPM()

        with self.assertRaises(ValueError):
            runtime.synthesize(
                ResponseRequest(
                    model="voxcpm2",
                    input="Hallo wereld",
                    language="Dutch",
                    voice=VoiceSpec(preset="missing"),
                )
            )

    def test_voxcpm_import_is_lazy(self) -> None:
        runtime = VoxCPM2TTSRuntime(
            model_name="voxcpm2",
            model_settings=ModelSettings(backend="voxcpm2"),
        )

        with mock.patch("builtins.__import__", side_effect=ImportError("missing")):
            with self.assertRaises(RuntimeError):
                runtime.load()


def _silent_wav(*, seconds: float, sample_rate_hz: int = 16000) -> bytes:
    buffer = io.BytesIO()
    frames = b"\x00\x00" * int(seconds * sample_rate_hz)
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(frames)
    return buffer.getvalue()
