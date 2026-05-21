from __future__ import annotations

import base64
import io
import unittest
import wave
from unittest import mock

import numpy as np

from app.config import ModelSettings
from app.config import VoxCPM2WarmupCase
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
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        self.generate_calls.append(kwargs)
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
                voice=VoiceSpec(
                    instructions=(
                        "Speak in Dutch. Pronounce numbers, abbreviations, and short fragments in Dutch. "
                        "Match the speaking pace, rhythm, and articulation of the reference audio."
                    ),
                    reference_audio=reference,
                ),
            )
        )

        self.assertEqual(result.metadata["reference_source_duration_ms"], 2000)
        self.assertTrue(result.metadata["reference_clipped"])
        self.assertLessEqual(result.metadata["reference_duration_ms"], 1000)
        self.assertIsNotNone(model.generate_kwargs["reference_wav_path"])
        self.assertIn("Speak in Dutch", model.generate_kwargs["text"])
        self.assertIn("Pronounce numbers", model.generate_kwargs["text"])
        self.assertIn("Match the speaking pace", model.generate_kwargs["text"])

    def test_prompt_text_engages_ultimate_cloning(self) -> None:
        model = FakeVoxCPM()
        runtime = VoxCPM2TTSRuntime(
            model_name="voxcpm2",
            model_settings=ModelSettings(backend="voxcpm2"),
        )
        runtime._model = model
        reference = ReferenceAudio(
            mime_type="audio/wav",
            data_base64=base64.b64encode(_silent_wav(seconds=1.0)).decode("ascii"),
            max_duration_s=1.0,
            prompt_text="Ik lees dit korte bericht met een rustige stem.",
        )

        result = runtime.synthesize(
            ResponseRequest(
                model="voxcpm2",
                input="Hallo wereld",
                language="Dutch",
                voice=VoiceSpec(
                    instructions="Speak in Dutch.",
                    reference_audio=reference,
                ),
            )
        )

        kwargs = model.generate_kwargs
        self.assertIsNotNone(kwargs["reference_wav_path"])
        self.assertEqual(kwargs["prompt_wav_path"], kwargs["reference_wav_path"])
        self.assertEqual(
            kwargs["prompt_text"],
            "Ik lees dit korte bericht met een rustige stem.",
        )
        self.assertTrue(result.metadata["ultimate_cloning"])

    def test_prompt_only_mode_drops_reference_wav_path(self) -> None:
        model = FakeVoxCPM()
        runtime = VoxCPM2TTSRuntime(
            model_name="voxcpm2",
            model_settings=ModelSettings(backend="voxcpm2"),
        )
        runtime._model = model
        reference = ReferenceAudio(
            mime_type="audio/wav",
            data_base64=base64.b64encode(_silent_wav(seconds=1.0)).decode("ascii"),
            max_duration_s=1.0,
            prompt_text="Ik lees dit korte bericht met een rustige stem.",
            also_use_as_reference=False,
        )

        result = runtime.synthesize(
            ResponseRequest(
                model="voxcpm2",
                input="Hallo wereld",
                language="Dutch",
                voice=VoiceSpec(
                    instructions="Speak in Dutch.",
                    reference_audio=reference,
                ),
            )
        )

        kwargs = model.generate_kwargs
        self.assertNotIn("reference_wav_path", kwargs)
        self.assertIsNotNone(kwargs["prompt_wav_path"])
        self.assertEqual(
            kwargs["prompt_text"],
            "Ik lees dit korte bericht met een rustige stem.",
        )
        self.assertTrue(result.metadata["ultimate_cloning"])
        self.assertFalse(result.metadata["ultimate_cloning_with_reference"])

    def test_missing_prompt_text_keeps_reference_only_mode(self) -> None:
        model = FakeVoxCPM()
        runtime = VoxCPM2TTSRuntime(
            model_name="voxcpm2",
            model_settings=ModelSettings(backend="voxcpm2"),
        )
        runtime._model = model
        reference = ReferenceAudio(
            mime_type="audio/wav",
            data_base64=base64.b64encode(_silent_wav(seconds=1.0)).decode("ascii"),
            max_duration_s=1.0,
        )

        result = runtime.synthesize(
            ResponseRequest(
                model="voxcpm2",
                input="Hallo wereld",
                language="Dutch",
                voice=VoiceSpec(
                    instructions="Speak in Dutch.",
                    reference_audio=reference,
                ),
            )
        )

        kwargs = model.generate_kwargs
        self.assertIsNotNone(kwargs["reference_wav_path"])
        self.assertNotIn("prompt_wav_path", kwargs)
        self.assertNotIn("prompt_text", kwargs)
        self.assertFalse(result.metadata["ultimate_cloning"])

    def test_reference_audio_does_not_add_implicit_pace_prompt(self) -> None:
        model = FakeVoxCPM()
        runtime = VoxCPM2TTSRuntime(
            model_name="voxcpm2",
            model_settings=ModelSettings(backend="voxcpm2"),
        )
        runtime._model = model
        reference = ReferenceAudio(
            mime_type="audio/wav",
            data_base64=base64.b64encode(_silent_wav(seconds=1.0)).decode("ascii"),
            max_duration_s=1.0,
        )

        result = runtime.synthesize(
            ResponseRequest(
                model="voxcpm2",
                input="Hallo wereld",
                language="Dutch",
                voice=VoiceSpec(
                    instructions="Speak in Dutch.",
                    reference_audio=reference,
                ),
            )
        )

        self.assertIsNotNone(model.generate_kwargs["reference_wav_path"])
        self.assertIn("Speak in Dutch", model.generate_kwargs["text"])
        self.assertNotIn("Match the speaking pace", model.generate_kwargs["text"])

    def test_voice_preset_is_rejected(self) -> None:
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
                    voice=VoiceSpec(preset="configured"),
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

    def test_load_runs_default_warmup_cases_when_enabled(self) -> None:
        model = FakeVoxCPM()
        runtime = VoxCPM2TTSRuntime(
            model_name="voxcpm2",
            model_settings=ModelSettings(
                backend="voxcpm2",
                voxcpm2_warmup_enabled=True,
            ),
        )
        runtime._model = model

        runtime.load()

        self.assertEqual(len(model.generate_calls), 5)
        self.assertEqual({4, 6, 10}, {call["inference_timesteps"] for call in model.generate_calls})
        self.assertTrue(any(call["reference_wav_path"] is None for call in model.generate_calls))
        self.assertTrue(any(call["reference_wav_path"] is not None for call in model.generate_calls))

    def test_load_runs_configured_warmup_cases(self) -> None:
        model = FakeVoxCPM()
        runtime = VoxCPM2TTSRuntime(
            model_name="voxcpm2",
            model_settings=ModelSettings(
                backend="voxcpm2",
                voxcpm2_warmup_enabled=True,
                voxcpm2_warmup_cases=(
                    VoxCPM2WarmupCase(
                        name="custom",
                        text="Custom warmup",
                        reference_audio=True,
                        cfg_value=1.5,
                        inference_timesteps=6,
                    ),
                ),
            ),
        )
        runtime._model = model

        runtime.load()

        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(model.generate_kwargs["cfg_value"], 1.5)
        self.assertEqual(model.generate_kwargs["inference_timesteps"], 6)
        self.assertIsNotNone(model.generate_kwargs["reference_wav_path"])
        self.assertIn("Custom warmup", model.generate_kwargs["text"])


def _silent_wav(*, seconds: float, sample_rate_hz: int = 16000) -> bytes:
    buffer = io.BytesIO()
    frames = b"\x00\x00" * int(seconds * sample_rate_hz)
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(frames)
    return buffer.getvalue()
