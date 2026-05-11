from __future__ import annotations

import base64
import io
import unittest
import wave

import numpy as np

from app.config import ModelSettings
from app.config import NanoVLLMWarmupCase
from app.engine.nanovllm_voxcpm import NanoVLLMVoxCPMTTSRuntime
from app.schemas import ReferenceAudio
from app.schemas import ResponseRequest
from app.schemas import VoiceSpec


class FakeNanoServer:
    def __init__(self) -> None:
        self.encoded_wav: bytes | None = None
        self.encoded_wavs: list[bytes] = []
        self.generate_kwargs = None
        self.generate_calls: list[dict] = []
        self.stopped = False

    async def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        self.encoded_wav = wav
        self.encoded_wavs.append(wav)
        self.encoded_format = wav_format
        return f"latents-{len(self.encoded_wavs)}".encode("ascii")

    async def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        self.generate_calls.append(kwargs)
        yield np.zeros(1600, dtype=np.float32)

    async def stop(self) -> None:
        self.stopped = True


class NanoVLLMVoxCPMTests(unittest.TestCase):
    def test_synthesize_encodes_reference_and_generates_wav(self) -> None:
        server = FakeNanoServer()
        runtime = NanoVLLMVoxCPMTTSRuntime(
            model_name="nanovllm_voxcpm",
            model_settings=ModelSettings(
                backend="nanovllm_voxcpm",
                nanovllm_reference_max_duration_s=8.0,
                nanovllm_max_num_seqs=4,
            ),
        )
        runtime._start_loop()
        runtime._server = server
        runtime._model_info = {"sample_rate": 16000}
        self.addCleanup(runtime.close)

        result = runtime.synthesize(
            ResponseRequest(
                model="nanovllm_voxcpm",
                input="Hallo wereld",
                language="Dutch",
                voice=VoiceSpec(
                    preset="configured",
                    reference_audio=ReferenceAudio(
                        mime_type="audio/wav",
                        data_base64=base64.b64encode(_silent_wav(seconds=2.0)).decode("ascii"),
                        max_duration_s=1.0,
                    ),
                    reference_audio_match="voice_and_pace",
                ),
            )
        )

        self.assertEqual(runtime.runtime_capability, 4)
        self.assertEqual(result.sample_rate_hz, 16000)
        self.assertEqual(result.duration_ms, 100)
        self.assertEqual(result.metadata["engine"], "nanovllm_voxcpm")
        self.assertTrue(result.metadata["reference_clipped"])
        self.assertEqual(result.metadata["reference_audio_match"], "voice_and_pace")
        self.assertLessEqual(_wav_duration_ms(server.encoded_wav or b""), 1000)
        self.assertEqual(server.generate_kwargs["ref_audio_latents"], b"latents-1")
        self.assertIn("Match the speaking pace", server.generate_kwargs["target_text"])

    def test_warmup_runs_configured_cases(self) -> None:
        server = FakeNanoServer()
        runtime = NanoVLLMVoxCPMTTSRuntime(
            model_name="nanovllm_voxcpm",
            model_settings=ModelSettings(
                backend="nanovllm_voxcpm",
                nanovllm_warmup_enabled=True,
                nanovllm_warmup_cases=(
                    NanoVLLMWarmupCase(
                        name="no_ref",
                        text="Warm no reference.",
                        reference_audio=False,
                        temperature=0.01,
                        max_generate_length=64,
                    ),
                    NanoVLLMWarmupCase(
                        name="ref_pace",
                        text="Warm reference.",
                        reference_audio=True,
                        reference_audio_match="voice_and_pace",
                        reference_duration_s=0.5,
                        temperature=0.02,
                        max_generate_length=32,
                    ),
                ),
            ),
        )
        runtime._start_loop()
        runtime._server = server
        self.addCleanup(runtime.close)

        runtime._run_on_loop(runtime._warmup_async())

        self.assertEqual(len(server.generate_calls), 2)
        self.assertEqual(len(server.encoded_wavs), 1)
        self.assertIsNone(server.generate_calls[0]["ref_audio_latents"])
        self.assertEqual(server.generate_calls[0]["temperature"], 0.01)
        self.assertEqual(server.generate_calls[0]["max_generate_length"], 64)
        self.assertEqual(server.generate_calls[1]["ref_audio_latents"], b"latents-1")
        self.assertEqual(server.generate_calls[1]["temperature"], 0.02)
        self.assertEqual(server.generate_calls[1]["max_generate_length"], 32)
        self.assertIn("Match the speaking pace", server.generate_calls[1]["target_text"])
        self.assertLessEqual(_wav_duration_ms(server.encoded_wavs[0]), 500)


def _silent_wav(*, seconds: float, sample_rate_hz: int = 16000) -> bytes:
    buffer = io.BytesIO()
    frames = b"\x00\x00" * int(seconds * sample_rate_hz)
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(frames)
    return buffer.getvalue()


def _wav_duration_ms(data: bytes) -> int:
    with wave.open(io.BytesIO(data), "rb") as reader:
        return int((reader.getnframes() / reader.getframerate()) * 1000)


if __name__ == "__main__":
    unittest.main()
