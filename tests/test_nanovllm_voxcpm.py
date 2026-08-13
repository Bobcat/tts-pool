from __future__ import annotations

import io
import threading
import unittest
import wave

import numpy as np

from app.config import ModelSettings
from app.config import NanoVLLMWarmupCase
from app.engine.nanovllm_voxcpm import NanoVLLMVoxCPMTTSRuntime
from app.engine.streaming import StreamAudioChunk
from app.engine.streaming import StreamCompleted
from app.engine.streaming import StreamStarted
from app.engine.streaming import SynthesisCancelled
from app.engine.streaming import SynthesisHandle
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


class ClosingNanoServer(FakeNanoServer):
    def __init__(self) -> None:
        super().__init__()
        self.second_chunk_yielded = threading.Event()
        self.generate_closed = threading.Event()

    async def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        self.generate_calls.append(kwargs)
        try:
            yield np.zeros(1600, dtype=np.float32)
            self.second_chunk_yielded.set()
            yield np.zeros(1600, dtype=np.float32)
        finally:
            self.generate_closed.set()


class NanoVLLMVoxCPMTests(unittest.TestCase):
    def test_stream_emits_pcm_before_completed_event(self) -> None:
        server = FakeNanoServer()
        runtime = NanoVLLMVoxCPMTTSRuntime(
            model_name="nanovllm_voxcpm",
            model_settings=ModelSettings(backend="nanovllm_voxcpm"),
        )
        runtime._start_loop()
        runtime._server = server
        runtime._model_info = {"sample_rate": 16_000}
        self.addCleanup(runtime.close)
        handle = SynthesisHandle(
            response_id="ttsresp_test",
            model_name="nanovllm_voxcpm",
            max_buffer_chunks=4,
            max_buffer_bytes=16_384,
            stalled_consumer_timeout_s=1.0,
        )

        request = ResponseRequest(
            model="nanovllm_voxcpm",
            input="Hallo wereld",
            language="Dutch",
        )
        complete = runtime.synthesize(request)
        result = runtime.synthesize_stream(request, handle)
        handle.complete(result, metrics=result.metrics)
        events = [handle.read_event(), handle.read_event(), handle.read_event()]

        self.assertIsInstance(events[0], StreamStarted)
        self.assertIsInstance(events[1], StreamAudioChunk)
        self.assertIsInstance(events[2], StreamCompleted)
        self.assertEqual(events[1].sequence_number, 0)
        self.assertEqual(events[1].first_sample, 0)
        self.assertEqual(len(events[1].pcm), 3_200)
        self.assertEqual(result.total_sample_count, 1_600)
        self.assertEqual(result.duration_ms, 100)
        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(_wav_frames(complete.audio), events[1].pcm)

    def test_stream_cancellation_closes_runtime_generator(self) -> None:
        server = ClosingNanoServer()
        runtime = NanoVLLMVoxCPMTTSRuntime(
            model_name="nanovllm_voxcpm",
            model_settings=ModelSettings(backend="nanovllm_voxcpm"),
        )
        runtime._start_loop()
        runtime._server = server
        runtime._model_info = {"sample_rate": 16_000}
        self.addCleanup(runtime.close)
        handle = SynthesisHandle(
            response_id="ttsresp_cancel",
            model_name="nanovllm_voxcpm",
            max_buffer_chunks=1,
            max_buffer_bytes=3_200,
            stalled_consumer_timeout_s=1.0,
        )
        request = ResponseRequest(
            model="nanovllm_voxcpm",
            input="Hallo wereld",
            language="Dutch",
        )
        errors: list[Exception] = []

        def synthesize() -> None:
            try:
                runtime.synthesize_stream(request, handle)
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=synthesize)
        worker.start()
        self.assertTrue(server.second_chunk_yielded.wait(timeout=1.0))
        handle.cancel()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], SynthesisCancelled)
        self.assertTrue(server.generate_closed.wait(timeout=1.0))

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
                    instructions=(
                        "Speak in Dutch. Pronounce numbers, abbreviations, and short fragments in Dutch. "
                        "Match the speaking pace, rhythm, and articulation of the reference audio."
                    ),
                    reference_audio=ReferenceAudio(
                        mime_type="audio/wav",
                        data=_silent_wav(seconds=2.0),
                        max_duration_s=1.0,
                    ),
                ),
            )
        )

        self.assertEqual(runtime.runtime_capability, 4)
        self.assertEqual(result.sample_rate_hz, 16000)
        self.assertEqual(result.duration_ms, 100)
        self.assertEqual(result.metadata["engine"], "nanovllm_voxcpm")
        self.assertTrue(result.metadata["reference_clipped"])
        self.assertLessEqual(_wav_duration_ms(server.encoded_wav or b""), 1000)
        self.assertEqual(server.generate_kwargs["ref_audio_latents"], b"latents-1")
        self.assertIn("Speak in Dutch", server.generate_kwargs["target_text"])
        self.assertIn("Pronounce numbers", server.generate_kwargs["target_text"])
        self.assertIn("Match the speaking pace", server.generate_kwargs["target_text"])

    def test_prompt_text_engages_ultimate_cloning(self) -> None:
        server = FakeNanoServer()
        runtime = NanoVLLMVoxCPMTTSRuntime(
            model_name="nanovllm_voxcpm",
            model_settings=ModelSettings(
                backend="nanovllm_voxcpm",
                nanovllm_reference_max_duration_s=8.0,
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
                    instructions="Speak in Dutch.",
                    reference_audio=ReferenceAudio(
                        mime_type="audio/wav",
                        data=_silent_wav(seconds=1.0),
                        max_duration_s=1.0,
                        prompt_text="Ik lees dit korte bericht met een rustige stem.",
                    ),
                ),
            )
        )

        kwargs = server.generate_kwargs
        self.assertEqual(kwargs["ref_audio_latents"], b"latents-1")
        self.assertEqual(kwargs["prompt_latents"], b"latents-1")
        self.assertEqual(
            kwargs["prompt_text"],
            "Ik lees dit korte bericht met een rustige stem.",
        )
        self.assertTrue(result.metadata["ultimate_cloning"])

    def test_prompt_text_reference_audio_is_not_clipped(self) -> None:
        server = FakeNanoServer()
        runtime = NanoVLLMVoxCPMTTSRuntime(
            model_name="nanovllm_voxcpm",
            model_settings=ModelSettings(
                backend="nanovllm_voxcpm",
                nanovllm_reference_max_duration_s=8.0,
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
                    instructions="Speak in Dutch.",
                    reference_audio=ReferenceAudio(
                        mime_type="audio/wav",
                        data=_silent_wav(seconds=2.0),
                        max_duration_s=1.0,
                        prompt_text="Ik lees dit korte bericht met een rustige stem.",
                    ),
                ),
            )
        )

        self.assertFalse(result.metadata["reference_clipped"])
        self.assertTrue(result.metadata["reference_clip_skipped_for_prompt_text"])
        self.assertEqual(result.metadata["reference_duration_ms"], 2000)
        self.assertEqual(_wav_duration_ms(server.encoded_wav or b""), 2000)

    def test_prompt_only_mode_drops_ref_audio_latents(self) -> None:
        server = FakeNanoServer()
        runtime = NanoVLLMVoxCPMTTSRuntime(
            model_name="nanovllm_voxcpm",
            model_settings=ModelSettings(
                backend="nanovllm_voxcpm",
                nanovllm_reference_max_duration_s=8.0,
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
                    instructions="Speak in Dutch.",
                    reference_audio=ReferenceAudio(
                        mime_type="audio/wav",
                        data=_silent_wav(seconds=1.0),
                        max_duration_s=1.0,
                        prompt_text="Ik lees dit korte bericht met een rustige stem.",
                        also_use_as_reference=False,
                    ),
                ),
            )
        )

        kwargs = server.generate_kwargs
        self.assertNotIn("ref_audio_latents", kwargs)
        self.assertEqual(kwargs["prompt_latents"], b"latents-1")
        self.assertEqual(
            kwargs["prompt_text"],
            "Ik lees dit korte bericht met een rustige stem.",
        )
        self.assertTrue(result.metadata["ultimate_cloning"])
        self.assertFalse(result.metadata["ultimate_cloning_with_reference"])

    def test_missing_prompt_text_keeps_reference_only_mode(self) -> None:
        server = FakeNanoServer()
        runtime = NanoVLLMVoxCPMTTSRuntime(
            model_name="nanovllm_voxcpm",
            model_settings=ModelSettings(
                backend="nanovllm_voxcpm",
                nanovllm_reference_max_duration_s=8.0,
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
                    instructions="Speak in Dutch.",
                    reference_audio=ReferenceAudio(
                        mime_type="audio/wav",
                        data=_silent_wav(seconds=1.0),
                        max_duration_s=1.0,
                    ),
                ),
            )
        )

        kwargs = server.generate_kwargs
        self.assertEqual(kwargs["ref_audio_latents"], b"latents-1")
        self.assertNotIn("prompt_latents", kwargs)
        self.assertNotIn("prompt_text", kwargs)
        self.assertFalse(result.metadata["ultimate_cloning"])

    def test_voice_preset_is_rejected(self) -> None:
        runtime = NanoVLLMVoxCPMTTSRuntime(
            model_name="nanovllm_voxcpm",
            model_settings=ModelSettings(backend="nanovllm_voxcpm"),
        )
        runtime._start_loop()
        runtime._server = FakeNanoServer()
        runtime._model_info = {"sample_rate": 16000}
        self.addCleanup(runtime.close)

        with self.assertRaises(ValueError):
            runtime.synthesize(
                ResponseRequest(
                    model="nanovllm_voxcpm",
                    input="Hallo wereld",
                    language="Dutch",
                    voice=VoiceSpec(preset="configured"),
                )
            )

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


def _wav_frames(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data), "rb") as reader:
        return reader.readframes(reader.getnframes())


if __name__ == "__main__":
    unittest.main()
