from __future__ import annotations

import asyncio
import threading
import unittest

import grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from app.config import GrpcSettings
from app.engine.common import UnknownModelError
from app.engine.streaming import RuntimeStreamResult
from app.engine.streaming import SynthesisHandle
from app.grpc_api.adapter import request_from_proto
from app.grpc_api.service import COUNTERS
from app.grpc_api.server import GrpcServer
from app.grpc_api.server import SERVICE_NAME
from app.grpc_api.v1 import tts_pb2
from app.grpc_api.v1 import tts_pb2_grpc
from app.schemas import MAX_REFERENCE_AUDIO_BYTES


class RecordingEngine:
    def __init__(self) -> None:
        self.request = None

    def stream(self, request):
        self.request = request
        handle = SynthesisHandle(
            response_id="ttsresp_test",
            model_name=request.model,
            max_buffer_chunks=4,
            max_buffer_bytes=1024,
            stalled_consumer_timeout_s=1.0,
        )
        handle.mark_dispatched(queue_wait_ms=3.5)
        handle.emit_started(sample_rate_hz=16_000)
        handle.emit_audio(sequence_number=0, first_sample=0, pcm=b"\x01\x00\x02\x00")
        handle.complete(
            RuntimeStreamResult(
                total_sample_count=2,
                duration_ms=1,
                chunk_count=1,
                metrics={"generate_ms": 5.0},
                metadata={"engine": "fake"},
            ),
            metrics={"generate_ms": 5.0},
        )
        return handle


class RejectingEngine:
    def stream(self, request):
        raise UnknownModelError(request.model)


class ExplodingEngine:
    def stream(self, request):
        handle = SynthesisHandle(
            response_id="ttsresp_exploding",
            model_name=request.model,
            max_buffer_chunks=1,
            max_buffer_bytes=1024,
            stalled_consumer_timeout_s=1.0,
        )
        handle.fail(RuntimeError("private runtime traceback and filesystem path"))
        return handle


class DelayedEngine:
    def stream(self, request):
        handle = SynthesisHandle(
            response_id=f"ttsresp_{request.input}",
            model_name=request.model,
            max_buffer_chunks=2,
            max_buffer_bytes=1024,
            stalled_consumer_timeout_s=1.0,
        )

        def produce() -> None:
            handle.emit_started(sample_rate_hz=16_000)
            handle.emit_audio(sequence_number=0, first_sample=0, pcm=b"\x00\x00")
            handle.complete(RuntimeStreamResult(1, 1, 1, {}, {}), metrics={})

        threading.Timer(0.02 if request.input == "slow" else 0.001, produce).start()
        return handle


class GrpcAdapterTests(unittest.TestCase):
    def test_binary_reference_audio_and_optional_values_are_mapped(self) -> None:
        request = request_from_proto(
            tts_pb2.SynthesisRequest(
                model="nanovllm_voxcpm",
                input="Hallo",
                language="Dutch",
                fairness_key=" principal ",
                output_encoding=tts_pb2.AUDIO_ENCODING_PCM_S16LE,
                voice=tts_pb2.VoiceSpec(
                    instructions="Speak clearly.",
                    reference_audio=tts_pb2.ReferenceAudio(
                        mime_type="audio/wav",
                        data=b"RIFF-test",
                        max_duration_s=3.0,
                        prompt_text="Reference text",
                    ),
                ),
                generation=tts_pb2.GenerationParams(
                    nanovllm_voxcpm=tts_pb2.NanoVLLMVoxCPMGenerationParams(
                        temperature=0.8,
                    )
                ),
            )
        )

        self.assertEqual(request.fairness_key, "principal")
        self.assertEqual(request.voice.reference_audio.decoded_bytes(), b"RIFF-test")
        self.assertEqual(request.voice.reference_audio.data, b"RIFF-test")
        self.assertTrue(request.voice.reference_audio.also_use_as_reference)
        self.assertEqual(request.voice.reference_audio.max_duration_s, 3.0)
        self.assertEqual(request.generation.nanovllm_voxcpm.temperature, 0.8)
        self.assertIsNone(request.generation.nanovllm_voxcpm.cfg_value)

    def test_required_text_fields_are_validated_centrally(self) -> None:
        for field_name in ("model", "input", "language"):
            values = {"model": "model", "input": "text", "language": "English"}
            values[field_name] = "  "
            with self.subTest(field_name=field_name), self.assertRaises(Exception):
                request_from_proto(tts_pb2.SynthesisRequest(**values))

    def test_binary_reference_audio_limit_is_enforced(self) -> None:
        with self.assertRaises(Exception):
            request_from_proto(
                tts_pb2.SynthesisRequest(
                    model="model",
                    input="text",
                    language="English",
                    voice=tts_pb2.VoiceSpec(
                        reference_audio=tts_pb2.ReferenceAudio(
                            data=b"x" * (MAX_REFERENCE_AUDIO_BYTES + 1),
                        )
                    ),
                )
            )

    def test_generated_descriptor_uses_canonical_proto_path(self) -> None:
        self.assertEqual(tts_pb2.DESCRIPTOR.name, "tts/v1/tts.proto")


class GrpcServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = RecordingEngine()
        self.server = GrpcServer(
            settings=GrpcSettings(
                enabled=True,
                host="127.0.0.1",
                port=0,
                shutdown_grace_s=0.1,
            ),
            engine=self.engine,
        )
        await self.server.start()
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{self.server.bound_port}")
        await self.channel.channel_ready()

    async def asyncTearDown(self) -> None:
        await self.channel.close()
        await self.server.stop()

    async def test_health_and_synthesis_stream(self) -> None:
        counters_before = COUNTERS.snapshot()
        health_stub = health_pb2_grpc.HealthStub(self.channel)
        health_response = await health_stub.Check(health_pb2.HealthCheckRequest(service=SERVICE_NAME))
        self.assertEqual(health_response.status, health_pb2.HealthCheckResponse.SERVING)

        stub = tts_pb2_grpc.TTSServiceStub(self.channel)
        events = [
            event
            async for event in stub.Synthesize(
                tts_pb2.SynthesisRequest(
                    model="nanovllm_voxcpm",
                    input="Hallo",
                    language="Dutch",
                    output_encoding=tts_pb2.AUDIO_ENCODING_PCM_S16LE,
                )
            )
        ]

        self.assertEqual([event.WhichOneof("payload") for event in events], ["started", "audio_chunk", "completed"])
        self.assertEqual(events[0].started.response_id, "ttsresp_test")
        self.assertEqual(events[0].started.scheduler_queue_wait_ms, 3.5)
        self.assertEqual(events[1].audio_chunk.pcm, b"\x01\x00\x02\x00")
        self.assertEqual(events[2].completed.total_sample_count, 2)
        self.assertEqual(events[2].completed.metadata["engine"], "fake")
        self.assertGreaterEqual(events[2].completed.metrics["grpc_first_chunk_wall_ms"], 0.0)
        self.assertGreaterEqual(events[2].completed.metrics["grpc_stream_wall_ms"], 0.0)
        self.assertEqual(events[2].completed.metrics["stream_max_buffered_chunks"], 1.0)
        counters_after = COUNTERS.snapshot()
        self.assertEqual(
            counters_after["requests_total"],
            counters_before.get("requests_total", 0) + 1,
        )
        self.assertEqual(
            counters_after["status_ok"],
            counters_before.get("status_ok", 0) + 1,
        )
        self.assertEqual(
            counters_after["complete_runtime_completed"],
            counters_before.get("complete_runtime_completed", 0) + 1,
        )

    async def test_error_code_is_exposed_as_trailing_metadata(self) -> None:
        await self.channel.close()
        await self.server.stop()
        self.server = GrpcServer(
            settings=GrpcSettings(enabled=True, host="127.0.0.1", port=0),
            engine=RejectingEngine(),
        )
        await self.server.start()
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{self.server.bound_port}")
        stub = tts_pb2_grpc.TTSServiceStub(self.channel)

        with self.assertRaises(grpc.aio.AioRpcError) as raised:
            async for _ in stub.Synthesize(
                tts_pb2.SynthesisRequest(
                    model="missing",
                    input="Hallo",
                    language="Dutch",
                )
            ):
                pass

        self.assertEqual(raised.exception.code(), grpc.StatusCode.NOT_FOUND)
        trailing = dict(raised.exception.trailing_metadata())
        self.assertEqual(trailing["tts-error-code"], "unknown_model")

    async def test_internal_error_detail_is_sanitized(self) -> None:
        await self.channel.close()
        await self.server.stop()
        self.server = GrpcServer(
            settings=GrpcSettings(enabled=True, host="127.0.0.1", port=0),
            engine=ExplodingEngine(),
        )
        await self.server.start()
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{self.server.bound_port}")
        stub = tts_pb2_grpc.TTSServiceStub(self.channel)

        with self.assertRaises(grpc.aio.AioRpcError) as raised:
            async for _ in stub.Synthesize(
                tts_pb2.SynthesisRequest(model="model", input="text", language="English")
            ):
                pass

        self.assertEqual(raised.exception.code(), grpc.StatusCode.INTERNAL)
        self.assertEqual(raised.exception.details(), "TTS synthesis failed")
        self.assertNotIn("filesystem", raised.exception.details())

    async def test_waiting_rpc_does_not_delay_another_rpc_on_same_channel(self) -> None:
        await self.channel.close()
        await self.server.stop()
        self.server = GrpcServer(
            settings=GrpcSettings(enabled=True, host="127.0.0.1", port=0),
            engine=DelayedEngine(),
        )
        await self.server.start()
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{self.server.bound_port}")
        stub = tts_pb2_grpc.TTSServiceStub(self.channel)

        async def collect(text: str) -> list[str]:
            return [
                event.WhichOneof("payload")
                async for event in stub.Synthesize(
                    tts_pb2.SynthesisRequest(
                        model="model",
                        input=text,
                        language="English",
                    )
                )
            ]

        slow = asyncio.create_task(collect("slow"))
        await asyncio.sleep(0)
        fast = asyncio.create_task(collect("fast"))

        self.assertEqual(
            await asyncio.wait_for(fast, timeout=0.2),
            ["started", "audio_chunk", "completed"],
        )
        self.assertFalse(slow.done())
        self.assertEqual(
            await asyncio.wait_for(slow, timeout=0.2),
            ["started", "audio_chunk", "completed"],
        )


if __name__ == "__main__":
    unittest.main()
