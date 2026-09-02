"""Versioned TTS streaming RPC implementation."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import logging
import threading
import time

import grpc
from google.protobuf.struct_pb2 import Struct
from pydantic import ValidationError

from app.engine import ModelStateError
from app.engine import RequestAdmissionError
from app.engine import UnknownModelError
from app.engine.streaming import StreamAudioChunk
from app.engine.streaming import StreamCompleted
from app.engine.streaming import StreamConsumerStalled
from app.engine.streaming import StreamStarted
from app.engine.streaming import SynthesisCancelled
from app.engine.streaming import SynthesisHandle

from .adapter import request_from_proto
from .v1 import tts_pb2
from .v1 import tts_pb2_grpc


LOGGER = logging.getLogger("uvicorn.error")
ERROR_CODE_METADATA_KEY = "tts-error-code"


class GrpcSynthesisCounters:
    def __init__(self) -> None:
        self._values: dict[str, int] = {}
        self._lock = threading.Lock()

    def increment(self, name: str) -> None:
        with self._lock:
            self._values[name] = self._values.get(name, 0) + 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._values)


COUNTERS = GrpcSynthesisCounters()


class TTSService(tts_pb2_grpc.TTSServiceServicer):
    def __init__(self, engine: object) -> None:
        self._engine = engine

    async def Synthesize(self, request: tts_pb2.SynthesisRequest, context: grpc.aio.ServicerContext):
        handle: SynthesisHandle | None = None
        received_at = time.perf_counter()
        first_chunk_written_at: float | None = None
        COUNTERS.increment("requests_total")
        try:
            engine_request = request_from_proto(request)
            handle = self._engine.stream(engine_request)
            context.add_done_callback(lambda _: handle.cancel())
            while True:
                event = await handle.read_event_async()
                if event is None:
                    return
                if isinstance(event, StreamCompleted):
                    metrics = dict(event.metrics)
                    metrics["grpc_stream_wall_ms"] = max(
                        0.0,
                        (time.perf_counter() - received_at) * 1000.0,
                    )
                    metrics["grpc_first_chunk_wall_ms"] = (
                        None
                        if first_chunk_written_at is None
                        else max(0.0, (first_chunk_written_at - received_at) * 1000.0)
                    )
                    event = replace(event, metrics=metrics)
                yield _event_to_proto(event)
                if isinstance(event, StreamAudioChunk) and first_chunk_written_at is None:
                    first_chunk_written_at = time.perf_counter()
                if isinstance(event, StreamCompleted):
                    COUNTERS.increment("status_ok")
                    COUNTERS.increment(
                        "native_streaming_completed"
                        if bool(event.metadata.get("native_streaming"))
                        else "complete_runtime_completed"
                    )
                    LOGGER.info(
                        "%s",
                        json.dumps(
                            {
                                "event": "tts_pool.grpc_stream",
                                "request_id": handle.response_id,
                                "model": handle.model_name,
                                "outcome": "completed",
                                "status": "OK",
                                "metrics": event.metrics,
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                    )
                    return
        except asyncio.CancelledError:
            if handle is not None:
                cancelled_at = time.perf_counter()
                COUNTERS.increment("status_cancelled")
                COUNTERS.increment(
                    "active_cancellations" if handle.dispatched else "queued_cancellations"
                )
                handle.cancel()
                _log_cancellation_when_runtime_releases(handle, cancelled_at=cancelled_at)
            raise
        except Exception as exc:
            if handle is not None:
                cancelled_at = time.perf_counter()
                handle.cancel()
            status, code = _status_for_error(exc)
            COUNTERS.increment(f"status_{status.name.lower()}")
            if isinstance(exc, SynthesisCancelled) and handle is not None:
                COUNTERS.increment(
                    "active_cancellations" if handle.dispatched else "queued_cancellations"
                )
                _log_cancellation_when_runtime_releases(handle, cancelled_at=cancelled_at)
            if isinstance(exc, StreamConsumerStalled):
                COUNTERS.increment("stalled_consumers")
            if not isinstance(exc, SynthesisCancelled):
                LOGGER.info(
                    "%s",
                    json.dumps(
                        {
                            "event": "tts_pool.grpc_stream",
                            "request_id": handle.response_id if handle is not None else None,
                            "model": handle.model_name if handle is not None else request.model,
                            "outcome": "failed",
                            "status": status.name,
                            "code": code,
                            "error": str(exc),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                )
            detail = _public_error_detail(exc, code)
            await context.abort(
                status,
                detail,
                trailing_metadata=((ERROR_CODE_METADATA_KEY, code),),
            )


def _event_to_proto(event: StreamStarted | StreamAudioChunk | StreamCompleted) -> tts_pb2.SynthesisEvent:
    if isinstance(event, StreamStarted):
        return tts_pb2.SynthesisEvent(
            started=tts_pb2.Started(
                response_id=event.response_id,
                model=event.model,
                sample_rate_hz=event.sample_rate_hz,
                channel_count=event.channel_count,
                encoding=tts_pb2.AUDIO_ENCODING_PCM_S16LE,
                scheduler_queue_wait_ms=event.scheduler_queue_wait_ms,
            )
        )
    if isinstance(event, StreamAudioChunk):
        return tts_pb2.SynthesisEvent(
            audio_chunk=tts_pb2.AudioChunk(
                sequence_number=event.sequence_number,
                first_sample=event.first_sample,
                pcm=event.pcm,
            )
        )
    metrics = Struct()
    metrics.update(event.metrics)
    metadata = Struct()
    metadata.update(event.metadata)
    return tts_pb2.SynthesisEvent(
        completed=tts_pb2.Completed(
            total_sample_count=event.total_sample_count,
            duration_ms=event.duration_ms,
            chunk_count=event.chunk_count,
            metrics=metrics,
            metadata=metadata,
        )
    )


def _status_for_error(exc: Exception) -> tuple[grpc.StatusCode, str]:
    if isinstance(exc, UnknownModelError):
        return grpc.StatusCode.NOT_FOUND, "unknown_model"
    if isinstance(exc, ModelStateError):
        return grpc.StatusCode.FAILED_PRECONDITION, exc.code
    if isinstance(exc, RequestAdmissionError):
        return grpc.StatusCode.RESOURCE_EXHAUSTED, exc.code
    if isinstance(exc, SynthesisCancelled):
        return grpc.StatusCode.CANCELLED, "synthesis_cancelled"
    if isinstance(exc, StreamConsumerStalled):
        return grpc.StatusCode.RESOURCE_EXHAUSTED, "stream_consumer_stalled"
    if isinstance(exc, (ValidationError, ValueError)):
        return grpc.StatusCode.INVALID_ARGUMENT, "invalid_request"
    return grpc.StatusCode.INTERNAL, "synthesis_failed"


def _public_error_detail(exc: Exception, code: str) -> str:
    if code == "synthesis_failed":
        return "TTS synthesis failed"
    return str(exc) or code


def _log_cancellation_when_runtime_releases(
    handle: SynthesisHandle,
    *,
    cancelled_at: float,
) -> None:
    def log_release(_: SynthesisHandle) -> None:
        LOGGER.info(
            "%s",
            json.dumps(
                {
                    "event": "tts_pool.grpc_stream",
                    "request_id": handle.response_id,
                    "model": handle.model_name,
                    "outcome": "cancelled",
                    "status": "CANCELLED",
                    "code": "synthesis_cancelled",
                    "cancellation_scope": "active" if handle.dispatched else "queued",
                    "metrics": {
                        "cancellation_to_runtime_release_ms": max(
                            0.0,
                            (time.perf_counter() - cancelled_at) * 1000.0,
                        )
                    },
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )

    handle.add_terminal_callback(log_release)
