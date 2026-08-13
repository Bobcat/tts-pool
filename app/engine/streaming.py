"""Bounded event delivery between model runtimes and streaming transports."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable
from typing import TypeAlias


class SynthesisCancelled(Exception):
    pass


class StreamConsumerStalled(Exception):
    pass


@dataclass(frozen=True)
class StreamStarted:
    response_id: str
    model: str
    sample_rate_hz: int
    channel_count: int
    encoding: str
    scheduler_queue_wait_ms: float


@dataclass(frozen=True)
class StreamAudioChunk:
    sequence_number: int
    first_sample: int
    pcm: bytes


@dataclass(frozen=True)
class StreamCompleted:
    total_sample_count: int
    duration_ms: int
    chunk_count: int
    metrics: dict[str, float | int | None]
    metadata: dict[str, object]


@dataclass(frozen=True)
class RuntimeStreamResult:
    total_sample_count: int
    duration_ms: int
    chunk_count: int
    metrics: dict[str, float | int | None]
    metadata: dict[str, object]


StreamEvent: TypeAlias = StreamStarted | StreamAudioChunk | StreamCompleted
TerminalCallback: TypeAlias = Callable[["SynthesisHandle"], None]


LOGGER = logging.getLogger(__name__)


class SynthesisHandle:
    def __init__(
        self,
        *,
        response_id: str,
        model_name: str,
        max_buffer_chunks: int,
        max_buffer_bytes: int,
        stalled_consumer_timeout_s: float,
    ) -> None:
        self.response_id = response_id
        self.model_name = model_name
        self.created_at = time.perf_counter()
        self.scheduler_queue_wait_ms = 0.0
        self._dispatched = False
        self._max_buffer_chunks = max(1, int(max_buffer_chunks))
        self._max_buffer_bytes = max(1, int(max_buffer_bytes))
        self._stalled_consumer_timeout_s = max(0.001, float(stalled_consumer_timeout_s))
        self._events: deque[StreamEvent] = deque()
        self._buffered_audio_chunks = 0
        self._buffered_audio_bytes = 0
        self._maximum_buffered_audio_chunks = 0
        self._maximum_buffered_audio_bytes = 0
        self._terminal = False
        self._error: Exception | None = None
        self._cancelled = threading.Event()
        self._cancel_callback: Callable[[], None] | None = None
        self._terminal_callbacks: list[TerminalCallback] = []
        self._async_waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = set()
        self._cond = threading.Condition()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def max_audio_chunk_bytes(self) -> int:
        return self._max_buffer_bytes

    @property
    def dispatched(self) -> bool:
        with self._cond:
            return self._dispatched

    def mark_dispatched(self, *, queue_wait_ms: float) -> None:
        with self._cond:
            self.scheduler_queue_wait_ms = max(0.0, float(queue_wait_ms))
            self._dispatched = True

    def set_cancel_callback(self, callback: Callable[[], None]) -> None:
        call_now = False
        with self._cond:
            if self._cancelled.is_set():
                call_now = True
            else:
                self._cancel_callback = callback
        if call_now:
            callback()

    def add_terminal_callback(self, callback: TerminalCallback) -> None:
        call_now = False
        with self._cond:
            if self._terminal:
                call_now = True
            else:
                self._terminal_callbacks.append(callback)
        if call_now:
            callback(self)

    def cancel(self) -> None:
        callback: Callable[[], None] | None
        with self._cond:
            if self._terminal or self._cancelled.is_set():
                return
            self._cancelled.set()
            callback = self._cancel_callback
            self._cond.notify_all()
            self._wake_async_waiters_locked()
        if callback is not None:
            callback()

    def emit_started(self, *, sample_rate_hz: int, channel_count: int = 1) -> None:
        self._append_control(
            StreamStarted(
                response_id=self.response_id,
                model=self.model_name,
                sample_rate_hz=int(sample_rate_hz),
                channel_count=int(channel_count),
                encoding="pcm_s16le",
                scheduler_queue_wait_ms=self.scheduler_queue_wait_ms,
            )
        )

    def emit_audio(self, *, sequence_number: int, first_sample: int, pcm: bytes) -> None:
        event = StreamAudioChunk(
            sequence_number=int(sequence_number),
            first_sample=int(first_sample),
            pcm=bytes(pcm),
        )
        event_bytes = len(event.pcm)
        if event_bytes > self._max_buffer_bytes:
            raise StreamConsumerStalled("one audio chunk exceeds the configured stream buffer")
        deadline = time.monotonic() + self._stalled_consumer_timeout_s
        with self._cond:
            while (
                self._buffered_audio_chunks >= self._max_buffer_chunks
                or self._buffered_audio_bytes + event_bytes > self._max_buffer_bytes
            ):
                self._raise_if_stopped_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StreamConsumerStalled("stream consumer did not drain its audio buffer")
                self._cond.wait(timeout=remaining)
            self._raise_if_stopped_locked()
            self._events.append(event)
            self._buffered_audio_chunks += 1
            self._buffered_audio_bytes += event_bytes
            self._maximum_buffered_audio_chunks = max(
                self._maximum_buffered_audio_chunks,
                self._buffered_audio_chunks,
            )
            self._maximum_buffered_audio_bytes = max(
                self._maximum_buffered_audio_bytes,
                self._buffered_audio_bytes,
            )
            self._cond.notify_all()
            self._wake_async_waiters_locked()

    def complete(self, result: RuntimeStreamResult, *, metrics: dict[str, float | int | None]) -> None:
        if self.cancelled:
            self.fail(SynthesisCancelled("synthesis was cancelled before completion"))
            return
        combined_metrics = dict(metrics)
        with self._cond:
            combined_metrics["stream_max_buffered_chunks"] = self._maximum_buffered_audio_chunks
            combined_metrics["stream_max_buffered_bytes"] = self._maximum_buffered_audio_bytes
        event = StreamCompleted(
            total_sample_count=result.total_sample_count,
            duration_ms=result.duration_ms,
            chunk_count=result.chunk_count,
            metrics=combined_metrics,
            metadata=dict(result.metadata),
        )
        callbacks = self._finish(event=event, error=None)
        self._run_terminal_callbacks(callbacks)

    def fail(self, error: Exception) -> None:
        callbacks = self._finish(event=None, error=error)
        self._run_terminal_callbacks(callbacks)

    def read_event(self) -> StreamEvent | None:
        with self._cond:
            while not self._events and not self._terminal and not self._cancelled.is_set():
                self._cond.wait()
            return self._pop_event_locked()

    async def read_event_async(self) -> StreamEvent | None:
        loop = asyncio.get_running_loop()
        while True:
            with self._cond:
                if self._events or self._terminal or self._cancelled.is_set():
                    return self._pop_event_locked()
                waiter: asyncio.Future[None] = loop.create_future()
                waiter_entry = (loop, waiter)
                self._async_waiters.add(waiter_entry)
            try:
                await waiter
            finally:
                with self._cond:
                    self._async_waiters.discard(waiter_entry)

    def _append_control(self, event: StreamEvent) -> None:
        with self._cond:
            self._raise_if_stopped_locked()
            self._events.append(event)
            self._cond.notify_all()
            self._wake_async_waiters_locked()

    def _finish(self, *, event: StreamEvent | None, error: Exception | None) -> list[TerminalCallback]:
        with self._cond:
            if self._terminal:
                return []
            if event is not None:
                self._events.append(event)
            self._error = error
            self._terminal = True
            callbacks = list(self._terminal_callbacks)
            self._terminal_callbacks.clear()
            self._cond.notify_all()
            self._wake_async_waiters_locked()
            return callbacks

    def _pop_event_locked(self) -> StreamEvent | None:
        if self._cancelled.is_set():
            raise SynthesisCancelled("synthesis was cancelled")
        if self._events:
            event = self._events.popleft()
            if isinstance(event, StreamAudioChunk):
                self._buffered_audio_chunks -= 1
                self._buffered_audio_bytes -= len(event.pcm)
                self._cond.notify_all()
            return event
        if self._error is not None:
            raise self._error
        return None

    def _wake_async_waiters_locked(self) -> None:
        waiters = tuple(self._async_waiters)
        for loop, waiter in waiters:
            loop.call_soon_threadsafe(self._resolve_waiter, waiter)

    @staticmethod
    def _resolve_waiter(waiter: asyncio.Future[None]) -> None:
        if not waiter.done():
            waiter.set_result(None)

    def _raise_if_stopped_locked(self) -> None:
        if self._cancelled.is_set():
            raise SynthesisCancelled("synthesis was cancelled")
        if self._terminal:
            if self._error is not None:
                raise self._error
            raise SynthesisCancelled("synthesis stream is closed")

    def _run_terminal_callbacks(self, callbacks: list[TerminalCallback]) -> None:
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                LOGGER.exception("synthesis terminal callback failed")
