from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import io
import threading
import time
from typing import Callable
import wave

from app.config import FairnessSettings
from app.schemas import EngineResult
from app.schemas import ResponseRequest

from .common import ModelStateError
from .common import RequestAdmissionError
from .streaming import RuntimeStreamResult
from .streaming import SynthesisCancelled
from .streaming import SynthesisHandle


@dataclass(frozen=True)
class FairnessKeySnapshot:
    fairness_key: str | None
    pending: int
    active: int
    weight: float
    score: float
    rejected_per_key_limit: int
    rejected_executor_limit: int


@dataclass(frozen=True)
class ExecutorSnapshot:
    queue_depth: int
    runtime_inflight: int
    configured_target_inflight: int
    effective_target_inflight: int
    accepting_new_requests: bool
    fairness_keys: tuple[FairnessKeySnapshot, ...] = ()
    fairness_rejected_per_key_limit: int = 0
    fairness_rejected_executor_limit: int = 0


@dataclass
class SchedulerJob:
    request: ResponseRequest
    stream_handle: SynthesisHandle
    enqueued_at: float
    job_id: int = 0
    fairness_key: str | None = None


@dataclass
class _FairnessState:
    pending_jobs: deque[SchedulerJob]
    weight: float
    virtual_service_ms: float
    active_started_at: dict[int, float]
    activation_order: int
    last_selected_order: int
    idle_since: float | None = None
    rejected_per_key_limit: int = 0
    rejected_executor_limit: int = 0


class _FairPendingQueue:
    def __init__(self, *, model_name: str, settings: FairnessSettings) -> None:
        self._model_name = model_name
        self._settings = settings
        self._states: dict[str | None, _FairnessState] = {}
        self._pending_count = 0
        self._next_job_id = 1
        self._next_activation_order = 0
        self._next_selection_order = 1
        self._rejected_per_key_limit = 0
        self._rejected_executor_limit = 0

    @property
    def pending_count(self) -> int:
        return self._pending_count

    @property
    def rejected_per_key_limit(self) -> int:
        return self._rejected_per_key_limit

    @property
    def rejected_executor_limit(self) -> int:
        return self._rejected_executor_limit

    def enqueue(
        self,
        *,
        request: ResponseRequest,
        stream_handle: SynthesisHandle,
        now: float,
    ) -> SchedulerJob:
        self._expire_idle_states(now)
        fairness_key = request.fairness_key
        state = self._states.get(fairness_key)
        if state is not None and len(state.pending_jobs) >= self._settings.max_pending_per_key:
            state.rejected_per_key_limit += 1
            self._rejected_per_key_limit += 1
            raise RequestAdmissionError(
                status_code=429,
                code="fairness_key_queue_full",
                message=(
                    f"pending queue for fairness key {self._display_key(fairness_key)!r} "
                    f"is full for model {self._model_name!r}"
                ),
            )
        if self._pending_count >= self._settings.max_pending_per_executor:
            if state is not None:
                state.rejected_executor_limit += 1
            self._rejected_executor_limit += 1
            raise RequestAdmissionError(
                status_code=429,
                code="executor_queue_full",
                message=f"pending queue is full for model {self._model_name!r}",
            )
        if state is None:
            state = self._new_state(fairness_key, now)
            self._states[fairness_key] = state

        job = SchedulerJob(
            request=request,
            stream_handle=stream_handle,
            enqueued_at=now,
            job_id=self._next_job_id,
            fairness_key=fairness_key,
        )
        self._next_job_id += 1
        state.pending_jobs.append(job)
        state.idle_since = None
        self._pending_count += 1
        return job

    def remove(self, job: SchedulerJob, *, now: float) -> bool:
        state = self._states.get(job.fairness_key)
        if state is None:
            return False
        match = next(
            (candidate for candidate in state.pending_jobs if candidate.job_id == job.job_id),
            None,
        )
        if match is None:
            return False
        state.pending_jobs.remove(match)
        self._pending_count -= 1
        self._mark_idle_if_needed(state, now)
        return True

    def pop_next(self, *, now: float) -> SchedulerJob | None:
        self._expire_idle_states(now)
        candidates = [state for state in self._states.values() if state.pending_jobs]
        if not candidates:
            return None
        selected_state = min(
            candidates,
            key=lambda state: (
                self._score(state, now),
                len(state.active_started_at) >= self._settings.soft_max_inflight_per_key,
                state.last_selected_order,
                state.activation_order,
            ),
        )
        job = selected_state.pending_jobs.popleft()
        selected_state.active_started_at[job.job_id] = now
        selected_state.last_selected_order = self._next_selection_order
        selected_state.idle_since = None
        self._next_selection_order += 1
        self._pending_count -= 1
        return job

    def complete(self, job: SchedulerJob, *, service_ms: float, now: float) -> None:
        state = self._states.get(job.fairness_key)
        if state is None or state.active_started_at.pop(job.job_id, None) is None:
            return
        state.virtual_service_ms += max(0.0, service_ms) / state.weight
        self._mark_idle_if_needed(state, now)

    def abandon(self, job: SchedulerJob, *, now: float) -> None:
        state = self._states.get(job.fairness_key)
        if state is None:
            return
        state.active_started_at.pop(job.job_id, None)
        self._mark_idle_if_needed(state, now)

    def drain(self, *, now: float) -> list[SchedulerJob]:
        drained: list[SchedulerJob] = []
        for state in self._states.values():
            drained.extend(state.pending_jobs)
            state.pending_jobs.clear()
            self._mark_idle_if_needed(state, now)
        self._pending_count = 0
        return drained

    def score(self, fairness_key: str | None, *, now: float) -> float | None:
        self._expire_idle_states(now)
        state = self._states.get(fairness_key)
        if state is None:
            return None
        return self._score(state, now)

    def snapshots(self, *, now: float) -> tuple[FairnessKeySnapshot, ...]:
        self._expire_idle_states(now)
        return tuple(
            FairnessKeySnapshot(
                fairness_key=fairness_key,
                pending=len(state.pending_jobs),
                active=len(state.active_started_at),
                weight=state.weight,
                score=self._score(state, now),
                rejected_per_key_limit=state.rejected_per_key_limit,
                rejected_executor_limit=state.rejected_executor_limit,
            )
            for fairness_key, state in sorted(
                self._states.items(),
                key=lambda item: (item[0] is not None, item[0] or ""),
            )
            if state.pending_jobs or state.active_started_at
        )

    def _new_state(self, fairness_key: str | None, now: float) -> _FairnessState:
        active_states = [
            state
            for state in self._states.values()
            if state.pending_jobs or state.active_started_at
        ]
        baseline = min((self._score(state, now) for state in active_states), default=0.0)
        state = _FairnessState(
            pending_jobs=deque(),
            weight=self._weight(fairness_key),
            virtual_service_ms=baseline,
            active_started_at={},
            activation_order=self._next_activation_order,
            last_selected_order=0,
        )
        self._next_activation_order += 1
        return state

    @staticmethod
    def _score(state: _FairnessState, now: float) -> float:
        active_elapsed_ms = sum(
            max(0.0, now - started_at) * 1000.0
            for started_at in state.active_started_at.values()
        )
        return state.virtual_service_ms + (active_elapsed_ms / state.weight)

    def _weight(self, fairness_key: str | None) -> float:
        if fairness_key is None:
            return self._settings.default_weight
        return self._settings.weights.get(fairness_key, self._settings.default_weight)

    @staticmethod
    def _mark_idle_if_needed(state: _FairnessState, now: float) -> None:
        if not state.pending_jobs and not state.active_started_at:
            state.idle_since = now

    def _expire_idle_states(self, now: float) -> None:
        expired_keys = [
            fairness_key
            for fairness_key, state in self._states.items()
            if state.idle_since is not None
            and now - state.idle_since >= self._settings.idle_state_ttl_s
        ]
        for fairness_key in expired_keys:
            del self._states[fairness_key]

    @staticmethod
    def _display_key(fairness_key: str | None) -> str:
        return fairness_key if fairness_key is not None else "anonymous"


class LoadedModelExecutor:
    def __init__(
        self,
        *,
        model_name: str,
        complete_fn: Callable[[ResponseRequest], EngineResult],
        stream_fn: Callable[[ResponseRequest, SynthesisHandle], RuntimeStreamResult] | None = None,
        configured_target_inflight: int,
        fairness_settings: FairnessSettings,
        runtime_capability: int = 1,
    ) -> None:
        self.model_name = str(model_name)
        self._complete_fn = complete_fn
        self._stream_fn = stream_fn
        self._configured_target_inflight = max(1, int(configured_target_inflight))
        self._effective_target_inflight = min(self._configured_target_inflight, max(1, int(runtime_capability)))
        self._pending_queue = _FairPendingQueue(
            model_name=self.model_name,
            settings=fairness_settings,
        )
        self._accepting_new_requests = True
        self._stop_requested = False
        self._runtime_inflight = 0
        self._active_stream_handles: dict[int, SynthesisHandle] = {}
        self._cond = threading.Condition()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"tts-pool-executor-{self.model_name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def enqueue_stream(self, request: ResponseRequest, handle: SynthesisHandle) -> None:
        with self._cond:
            if not self._accepting_new_requests or self._stop_requested:
                raise ModelStateError(self.model_name, "model_unloading")
            job = self._pending_queue.enqueue(
                request=request,
                stream_handle=handle,
                now=time.perf_counter(),
            )
            handle.set_cancel_callback(lambda: self._cancel_stream_job(job))
            self._cond.notify_all()

    def snapshot(self) -> ExecutorSnapshot:
        with self._cond:
            now = time.perf_counter()
            return ExecutorSnapshot(
                queue_depth=self._pending_queue.pending_count,
                runtime_inflight=self._runtime_inflight,
                configured_target_inflight=self._configured_target_inflight,
                effective_target_inflight=self._effective_target_inflight,
                accepting_new_requests=self._accepting_new_requests,
                fairness_keys=self._pending_queue.snapshots(now=now),
                fairness_rejected_per_key_limit=self._pending_queue.rejected_per_key_limit,
                fairness_rejected_executor_limit=self._pending_queue.rejected_executor_limit,
            )

    def begin_shutdown(self) -> None:
        cancelled_jobs: list[SchedulerJob] = []
        active_handles: list[SynthesisHandle] = []
        with self._cond:
            self._accepting_new_requests = False
            self._stop_requested = True
            cancelled_jobs.extend(self._pending_queue.drain(now=time.perf_counter()))
            active_handles.extend(self._active_stream_handles.values())
            self._cond.notify_all()
        for job in cancelled_jobs:
            self._fail_job(job, ModelStateError(self.model_name, "model_unloading"))
        for handle in active_handles:
            handle.cancel()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        while True:
            with self._cond:
                while True:
                    if (
                        self._stop_requested
                        and self._pending_queue.pending_count == 0
                        and self._runtime_inflight == 0
                    ):
                        return
                    if (
                        self._pending_queue.pending_count > 0
                        and self._runtime_inflight < self._effective_target_inflight
                    ):
                        dequeued_at = time.perf_counter()
                        job = self._pending_queue.pop_next(now=dequeued_at)
                        if job is None:
                            continue
                        self._runtime_inflight += 1
                        self._active_stream_handles[job.job_id] = job.stream_handle
                        break
                    self._cond.wait()
            try:
                worker = threading.Thread(
                    target=self._run_job,
                    kwargs={"job": job, "dequeued_at": dequeued_at},
                    name=f"tts-pool-runtime-{self.model_name}",
                    daemon=True,
                )
                worker.start()
            except Exception as exc:
                with self._cond:
                    if self._runtime_inflight > 0:
                        self._runtime_inflight -= 1
                    self._active_stream_handles.pop(job.job_id, None)
                    self._pending_queue.abandon(job, now=time.perf_counter())
                    self._cond.notify_all()
                self._fail_job(job, exc)

    def _run_job(self, *, job: SchedulerJob, dequeued_at: float) -> None:
        stream_result: RuntimeStreamResult | None = None
        error: Exception | None = None
        backend_started_at = time.perf_counter()
        backend_finished_at: float | None = None
        try:
            job.stream_handle.mark_dispatched(
                queue_wait_ms=max(0.0, (dequeued_at - job.enqueued_at) * 1000.0)
            )
            if job.stream_handle.cancelled:
                raise SynthesisCancelled("synthesis was cancelled before runtime dispatch")
            stream_fn = self._stream_fn
            if stream_fn is not None:
                stream_result = stream_fn(job.request, job.stream_handle)
            else:
                result = self._complete_fn(job.request)
                stream_result = _emit_complete_result(result, job.stream_handle)
            backend_finished_at = time.perf_counter()
        except Exception as exc:
            error = exc
        finally:
            if backend_finished_at is None:
                backend_finished_at = time.perf_counter()
            service_ms = max(0.0, (backend_finished_at - backend_started_at) * 1000.0)
            with self._cond:
                if self._runtime_inflight > 0:
                    self._runtime_inflight -= 1
                self._active_stream_handles.pop(job.job_id, None)
                self._pending_queue.complete(
                    job,
                    service_ms=service_ms,
                    now=backend_finished_at,
                )
                self._cond.notify_all()
        if error is not None:
            self._fail_job(job, error)
        elif stream_result is not None:
            total_ms = max(0.0, (backend_finished_at - job.stream_handle.created_at) * 1000.0)
            backend_ms = max(0.0, (backend_finished_at - backend_started_at) * 1000.0)
            metrics = dict(stream_result.metrics)
            metrics["engine_queue_wait_ms"] = job.stream_handle.scheduler_queue_wait_ms
            metrics["backend_synthesis_wall_ms"] = backend_ms
            metrics["engine_total_wall_ms"] = total_ms
            metrics["engine_outside_backend_wall_ms"] = max(0.0, total_ms - backend_ms)
            job.stream_handle.complete(stream_result, metrics=metrics)

    def _cancel_stream_job(self, job: SchedulerJob) -> None:
        removed = False
        with self._cond:
            removed = self._pending_queue.remove(job, now=time.perf_counter())
            if removed:
                self._cond.notify_all()
        if removed:
            job.stream_handle.fail(SynthesisCancelled("synthesis was cancelled while queued"))

    def _fail_job(self, job: SchedulerJob, error: Exception) -> None:
        job.stream_handle.fail(error)


def _emit_complete_result(result: EngineResult, handle: SynthesisHandle) -> RuntimeStreamResult:
    if result.mime_type not in {"audio/wav", "audio/x-wav"}:
        raise ValueError("complete-only runtime must return WAV audio")
    try:
        source = wave.open(io.BytesIO(result.audio), "rb")
    except (EOFError, wave.Error) as exc:
        raise ValueError("complete-only runtime returned invalid WAV audio") from exc
    with source:
        channel_count = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate_hz = source.getframerate()
        if channel_count != 1 or sample_width != 2 or source.getcomptype() != "NONE":
            raise ValueError("complete-only runtime must return mono PCM S16LE WAV audio")
        frame_bytes = channel_count * sample_width
        chunk_bytes = min(handle.max_audio_chunk_bytes, 64 * 1024)
        chunk_bytes -= chunk_bytes % frame_bytes
        if chunk_bytes <= 0:
            raise ValueError("stream buffer is smaller than one PCM frame")
        frames_per_chunk = chunk_bytes // frame_bytes
        handle.emit_started(sample_rate_hz=sample_rate_hz, channel_count=channel_count)
        sequence_number = 0
        first_sample = 0
        while True:
            pcm = source.readframes(frames_per_chunk)
            if not pcm:
                break
            handle.emit_audio(
                sequence_number=sequence_number,
                first_sample=first_sample,
                pcm=pcm,
            )
            sequence_number += 1
            first_sample += len(pcm) // frame_bytes

    duration_ms = result.duration_ms
    if duration_ms is None:
        duration_ms = round(first_sample * 1000 / sample_rate_hz)
    metadata = dict(result.metadata)
    metadata["native_streaming"] = False
    return RuntimeStreamResult(
        total_sample_count=first_sample,
        duration_ms=duration_ms,
        chunk_count=sequence_number,
        metrics=dict(result.metrics),
        metadata=metadata,
    )
