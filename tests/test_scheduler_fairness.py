from __future__ import annotations

import heapq
import io
import threading
import time
import unittest
from unittest import mock
import wave

from app.config import AppSettings
from app.config import EngineSettings
from app.config import FairnessSettings
from app.config import ModelSettings
from app.engine.common import ModelStateError
from app.engine.common import RequestAdmissionError
from app.engine.scheduler import _FairPendingQueue
from app.engine.scheduler import LoadedModelExecutor
from app.engine.router import TTSRouterEngine
from app.engine.streaming import RuntimeStreamResult
from app.engine.streaming import StreamAudioChunk
from app.engine.streaming import StreamCompleted
from app.engine.streaming import StreamStarted
from app.engine.streaming import SynthesisCancelled
from app.engine.streaming import SynthesisHandle
from app.schemas import EngineResult
from app.schemas import MAX_REFERENCE_AUDIO_BYTES
from app.schemas import ReferenceAudio
from app.schemas import ResponseRequest


def _request(*, key: str | None, text: str) -> ResponseRequest:
    return ResponseRequest(
        model="test-model",
        input=text,
        language="English",
        fairness_key=key,
    )


def _result() -> EngineResult:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x00\x00")
    return EngineResult(audio=output.getvalue(), sample_rate_hz=16000, duration_ms=1)


def _handle() -> SynthesisHandle:
    return SynthesisHandle(
        response_id="ttsresp_test",
        model_name="test-model",
        max_buffer_chunks=4,
        max_buffer_bytes=1024,
        stalled_consumer_timeout_s=1.0,
    )


class RequestFairnessContractTests(unittest.TestCase):
    def test_fairness_key_is_optional_and_normalized(self) -> None:
        anonymous = _request(key=None, text="anonymous")
        keyed = _request(key="  opaque-principal  ", text="keyed")
        maximum = _request(key=f"  {'x' * 128}  ", text="maximum")

        self.assertIsNone(anonymous.fairness_key)
        self.assertEqual(keyed.fairness_key, "opaque-principal")
        self.assertEqual(maximum.fairness_key, "x" * 128)

    def test_fairness_key_rejects_invalid_values(self) -> None:
        for value in (" \t ", "x" * 129, 123):
            with self.subTest(value=value), self.assertRaises(Exception):
                ResponseRequest(
                    model="test-model",
                    input="invalid",
                    language="English",
                    fairness_key=value,
                )

    def test_reference_audio_byte_length_is_bounded(self) -> None:
        with self.assertRaises(Exception):
            ReferenceAudio(data=b"x" * (MAX_REFERENCE_AUDIO_BYTES + 1))


class FairPendingQueueTests(unittest.TestCase):
    def _queue(self, **overrides: object) -> _FairPendingQueue:
        return _FairPendingQueue(
            model_name="test-model",
            settings=FairnessSettings(**overrides),
        )

    @staticmethod
    def _enqueue(
        queue: _FairPendingQueue,
        *,
        key: str | None,
        text: str,
        now: float = 0.0,
    ) -> None:
        queue.enqueue(
            request=_request(key=key, text=text),
            stream_handle=_handle(),
            now=now,
        )

    def test_preserves_fifo_order_within_one_key(self) -> None:
        queue = self._queue()
        self._enqueue(queue, key="a", text="first")
        self._enqueue(queue, key="a", text="second")

        first = queue.pop_next(now=0.0)
        self.assertIsNotNone(first)
        queue.complete(first, service_ms=1.0, now=0.001)
        second = queue.pop_next(now=0.001)

        self.assertEqual(first.request.input, "first")
        self.assertEqual(second.request.input, "second")

    def test_equal_scores_rotate_deterministically_between_keys(self) -> None:
        queue = self._queue()
        for text in ("a1", "a2"):
            self._enqueue(queue, key="a", text=text)
        for text in ("b1", "b2"):
            self._enqueue(queue, key="b", text=text)

        selected: list[str | None] = []
        for _ in range(4):
            job = queue.pop_next(now=0.0)
            selected.append(job.fairness_key)
            queue.complete(job, service_ms=0.0, now=0.0)

        self.assertEqual(selected, ["a", "b", "a", "b"])

    def test_weighted_slot_time_holds_at_capacities_one_two_and_four(self) -> None:
        for capacity in (1, 2, 4):
            with self.subTest(capacity=capacity):
                service = self._simulate_weighted_service(capacity=capacity)
                normalized_difference = abs((service["a"] / 2.0) - service["b"])
                self.assertLessEqual(normalized_difference, 80.0 * capacity)

    def test_active_elapsed_service_affects_selection(self) -> None:
        queue = self._queue(soft_max_inflight_per_key=2)
        self._enqueue(queue, key="a", text="a1")
        self._enqueue(queue, key="a", text="a2")
        self._enqueue(queue, key="b", text="b1")

        active = queue.pop_next(now=0.0)
        next_job = queue.pop_next(now=0.1)

        self.assertEqual(active.fairness_key, "a")
        self.assertEqual(next_job.fairness_key, "b")

    def test_soft_cap_breaks_equal_score_tie(self) -> None:
        queue = self._queue(soft_max_inflight_per_key=1)
        self._enqueue(queue, key="a", text="a1")
        self._enqueue(queue, key="a", text="a2")
        self._enqueue(queue, key="b", text="b1")

        first = queue.pop_next(now=0.0)
        second = queue.pop_next(now=0.0)

        self.assertEqual(first.fairness_key, "a")
        self.assertEqual(second.fairness_key, "b")

    def test_one_key_borrows_every_slot_while_alone(self) -> None:
        queue = self._queue()
        for index in range(4):
            self._enqueue(queue, key="a", text=f"a{index}")

        selected = [queue.pop_next(now=0.0).fairness_key for _ in range(4)]

        self.assertEqual(selected, ["a", "a", "a", "a"])

    def test_every_queued_key_at_soft_cap_still_fills_free_slot(self) -> None:
        queue = self._queue()
        for key in ("a", "b"):
            self._enqueue(queue, key=key, text=f"{key}1")
            self._enqueue(queue, key=key, text=f"{key}2")
        first = queue.pop_next(now=0.0)
        second = queue.pop_next(now=0.0)

        borrowed = queue.pop_next(now=0.0)

        self.assertEqual({first.fairness_key, second.fairness_key}, {"a", "b"})
        self.assertIn(borrowed.fairness_key, {"a", "b"})

    def test_new_waiting_key_gets_next_released_slot(self) -> None:
        queue = self._queue(max_pending_per_key=8, max_pending_per_executor=12)
        for index in range(5):
            self._enqueue(queue, key="a", text=f"a{index}")
        active = [queue.pop_next(now=0.0) for _ in range(4)]
        self._enqueue(queue, key="b", text="b1", now=0.1)

        queue.complete(active[0], service_ms=200.0, now=0.2)
        next_job = queue.pop_next(now=0.2)

        self.assertEqual(next_job.fairness_key, "b")

    def test_new_key_starts_at_current_minimum_score(self) -> None:
        queue = self._queue()
        self._enqueue(queue, key="a", text="a1")
        self._enqueue(queue, key="a", text="a2")
        first = queue.pop_next(now=0.0)
        queue.complete(first, service_ms=100.0, now=0.1)

        self._enqueue(queue, key="b", text="b1", now=0.1)

        self.assertEqual(queue.score("a", now=0.1), 100.0)
        self.assertEqual(queue.score("b", now=0.1), 100.0)

    def test_idle_state_expires(self) -> None:
        queue = self._queue(idle_state_ttl_s=10.0)
        self._enqueue(queue, key="a", text="a1")
        first = queue.pop_next(now=0.0)
        queue.complete(first, service_ms=100.0, now=1.0)

        self.assertEqual(queue.score("a", now=9.0), 100.0)
        self.assertIsNone(queue.score("a", now=11.0))

    def test_queue_limits_reject_before_enqueue(self) -> None:
        queue = self._queue(max_pending_per_key=1, max_pending_per_executor=2)
        self._enqueue(queue, key="a", text="a1")

        with self.assertRaises(RequestAdmissionError) as per_key_error:
            self._enqueue(queue, key="a", text="a2")
        self.assertEqual(per_key_error.exception.code, "fairness_key_queue_full")

        self._enqueue(queue, key="b", text="b1")
        with self.assertRaises(RequestAdmissionError) as executor_error:
            self._enqueue(queue, key="c", text="c1")
        self.assertEqual(executor_error.exception.code, "executor_queue_full")
        self.assertEqual(queue.pending_count, 2)

    def test_anonymous_bucket_and_snapshot_are_bounded(self) -> None:
        queue = self._queue()
        self._enqueue(queue, key=None, text="anonymous")
        self._enqueue(queue, key="a", text="keyed")
        active = queue.pop_next(now=0.0)

        snapshots = queue.snapshots(now=0.0)

        self.assertEqual(len(snapshots), 2)
        self.assertIsNone(snapshots[0].fairness_key)
        self.assertEqual(sum(item.pending for item in snapshots), 1)
        self.assertEqual(sum(item.active for item in snapshots), 1)
        queue.complete(active, service_ms=0.0, now=0.0)

    def test_drain_removes_pending_work_from_every_key(self) -> None:
        queue = self._queue()
        self._enqueue(queue, key="a", text="a")
        self._enqueue(queue, key="b", text="b")

        drained = queue.drain(now=1.0)

        self.assertEqual({job.fairness_key for job in drained}, {"a", "b"})
        self.assertEqual(queue.pending_count, 0)
        self.assertEqual(queue.snapshots(now=1.0), ())

    def _simulate_weighted_service(self, *, capacity: int) -> dict[str, float]:
        queue = self._queue(
            weights={"a": 2.0, "b": 1.0},
            max_pending_per_key=20,
            max_pending_per_executor=40,
        )
        for key in ("a", "b"):
            for index in range(capacity + 2):
                self._enqueue(queue, key=key, text=f"{key}{index}")

        active: list[tuple[float, int, object, float]] = []
        sequence = 0

        def start_next(now: float) -> None:
            nonlocal sequence
            job = queue.pop_next(now=now)
            duration_ms = 80.0 if job.fairness_key == "a" else 30.0
            sequence += 1
            heapq.heappush(
                active,
                (now + duration_ms / 1000.0, sequence, job, duration_ms),
            )

        for _ in range(capacity):
            start_next(0.0)

        service = {"a": 0.0, "b": 0.0}
        for index in range(600):
            completed_at, _, job, duration_ms = heapq.heappop(active)
            queue.complete(job, service_ms=duration_ms, now=completed_at)
            service[job.fairness_key] += duration_ms
            self._enqueue(
                queue,
                key=job.fairness_key,
                text=f"{job.fairness_key}-next-{index}",
                now=completed_at,
            )
            start_next(completed_at)
        return service


class LoadedModelExecutorFairnessTests(unittest.TestCase):
    def test_stream_job_uses_executor_and_reports_scheduler_metrics(self) -> None:
        def stream(_: ResponseRequest, handle: SynthesisHandle) -> RuntimeStreamResult:
            handle.emit_started(sample_rate_hz=16_000)
            handle.emit_audio(sequence_number=0, first_sample=0, pcm=b"\x00\x00")
            return RuntimeStreamResult(
                total_sample_count=1,
                duration_ms=1,
                chunk_count=1,
                metrics={"runtime_ms": 1.0},
                metadata={"engine": "test"},
            )

        executor = LoadedModelExecutor(
            model_name="test-model",
            complete_fn=lambda _: _result(),
            stream_fn=stream,
            configured_target_inflight=1,
            runtime_capability=1,
            fairness_settings=FairnessSettings(),
        )
        executor.start()
        handle = _handle()
        try:
            executor.enqueue_stream(_request(key="a", text="stream"), handle)
            events = [handle.read_event(), handle.read_event(), handle.read_event()]

            self.assertIsInstance(events[0], StreamStarted)
            self.assertIsInstance(events[1], StreamAudioChunk)
            self.assertIsInstance(events[2], StreamCompleted)
            self.assertIn("engine_queue_wait_ms", events[2].metrics)
            self.assertIn("backend_synthesis_wall_ms", events[2].metrics)
            self.assertIn("engine_total_wall_ms", events[2].metrics)
        finally:
            executor.begin_shutdown()
            executor.join(timeout=1.0)

    def test_queued_stream_cancellation_removes_job(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def complete(_: ResponseRequest) -> EngineResult:
            entered.set()
            release.wait(timeout=1.0)
            return _result()

        executor = LoadedModelExecutor(
            model_name="test-model",
            complete_fn=complete,
            configured_target_inflight=1,
            runtime_capability=1,
            fairness_settings=FairnessSettings(),
        )
        executor.start()
        active = _handle()
        executor.enqueue_stream(_request(key="a", text="active"), active)
        self.assertTrue(entered.wait(timeout=1.0))
        handle = _handle()
        try:
            executor.enqueue_stream(_request(key="b", text="pending"), handle)
            self.assertEqual(executor.snapshot().queue_depth, 1)

            handle.cancel()

            with self.assertRaises(SynthesisCancelled):
                handle.read_event()
            self.assertEqual(executor.snapshot().queue_depth, 0)
        finally:
            release.set()
            self.assertIsInstance(active.read_event(), StreamStarted)
            self.assertIsInstance(active.read_event(), StreamAudioChunk)
            self.assertIsInstance(active.read_event(), StreamCompleted)
            executor.begin_shutdown()
            executor.join(timeout=1.0)

    def test_active_stream_cancellation_releases_runtime_slot(self) -> None:
        entered = threading.Event()

        def stream(_: ResponseRequest, handle: SynthesisHandle) -> RuntimeStreamResult:
            handle.emit_started(sample_rate_hz=16_000)
            entered.set()
            while not handle.cancelled:
                time.sleep(0.005)
            raise SynthesisCancelled("cancelled in runtime")

        executor = LoadedModelExecutor(
            model_name="test-model",
            complete_fn=lambda _: _result(),
            stream_fn=stream,
            configured_target_inflight=1,
            runtime_capability=1,
            fairness_settings=FairnessSettings(),
        )
        executor.start()
        handle = _handle()
        try:
            executor.enqueue_stream(_request(key="a", text="active"), handle)
            self.assertTrue(entered.wait(timeout=1.0))
            self.assertIsInstance(handle.read_event(), StreamStarted)

            handle.cancel()

            with self.assertRaises(SynthesisCancelled):
                handle.read_event()
            deadline = time.monotonic() + 1.0
            while executor.snapshot().runtime_inflight and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(executor.snapshot().runtime_inflight, 0)
        finally:
            executor.begin_shutdown()
            executor.join(timeout=1.0)

    def test_backend_failure_is_charged_and_releases_slot(self) -> None:
        def fail(_: ResponseRequest) -> EngineResult:
            time.sleep(0.01)
            raise RuntimeError("backend failed")

        executor = LoadedModelExecutor(
            model_name="test-model",
            complete_fn=fail,
            configured_target_inflight=1,
            runtime_capability=1,
            fairness_settings=FairnessSettings(),
        )
        executor.start()
        try:
            handle = _handle()
            executor.enqueue_stream(_request(key="a", text="fail"), handle)
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                handle.read_event()

            self.assertEqual(executor.snapshot().runtime_inflight, 0)
            score = executor._pending_queue.score("a", now=time.perf_counter())
            self.assertIsNotNone(score)
            self.assertGreaterEqual(score, 5.0)
        finally:
            executor.begin_shutdown()
            executor.join(timeout=1.0)

    def test_worker_start_failure_abandons_without_charge(self) -> None:
        executor = LoadedModelExecutor(
            model_name="test-model",
            complete_fn=lambda _: _result(),
            configured_target_inflight=1,
            runtime_capability=1,
            fairness_settings=FairnessSettings(),
        )
        executor.start()
        try:
            with mock.patch(
                "app.engine.scheduler.threading.Thread",
                side_effect=RuntimeError("thread start failed"),
            ):
                handle = _handle()
                executor.enqueue_stream(_request(key="a", text="fail"), handle)
                with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                    handle.read_event()

            self.assertEqual(executor.snapshot().runtime_inflight, 0)
            self.assertEqual(
                executor._pending_queue.score("a", now=time.perf_counter()),
                0.0,
            )
        finally:
            executor.begin_shutdown()
            executor.join(timeout=1.0)

    def test_unload_drains_all_pending_buckets(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def complete(_: ResponseRequest) -> EngineResult:
            entered.set()
            release.wait(timeout=1.0)
            return _result()

        executor = LoadedModelExecutor(
            model_name="test-model",
            complete_fn=complete,
            configured_target_inflight=1,
            runtime_capability=1,
            fairness_settings=FairnessSettings(),
        )
        executor.start()
        first = _handle()
        executor.enqueue_stream(_request(key="a", text="active"), first)
        self.assertTrue(entered.wait(timeout=1.0))
        pending_a = _handle()
        pending_b = _handle()
        executor.enqueue_stream(_request(key="a", text="pending-a"), pending_a)
        executor.enqueue_stream(_request(key="b", text="pending-b"), pending_b)

        executor.begin_shutdown()
        for handle in (pending_a, pending_b):
            with self.assertRaises(ModelStateError) as error:
                handle.read_event()
            self.assertEqual(error.exception.code, "model_unloading")
        with self.assertRaises(SynthesisCancelled):
            first.read_event()

        release.set()
        executor.join(timeout=1.0)
        self.assertEqual(executor.snapshot().queue_depth, 0)
        self.assertFalse(executor.snapshot().accepting_new_requests)

    def test_effective_capacity_is_bounded_by_runtime(self) -> None:
        for runtime_capability in (1, 2, 4):
            with self.subTest(runtime_capability=runtime_capability):
                executor = LoadedModelExecutor(
                    model_name="test-model",
                    complete_fn=lambda _: _result(),
                    configured_target_inflight=4,
                    runtime_capability=runtime_capability,
                    fairness_settings=FairnessSettings(),
                )
                self.assertEqual(
                    executor.snapshot().effective_target_inflight,
                    runtime_capability,
                )


class RouterFairnessTests(unittest.TestCase):
    def test_admin_snapshot_shows_active_and_pending_keys(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class BlockingRuntime:
            runtime_capability = 1

            def load(self) -> None:
                pass

            def synthesize(self, _: ResponseRequest) -> EngineResult:
                entered.set()
                release.wait(timeout=1.0)
                return _result()

            def close(self) -> None:
                pass

        settings = AppSettings(
            engine=EngineSettings(
                models={
                    "test-model": ModelSettings(
                        backend="stub",
                        enabled=True,
                        target_inflight=1,
                    )
                },
                fairness=FairnessSettings(),
            )
        )
        with mock.patch(
            "app.engine.router._build_runtime",
            return_value=BlockingRuntime(),
        ), mock.patch(
            "app.engine.router.query_primary_gpu_used_mib",
            return_value=None,
        ):
            engine = TTSRouterEngine(settings)

        active = engine.stream(_request(key="a", text="active"))
        self.assertTrue(entered.wait(timeout=1.0))
        pending = engine.stream(_request(key="b", text="pending"))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if engine.admin_models_payload()["models"][0]["queue_depth"] == 1:
                break
            time.sleep(0.005)

        payload = engine.admin_models_payload()["models"][0]

        self.assertEqual(payload["runtime_inflight"], 1)
        self.assertEqual(payload["queue_depth"], 1)
        self.assertEqual(
            {item["fairness_key"] for item in payload["fairness"]["keys"]},
            {"a", "b"},
        )
        self.assertEqual(
            sum(item["active"] for item in payload["fairness"]["keys"]),
            1,
        )
        self.assertEqual(
            sum(item["pending"] for item in payload["fairness"]["keys"]),
            1,
        )

        release.set()
        for handle in (active, pending):
            self.assertIsInstance(handle.read_event(), StreamStarted)
            self.assertIsInstance(handle.read_event(), StreamAudioChunk)
            self.assertIsInstance(handle.read_event(), StreamCompleted)
        engine.close()


if __name__ == "__main__":
    unittest.main()
