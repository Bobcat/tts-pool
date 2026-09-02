from __future__ import annotations

import asyncio
import threading
import time
import unittest

from app.engine.streaming import StreamAudioChunk
from app.engine.streaming import StreamConsumerStalled
from app.engine.streaming import RuntimeStreamResult
from app.engine.streaming import SynthesisCancelled
from app.engine.streaming import SynthesisHandle


def _handle(*, chunks: int = 1, byte_limit: int = 8, timeout: float = 0.05) -> SynthesisHandle:
    return SynthesisHandle(
        response_id="ttsresp_test",
        model_name="test-model",
        max_buffer_chunks=chunks,
        max_buffer_bytes=byte_limit,
        stalled_consumer_timeout_s=timeout,
    )


class SynthesisHandleTests(unittest.TestCase):
    def test_buffer_is_bounded_by_chunks(self) -> None:
        handle = _handle()
        handle.emit_audio(sequence_number=0, first_sample=0, pcm=b"\x00\x00")

        with self.assertRaises(StreamConsumerStalled):
            handle.emit_audio(sequence_number=1, first_sample=1, pcm=b"\x00\x00")

    def test_consumer_releases_buffer_capacity(self) -> None:
        handle = _handle(timeout=0.5)
        handle.emit_audio(sequence_number=0, first_sample=0, pcm=b"\x00\x00")
        emitted = threading.Event()

        def emit_second() -> None:
            handle.emit_audio(sequence_number=1, first_sample=1, pcm=b"\x01\x00")
            emitted.set()

        thread = threading.Thread(target=emit_second)
        thread.start()
        time.sleep(0.02)
        self.assertFalse(emitted.is_set())
        first = handle.read_event()
        thread.join(timeout=1.0)

        self.assertIsInstance(first, StreamAudioChunk)
        self.assertTrue(emitted.is_set())
        self.assertEqual(handle.read_event().sequence_number, 1)

    def test_cancellation_wakes_blocked_producer(self) -> None:
        handle = _handle(timeout=1.0)
        handle.emit_audio(sequence_number=0, first_sample=0, pcm=b"\x00\x00")
        errors: list[Exception] = []

        def emit_second() -> None:
            try:
                handle.emit_audio(sequence_number=1, first_sample=1, pcm=b"\x01\x00")
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=emit_second)
        thread.start()
        time.sleep(0.02)
        handle.cancel()
        thread.join(timeout=1.0)

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], SynthesisCancelled)

    def test_cancelled_handle_cannot_complete_successfully(self) -> None:
        handle = _handle()
        handle.cancel()

        handle.complete(
            RuntimeStreamResult(0, 0, 0, {}, {}),
            metrics={},
        )

        with self.assertRaises(SynthesisCancelled):
            handle.read_event()

    def test_cancel_callback_runs_when_registered_after_cancellation(self) -> None:
        handle = _handle()
        called = threading.Event()
        handle.cancel()

        handle.set_cancel_callback(called.set)

        self.assertTrue(called.is_set())


class AsyncSynthesisHandleTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_reader_does_not_block_the_event_loop(self) -> None:
        handle = _handle()
        reader = asyncio.create_task(handle.read_event_async())
        await asyncio.sleep(0)
        heartbeat_ran = False

        async def heartbeat() -> None:
            nonlocal heartbeat_ran
            await asyncio.sleep(0)
            heartbeat_ran = True

        await heartbeat()
        handle.emit_audio(sequence_number=0, first_sample=0, pcm=b"\x00\x00")

        self.assertTrue(heartbeat_ran)
        self.assertIsInstance(await asyncio.wait_for(reader, timeout=0.1), StreamAudioChunk)

    async def test_cancellation_wakes_async_reader_directly(self) -> None:
        handle = _handle()
        reader = asyncio.create_task(handle.read_event_async())
        await asyncio.sleep(0)

        handle.cancel()

        with self.assertRaises(SynthesisCancelled):
            await asyncio.wait_for(reader, timeout=0.1)


if __name__ == "__main__":
    unittest.main()
