from __future__ import annotations

import io
import math
import struct
import time
import wave

from app.config import ModelSettings
from app.schemas import EngineResult
from app.schemas import ResponseRequest


class StubTTSRuntime:
    runtime_capability = 4

    def __init__(self, *, model_name: str, model_settings: ModelSettings) -> None:
        self.model_name = model_name
        self.model_settings = model_settings

    def load(self) -> None:
        return

    def close(self) -> None:
        return

    def synthesize(self, request: ResponseRequest) -> EngineResult:
        started_at = time.perf_counter()
        text = str(request.input or "").strip()
        if not text:
            raise ValueError("input must not be empty")
        sample_rate_hz = 16_000
        duration_s = min(2.0, max(0.25, len(text) / 30.0))
        audio = _tone_wav(duration_s=duration_s, sample_rate_hz=sample_rate_hz)
        total_ms = (time.perf_counter() - started_at) * 1000.0
        return EngineResult(
            audio=audio,
            mime_type="audio/wav",
            sample_rate_hz=sample_rate_hz,
            duration_ms=int(duration_s * 1000),
            metrics={
                "stub_total_wall_ms": total_ms,
                "input_chars": len(text),
                "output_audio_seconds": duration_s,
                "realtime_factor": (total_ms / 1000.0) / duration_s if duration_s > 0 else 0.0,
            },
            metadata={
                "engine": "stub",
                "language": request.language,
                "voice_preset": request.voice.preset or "",
            },
        )


def _tone_wav(*, duration_s: float, sample_rate_hz: int) -> bytes:
    frame_count = int(duration_s * sample_rate_hz)
    amplitude = 0.16
    frequency = 440.0
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        frames = bytearray()
        for index in range(frame_count):
            value = int(amplitude * 32767 * math.sin(2.0 * math.pi * frequency * (index / sample_rate_hz)))
            frames.extend(struct.pack("<h", value))
        writer.writeframes(bytes(frames))
    return buffer.getvalue()
