from __future__ import annotations

import time

from app.config import ModelSettings
from app.schemas import EngineResult
from app.schemas import ResponseRequest


class KokoroTTSRuntime:
    runtime_capability = 1

    def __init__(self, *, model_name: str, model_settings: ModelSettings) -> None:
        self.model_name = model_name
        self.model_settings = model_settings
        self._synthesizer = None
        self._engine = None

    def load(self) -> None:
        synthesizer = self._synthesizer_instance()
        model_instance = getattr(synthesizer, "_model_instance", None)
        if callable(model_instance):
            model_instance()

    def close(self) -> None:
        self._engine = None
        self._synthesizer = None

    def synthesize(self, request: ResponseRequest) -> EngineResult:
        started_at = time.perf_counter()
        if request.voice.reference_audio is not None:
            raise ValueError("kokoro does not support reference_audio")
        engine = self._engine_instance()
        synthesizer = self._synthesizer_instance()
        speed = request.generation.kokoro.speed
        previous_speed = getattr(synthesizer, "speed", None)
        if speed is not None:
            synthesizer.speed = float(speed)
        try:
            from realtime_tts_engine import TTSRequest

            result = engine.synthesize(
                TTSRequest(
                    text=request.input,
                    language=request.language,
                    voice=self._voice_for_request(request),
                )
            )
        finally:
            if speed is not None and previous_speed is not None:
                synthesizer.speed = previous_speed
        total_ms = (time.perf_counter() - started_at) * 1000.0
        metadata = dict(result.metadata)
        metadata["voice_instructions_ignored"] = bool(str(request.voice.instructions or "").strip())
        metrics = dict(result.timings)
        metrics["kokoro_service_total_wall_ms"] = total_ms
        return EngineResult(
            audio=result.audio,
            mime_type=result.mime_type,
            sample_rate_hz=result.sample_rate_hz,
            duration_ms=result.duration_ms,
            metrics=metrics,
            metadata=metadata,
        )

    def _engine_instance(self):
        if self._engine is None:
            from realtime_tts_engine import TTSEngine

            self._engine = TTSEngine(self._synthesizer_instance())
        return self._engine

    def _synthesizer_instance(self):
        if self._synthesizer is None:
            from realtime_tts_engine.kokoro import KokoroSynthesizer

            if not self.model_settings.model_path:
                raise ValueError("kokoro model_path is required")
            self._synthesizer = KokoroSynthesizer(
                model_root=self.model_settings.model_path,
                device=self.model_settings.device,
                speed=self.model_settings.kokoro_speed,
            )
        return self._synthesizer

    def _voice_for_request(self, request: ResponseRequest) -> str | None:
        requested = str(request.voice.preset or "").strip()
        if requested:
            return requested
        return self.model_settings.voice_presets.get(str(request.language or "").strip())
