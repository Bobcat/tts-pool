from __future__ import annotations

import logging
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

from app.config import ModelSettings
from app.schemas import EngineResult
from app.schemas import ResponseRequest
from app.schemas import VoiceSpec


LOGGER = logging.getLogger("tts_pool.kokoro")
KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE_HZ = 24_000

LANGUAGE_TO_PIPELINE = {
    "american english": "a",
    "en": "a",
    "en-us": "a",
    "english": "a",
    "british english": "b",
    "en-gb": "b",
    "es": "e",
    "es-es": "e",
    "spanish": "e",
    "fr": "f",
    "fr-fr": "f",
    "french": "f",
    "hi": "h",
    "hi-in": "h",
    "hindi": "h",
    "it": "i",
    "it-it": "i",
    "italian": "i",
    "brazilian portuguese": "p",
    "pt": "p",
    "pt-br": "p",
    "portuguese": "p",
    "ja": "j",
    "ja-jp": "j",
    "japanese": "j",
    "chinese": "z",
    "zh": "z",
    "zh-cn": "z",
    "cmn": "z",
    "mandarin chinese": "z",
}

DEFAULT_VOICE_BY_PIPELINE = {
    "a": "af_heart",
    "b": "bf_emma",
    "e": "ef_dora",
    "f": "ff_siwis",
    "h": "hf_alpha",
    "i": "if_sara",
    "p": "pf_dora",
    "j": "jf_alpha",
    "z": "zf_xiaobei",
}


class KokoroTTSRuntime:
    runtime_capability = 1

    def __init__(self, *, model_name: str, model_settings: ModelSettings) -> None:
        self.model_name = model_name
        self.model_settings = model_settings
        self.model_root = Path(model_settings.model_path).expanduser().resolve() if model_settings.model_path else None
        self.device = model_settings.device
        self._model: Any | None = None
        self._pipelines: dict[str, Any] = {}

    def load(self) -> None:
        self._model_instance()
        self._warmup()

    def close(self) -> None:
        self._pipelines = {}
        self._model = None

    def synthesize(self, request: ResponseRequest) -> EngineResult:
        started_at = time.perf_counter()
        if request.voice.reference_audio is not None:
            raise ValueError("kokoro does not support reference_audio")

        text = str(request.input or "").strip()
        if not text:
            raise ValueError("Kokoro input text must not be empty")
        language = str(request.language or "").strip()
        if not language:
            raise ValueError("Kokoro language must not be empty")

        preprocess_started_at = time.perf_counter()
        lang_code = _lang_code(language)
        voice_name = _voice_name(lang_code, self._voice_for_request(request))
        voice_path = self._voice_path(voice_name)
        model_cold_start = self._model is None
        pipeline_cold_start = lang_code not in self._pipelines
        pipeline = self._pipeline(lang_code)
        voice_cache_key = str(voice_path)
        voice_cold_start = _pipeline_voice_cold(pipeline, voice_cache_key)
        model = self._model
        device = _model_device_name(model, self.device)
        preprocess_ms = (time.perf_counter() - preprocess_started_at) * 1000.0

        speed = (
            self.model_settings.kokoro_speed
            if request.generation.kokoro.speed is None
            else float(request.generation.kokoro.speed)
        )
        chunks: list[Any] = []
        chunk_timings: list[dict[str, float | int]] = []
        generator_step_ms = 0.0
        postprocess_ms = 0.0
        first_audio_ms: float | None = None
        cuda_synchronized = False
        pipeline_started_at = time.perf_counter()
        iterator = iter(pipeline(text, voice=voice_cache_key, speed=speed))
        while True:
            step_started_at = time.perf_counter()
            cuda_synchronized = _cuda_synchronize(model) or cuda_synchronized
            try:
                result = next(iterator)
            except StopIteration:
                break
            cuda_synchronized = _cuda_synchronize(model) or cuda_synchronized
            step_ms = (time.perf_counter() - step_started_at) * 1000.0
            generator_step_ms += step_ms
            audio_value = getattr(result, "audio", None)
            if audio_value is None:
                continue
            if first_audio_ms is None:
                first_audio_ms = (time.perf_counter() - started_at) * 1000.0
            post_started_at = time.perf_counter()
            audio = _audio_to_numpy(audio_value)
            postprocess_ms += (time.perf_counter() - post_started_at) * 1000.0
            chunks.append(audio)
            chunk_timings.append(
                {
                    "index": len(chunks),
                    "generator_step_wall_ms": step_ms,
                    "grapheme_chars": len(str(getattr(result, "graphemes", "") or "")),
                    "phoneme_chars": len(str(getattr(result, "phonemes", "") or "")),
                    "audio_seconds": len(audio) / KOKORO_SAMPLE_RATE_HZ,
                }
            )
        pipeline_ms = (time.perf_counter() - pipeline_started_at) * 1000.0
        if not chunks:
            raise ValueError("Kokoro returned no audio")

        wav_started_at = time.perf_counter()
        audio = _concat_audio(chunks)
        wav_bytes = _wav_bytes(audio)
        wav_encode_ms = (time.perf_counter() - wav_started_at) * 1000.0
        postprocess_ms += wav_encode_ms
        duration_ms = int(len(audio) / KOKORO_SAMPLE_RATE_HZ * 1000)
        audio_seconds = len(audio) / KOKORO_SAMPLE_RATE_HZ
        total_ms = (time.perf_counter() - started_at) * 1000.0

        return EngineResult(
            audio=wav_bytes,
            mime_type="audio/wav",
            sample_rate_hz=KOKORO_SAMPLE_RATE_HZ,
            duration_ms=duration_ms,
            metrics={
                "kokoro_total_wall_ms": total_ms,
                "kokoro_pipeline_wall_ms": pipeline_ms,
                "kokoro_preprocess_ms": preprocess_ms,
                "kokoro_model_inference_ms": generator_step_ms,
                "kokoro_postprocess_ms": postprocess_ms,
                "kokoro_wav_encode_ms": wav_encode_ms,
                "kokoro_first_audio_wall_ms": first_audio_ms,
                "kokoro_chunks": len(chunks),
                "input_chars": len(text),
                "output_audio_seconds": audio_seconds,
                "realtime_factor": (total_ms / 1000.0) / audio_seconds if audio_seconds > 0 else 0.0,
                "model_cold_start": 1.0 if model_cold_start else 0.0,
                "pipeline_cold_start": 1.0 if pipeline_cold_start else 0.0,
                "voice_cold_start": 1.0 if voice_cold_start else 0.0,
            },
            metadata={
                "engine": "kokoro",
                "model_id": KOKORO_REPO_ID,
                "model_root": str(self.model_root or ""),
                "language": language,
                "language_code": lang_code,
                "voice": voice_name,
                "device": device,
                "speed": speed,
                "cuda_synchronized": cuda_synchronized,
                "model_cold_start": model_cold_start,
                "pipeline_cold_start": pipeline_cold_start,
                "voice_cold_start": voice_cold_start,
                "voice_instructions_ignored": bool(str(request.voice.instructions or "").strip()),
                "chunk_timings": chunk_timings,
            },
        )

    def _model_instance(self) -> Any:
        if self._model is None:
            try:
                import torch
                from kokoro import KModel
            except ImportError as exc:
                raise RuntimeError("Install Kokoro dependencies to use backend=kokoro") from exc

            if self.model_root is None:
                raise ValueError("kokoro model_path is required")
            model_path = self.model_root / "kokoro-v1_0.pth"
            config_path = self.model_root / "config.json"
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            if not config_path.exists():
                raise FileNotFoundError(config_path)

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = str(device)
            self._model = KModel(
                repo_id=KOKORO_REPO_ID,
                config=str(config_path),
                model=str(model_path),
            ).to(device).eval()
        return self._model

    def _pipeline(self, lang_code: str) -> Any:
        if lang_code not in self._pipelines:
            try:
                from kokoro import KPipeline
            except ImportError as exc:
                raise RuntimeError("Install Kokoro dependencies to use backend=kokoro") from exc
            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id=KOKORO_REPO_ID,
                model=self._model_instance(),
            )
        return self._pipelines[lang_code]

    def _voice_path(self, voice_name: str) -> Path:
        if self.model_root is None:
            raise ValueError("kokoro model_path is required")
        if "/" in voice_name or "\\" in voice_name:
            raise ValueError(f"unsafe Kokoro voice name: {voice_name!r}")
        path = self.model_root / "voices" / f"{voice_name}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _voice_for_request(self, request: ResponseRequest) -> str | None:
        requested = str(request.voice.preset or "").strip()
        if requested:
            return requested
        return self.model_settings.voice_presets.get(str(request.language or "").strip())

    def _warmup(self) -> None:
        if not self.model_settings.kokoro_warmup_enabled:
            return
        languages = _warmup_languages(self.model_settings)
        if not languages:
            return
        started_at = time.perf_counter()
        LOGGER.info("Kokoro warmup started model=%s languages=%s", self.model_name, len(languages))
        completed = 0
        for language in languages:
            try:
                _lang_code(language)
            except ValueError:
                LOGGER.warning("Kokoro warmup skipped unsupported language=%s", language)
                continue
            case_started_at = time.perf_counter()
            voice = _voice_preset_for_language(self.model_settings, language)
            try:
                result = self.synthesize(
                    ResponseRequest(
                        model=self.model_name,
                        input=_warmup_text(language),
                        language=language,
                        voice=VoiceSpec(preset=voice),
                        generation={"kokoro": {"speed": self.model_settings.kokoro_speed}},
                    )
                )
            except Exception as exc:
                LOGGER.warning(
                    "Kokoro warmup case failed model=%s language=%s error=%s",
                    self.model_name,
                    language,
                    exc,
                )
                continue
            completed += 1
            LOGGER.info(
                "Kokoro warmup case completed model=%s index=%s/%s language=%s voice=%s wall_ms=%.1f",
                self.model_name,
                completed,
                len(languages),
                language,
                result.metadata.get("voice"),
                (time.perf_counter() - case_started_at) * 1000.0,
            )
        LOGGER.info(
            "Kokoro warmup completed model=%s cases=%s wall_ms=%.1f",
            self.model_name,
            completed,
            (time.perf_counter() - started_at) * 1000.0,
        )


def _lang_code(language: str) -> str:
    key = str(language or "").strip().lower().replace("_", "-")
    try:
        return LANGUAGE_TO_PIPELINE[key]
    except KeyError as exc:
        supported = ", ".join(sorted(LANGUAGE_TO_PIPELINE))
        raise ValueError(f"unsupported Kokoro language {language!r}; supported: {supported}") from exc


def _voice_name(lang_code: str, voice: str | None) -> str:
    if voice is None:
        return DEFAULT_VOICE_BY_PIPELINE[lang_code]
    voice_name = str(voice or "").strip()
    if not voice_name:
        return DEFAULT_VOICE_BY_PIPELINE[lang_code]
    if voice_name.endswith(".pt"):
        voice_name = voice_name[:-3]
    if not voice_name.startswith(lang_code):
        raise ValueError(f"Kokoro voice {voice_name!r} does not match language pipeline {lang_code!r}")
    return voice_name


def _warmup_languages(model_settings: ModelSettings) -> tuple[str, ...]:
    candidates = model_settings.kokoro_warmup_languages or tuple(model_settings.voice_presets) or ("English",)
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        language = str(candidate or "").strip()
        key = language.lower()
        if not language or key in seen:
            continue
        seen.add(key)
        out.append(language)
    return tuple(out)


def _voice_preset_for_language(model_settings: ModelSettings, language: str) -> str | None:
    text = str(language or "").strip()
    if text in model_settings.voice_presets:
        return model_settings.voice_presets[text]
    folded = text.lower()
    for candidate, voice in model_settings.voice_presets.items():
        if str(candidate or "").strip().lower() == folded:
            return voice
    return None


def _warmup_text(language: str) -> str:
    key = str(language or "").strip().lower().replace("_", "-")
    return {
        "chinese": "准备好了。",
        "zh": "准备好了。",
        "zh-cn": "准备好了。",
        "japanese": "準備できました。",
        "ja": "準備できました。",
        "ja-jp": "準備できました。",
        "french": "Pret.",
        "fr": "Pret.",
        "spanish": "Listo.",
        "es": "Listo.",
        "italian": "Pronto.",
        "it": "Pronto.",
        "portuguese": "Pronto.",
        "pt": "Pronto.",
        "brazilian portuguese": "Pronto.",
        "hindi": "Taiyar hai.",
        "hi": "Taiyar hai.",
    }.get(key, "Ready.")


def _model_device_name(model: Any, configured_device: str | None) -> str:
    device = getattr(model, "device", None)
    if device is None:
        device = configured_device
    return str(device or "")


def _pipeline_voice_cold(pipeline: Any, voice_cache_key: str) -> bool:
    voices = getattr(pipeline, "voices", None)
    if not isinstance(voices, dict):
        return False
    return str(voice_cache_key or "") not in voices


def _cuda_synchronize(model: Any) -> bool:
    device = getattr(model, "device", None)
    if device is None or not str(device).startswith("cuda"):
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.cuda.synchronize(device)
        return True
    except Exception:
        return False


def _audio_to_numpy(audio: Any) -> Any:
    detach = getattr(audio, "detach", None)
    if callable(detach):
        audio = detach()
    cpu = getattr(audio, "cpu", None)
    if callable(cpu):
        audio = cpu()
    numpy = getattr(audio, "numpy", None)
    if callable(numpy):
        return numpy()
    import numpy as np

    return np.asarray(audio)


def _concat_audio(chunks: list[Any]) -> Any:
    if len(chunks) == 1:
        return chunks[0]
    import numpy as np

    return np.concatenate(chunks)


def _wav_bytes(audio: Any) -> bytes:
    import numpy as np

    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(KOKORO_SAMPLE_RATE_HZ)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()
