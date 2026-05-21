from __future__ import annotations

import io
import logging
import math
from pathlib import Path
import struct
import tempfile
import time
from typing import Any
import wave

from app.config import ModelSettings
from app.config import VoxCPM2WarmupCase
from app.schemas import EngineResult
from app.schemas import ResponseRequest

from .common import decode_reference_audio


LOGGER = logging.getLogger("tts_pool.engine.voxcpm2")

DEFAULT_WARMUP_CASES = (
    VoxCPM2WarmupCase(
        name="short_no_ref_10",
        text="Let's see if this works.",
        reference_audio=False,
        inference_timesteps=10,
    ),
    VoxCPM2WarmupCase(
        name="short_ref_10",
        text="Let's see if this works.",
        reference_audio=True,
        inference_timesteps=10,
    ),
    VoxCPM2WarmupCase(
        name="medium_ref_10",
        text="Let's see if this works clearly enough for live translation.",
        reference_audio=True,
        inference_timesteps=10,
    ),
    VoxCPM2WarmupCase(
        name="medium_ref_6",
        text="Let's see if this works clearly enough for live translation.",
        reference_audio=True,
        inference_timesteps=6,
    ),
    VoxCPM2WarmupCase(
        name="medium_ref_4",
        text="Let's see if this works clearly enough for live translation.",
        reference_audio=True,
        inference_timesteps=4,
    ),
)


class VoxCPM2TTSRuntime:
    runtime_capability = 1

    def __init__(self, *, model_name: str, model_settings: ModelSettings) -> None:
        self.model_name = model_name
        self.model_settings = model_settings
        self._model: Any | None = None

    def load(self) -> None:
        model = self._model_instance()
        self._warmup(model)

    def close(self) -> None:
        self._model = None

    def synthesize(self, request: ResponseRequest) -> EngineResult:
        started_at = time.perf_counter()
        model = self._model_instance()
        control = self._control_for_request(request)
        # In ultimate-cloning mode the (control)-prefix is no longer
        # the first thing in the model's tokenized input (prompt_text
        # comes before it), so VoxCPM2 reads it as literal speech and
        # narrates "(Speak in Swedish.)" before the target. Drop the
        # control wrapper here — voice and language are anchored by the
        # paired prompt audio + transcript anyway.
        if _prompt_text_for_request(request):
            control = ""
        text = _voxcpm2_text(request.input, control)
        with tempfile.TemporaryDirectory(prefix="tts-pool-voxcpm2-ref-") as tmpdir:
            reference_path, reference_metadata = self._reference_path(request, Path(tmpdir))
            generation = request.generation.voxcpm2
            cfg_value = (
                self.model_settings.voxcpm2_cfg_value
                if generation.cfg_value is None
                else float(generation.cfg_value)
            )
            inference_timesteps = (
                self.model_settings.voxcpm2_inference_timesteps
                if generation.inference_timesteps is None
                else int(generation.inference_timesteps)
            )
            normalize = (
                self.model_settings.voxcpm2_normalize
                if generation.normalize is None
                else bool(generation.normalize)
            )
            denoise = (
                self.model_settings.voxcpm2_denoise
                if generation.denoise is None
                else bool(generation.denoise)
            )
            # Ultimate-cloning when the request pairs the reference WAV
            # with a transcript. Two sub-modes:
            #   UC2 (default): same WAV passed as both prompt and
            #        reference — the "for better similarity" recipe
            #        from the VoxCPM2 docs.
            #   UC1: prompt-only — reference_wav_path dropped, keeping
            #        only the audio-continuation pair. Set via
            #        reference_audio.also_use_as_reference = False.
            # Reference-only mode applies when no prompt_text is set.
            prompt_text = _prompt_text_for_request(request)
            ultimate_use_reference = (
                prompt_text is not None
                and _ultimate_use_reference_for_request(request)
            )
            reference_metadata["ultimate_cloning"] = bool(prompt_text)
            reference_metadata["ultimate_cloning_with_reference"] = bool(ultimate_use_reference)
            generate_kwargs: dict[str, Any] = {
                "text": text,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
                "normalize": normalize,
                "denoise": denoise,
            }
            include_reference_wav = reference_path is not None and (
                not prompt_text or ultimate_use_reference
            )
            if include_reference_wav:
                generate_kwargs["reference_wav_path"] = str(reference_path)
            if prompt_text and reference_path is not None:
                generate_kwargs["prompt_wav_path"] = str(reference_path)
                generate_kwargs["prompt_text"] = prompt_text
            generate_started = time.perf_counter()
            wav = model.generate(**generate_kwargs)
            generate_wall_ms = (time.perf_counter() - generate_started) * 1000.0

        encode_started = time.perf_counter()
        sample_rate_hz = int(getattr(model.tts_model, "sample_rate", 0) or 0)
        if sample_rate_hz <= 0:
            raise ValueError("VoxCPM2 model did not expose a valid sample rate")
        wav_bytes, duration_ms, audio_seconds = _wav_bytes(wav, sample_rate_hz=sample_rate_hz)
        wav_encode_ms = (time.perf_counter() - encode_started) * 1000.0
        total_wall_ms = (time.perf_counter() - started_at) * 1000.0
        metadata = {
            "engine": "voxcpm2",
            "model_id": self.model_settings.voxcpm2_model_id,
            "language": request.language,
            "device": _voxcpm2_device(model),
            "load_denoiser": self.model_settings.voxcpm2_load_denoiser,
            "optimize": self.model_settings.voxcpm2_optimize,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "normalize": normalize,
            "denoise": denoise,
            "control": control,
            **reference_metadata,
        }
        return EngineResult(
            audio=wav_bytes,
            mime_type="audio/wav",
            sample_rate_hz=sample_rate_hz,
            duration_ms=duration_ms,
            metrics={
                "voxcpm2_total_wall_ms": total_wall_ms,
                "voxcpm2_generate_wall_ms": generate_wall_ms,
                "voxcpm2_wav_encode_ms": wav_encode_ms,
                "input_chars": len(request.input),
                "output_audio_seconds": audio_seconds,
                "realtime_factor": (total_wall_ms / 1000.0) / audio_seconds if audio_seconds > 0 else 0.0,
            },
            metadata=metadata,
        )

    def _model_instance(self) -> Any:
        if self._model is None:
            try:
                from voxcpm import VoxCPM
            except ImportError as exc:
                raise RuntimeError("Install voxcpm to use backend=voxcpm2") from exc
            kwargs = {
                "load_denoiser": self.model_settings.voxcpm2_load_denoiser,
                "optimize": self.model_settings.voxcpm2_optimize,
            }
            self._model = VoxCPM.from_pretrained(self.model_settings.voxcpm2_model_id, **kwargs)
        return self._model

    def _control_for_request(self, request: ResponseRequest) -> str:
        preset = str(request.voice.preset or "").strip()
        if preset:
            raise ValueError("voice.preset is unsupported for voxcpm2; send voice.instructions")
        return str(request.voice.instructions or "").strip()

    def _reference_path(self, request: ResponseRequest, tmpdir: Path) -> tuple[Path | None, dict[str, Any]]:
        reference_audio = request.voice.reference_audio
        if reference_audio is None:
            return None, {"reference_audio": False}
        source_path = tmpdir / "source.wav"
        source_path.write_bytes(decode_reference_audio(reference_audio))
        original_duration_ms = _wav_duration_ms(source_path)
        max_duration_s = (
            self.model_settings.voxcpm2_reference_max_duration_s
            if reference_audio.max_duration_s is None
            else max(1.0, min(60.0, float(reference_audio.max_duration_s)))
        )
        metadata: dict[str, Any] = {
            "reference_audio": True,
            "reference_mime_type": reference_audio.mime_type,
            "reference_max_duration_s": max_duration_s,
            "reference_source_duration_ms": original_duration_ms,
        }
        if original_duration_ms <= int(max_duration_s * 1000):
            metadata["reference_clipped"] = False
            metadata["reference_duration_ms"] = original_duration_ms
            return source_path, metadata
        clipped_path = tmpdir / "reference.wav"
        clipped_duration_ms = _copy_wav_tail(source_path, clipped_path, max_duration_s=max_duration_s)
        metadata["reference_clipped"] = True
        metadata["reference_duration_ms"] = clipped_duration_ms
        return clipped_path, metadata

    def _warmup(self, model: Any) -> None:
        if not self.model_settings.voxcpm2_warmup_enabled:
            return

        cases = self.model_settings.voxcpm2_warmup_cases or DEFAULT_WARMUP_CASES
        LOGGER.info("VoxCPM2 warmup started model=%s cases=%s", self.model_name, len(cases))
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="tts-pool-voxcpm2-warmup-") as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            reference_paths: dict[float, Path] = {}
            for index, case in enumerate(cases, start=1):
                reference_path = None
                if case.reference_audio:
                    reference_path = reference_paths.get(case.reference_duration_s)
                    if reference_path is None:
                        reference_path = tmpdir / f"reference-{len(reference_paths) + 1}.wav"
                        _write_warmup_reference_wav(reference_path, duration_s=case.reference_duration_s)
                        reference_paths[case.reference_duration_s] = reference_path

                cfg_value = (
                    self.model_settings.voxcpm2_cfg_value if case.cfg_value is None else float(case.cfg_value)
                )
                inference_timesteps = (
                    self.model_settings.voxcpm2_inference_timesteps
                    if case.inference_timesteps is None
                    else int(case.inference_timesteps)
                )
                case_started = time.perf_counter()
                model.generate(
                    text=_voxcpm2_text(case.text, ""),
                    reference_wav_path=str(reference_path) if reference_path is not None else None,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    normalize=self.model_settings.voxcpm2_normalize,
                    denoise=self.model_settings.voxcpm2_denoise,
                )
                LOGGER.info(
                    "VoxCPM2 warmup case completed model=%s index=%s/%s name=%s reference_audio=%s "
                    "inference_timesteps=%s wall_ms=%.1f",
                    self.model_name,
                    index,
                    len(cases),
                    case.name or f"case_{index}",
                    case.reference_audio,
                    inference_timesteps,
                    (time.perf_counter() - case_started) * 1000.0,
                )
        LOGGER.info(
            "VoxCPM2 warmup completed model=%s cases=%s wall_ms=%.1f",
            self.model_name,
            len(cases),
            (time.perf_counter() - started) * 1000.0,
        )


def _prompt_text_for_request(request: ResponseRequest) -> str | None:
    reference_audio = request.voice.reference_audio
    if reference_audio is None:
        return None
    text = str(reference_audio.prompt_text or "").strip()
    return text or None


def _ultimate_use_reference_for_request(request: ResponseRequest) -> bool:
    reference_audio = request.voice.reference_audio
    if reference_audio is None:
        return True
    return bool(reference_audio.also_use_as_reference)


def _voxcpm2_text(text: str, control: str) -> str:
    safe_text = str(text or "").strip()
    safe_control = str(control or "").strip()
    if safe_control:
        return f"({safe_control}){safe_text}"
    return safe_text


def _write_warmup_reference_wav(path: Path, *, duration_s: float, sample_rate_hz: int = 16000) -> None:
    frame_count = max(1, int(duration_s * sample_rate_hz))
    amplitude = 0.12 * 32767.0
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        for index in range(frame_count):
            sample = int(amplitude * math.sin(2.0 * math.pi * 220.0 * index / sample_rate_hz))
            writer.writeframesraw(struct.pack("<h", sample))


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as reader:
        framerate = reader.getframerate()
        frames = reader.getnframes()
        if framerate <= 0:
            raise ValueError("reference wav has invalid framerate")
        return int((frames / framerate) * 1000)


def _copy_wav_tail(source_path: Path, target_path: Path, *, max_duration_s: float) -> int:
    with wave.open(str(source_path), "rb") as reader:
        framerate = reader.getframerate()
        if framerate <= 0:
            raise ValueError("reference wav has invalid framerate")
        total_frames = reader.getnframes()
        keep_frames = min(total_frames, max(1, int(max_duration_s * framerate)))
        start_frame = max(0, total_frames - keep_frames)
        reader.setpos(start_frame)
        frames = reader.readframes(keep_frames)
        params = reader.getparams()
    with wave.open(str(target_path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)
    return int((keep_frames / framerate) * 1000)


def _wav_bytes(audio: Any, *, sample_rate_hz: int) -> tuple[bytes, int, float]:
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2 and samples.shape[0] == 1:
        samples = samples[0]
    if samples.ndim != 1:
        raise ValueError("VoxCPM2 returned unsupported audio shape")
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(pcm.tobytes())
    audio_seconds = float(len(pcm)) / float(sample_rate_hz)
    return buffer.getvalue(), int(audio_seconds * 1000), audio_seconds


def _voxcpm2_device(model: Any) -> str:
    device = getattr(getattr(model, "tts_model", None), "device", None)
    if device is None:
        return ""
    return str(device)
