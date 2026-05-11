from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from app.config import ModelSettings
from app.config import NanoVLLMWarmupCase
from app.schemas import EngineResult
from app.schemas import ResponseRequest

from .common import decode_reference_audio
from .voxcpm2 import _append_unique
from .voxcpm2 import _copy_wav_tail
from .voxcpm2 import _voxcpm2_text
from .voxcpm2 import _wav_bytes
from .voxcpm2 import _wav_duration_ms
from .voxcpm2 import _write_warmup_reference_wav
from .voxcpm2 import REFERENCE_CONTROL
from .voxcpm2 import VOICE_PRESETS


LOGGER = logging.getLogger("tts_pool.engine.nanovllm_voxcpm")
DEFAULT_WARMUP_CASES = (
    NanoVLLMWarmupCase(
        name="short_no_ref",
        text="Let's see if this works.",
        reference_audio=False,
        temperature=0.01,
        max_generate_length=256,
    ),
    NanoVLLMWarmupCase(
        name="short_ref_voice",
        text="Let's see if this works.",
        reference_audio=True,
        reference_audio_match="voice",
        reference_duration_s=2.0,
        temperature=0.01,
        max_generate_length=256,
    ),
    NanoVLLMWarmupCase(
        name="short_ref_voice_and_pace",
        text="Let's see if this works.",
        reference_audio=True,
        reference_audio_match="voice_and_pace",
        reference_duration_s=2.0,
        temperature=0.01,
        max_generate_length=256,
    ),
)


class NanoVLLMVoxCPMTTSRuntime:
    def __init__(self, *, model_name: str, model_settings: ModelSettings) -> None:
        self.model_name = model_name
        self.model_settings = model_settings
        self._server: Any | None = None
        self._model_info: dict[str, Any] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()

    @property
    def runtime_capability(self) -> int:
        return max(1, int(self.model_settings.nanovllm_max_num_seqs))

    def load(self) -> None:
        self._start_loop()
        try:
            self._run_on_loop(self._load_async())
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        server = self._server
        if server is not None and self._loop is not None:
            try:
                self._run_on_loop(server.stop())
            except Exception:
                LOGGER.exception("NanoVLLM VoxCPM stop failed model=%s", self.model_name)
        self._server = None
        self._model_info = {}
        self._stop_loop()

    def synthesize(self, request: ResponseRequest) -> EngineResult:
        if self._server is None:
            raise RuntimeError("NanoVLLM VoxCPM model is not loaded")
        started_at = time.perf_counter()
        control = self._control_for_request(request)
        text = _voxcpm2_text(request.input, control)
        reference_started_at = time.perf_counter()
        reference_wav, reference_metadata = self._reference_wav(request)
        reference_prepare_ms = (time.perf_counter() - reference_started_at) * 1000.0
        return self._run_on_loop(
            self._synthesize_async(
                request=request,
                started_at=started_at,
                text=text,
                control=control,
                reference_wav=reference_wav,
                reference_metadata=reference_metadata,
                reference_prepare_ms=reference_prepare_ms,
            )
        )

    async def _load_async(self) -> None:
        try:
            from nanovllm_voxcpm import VoxCPM
        except ImportError as exc:
            raise RuntimeError("Install nano-vllm-voxcpm to use backend=nanovllm_voxcpm") from exc

        self._server = VoxCPM.from_pretrained(
            model=self.model_settings.nanovllm_model_id,
            devices=list(self.model_settings.nanovllm_devices),
            inference_timesteps=self.model_settings.nanovllm_inference_timesteps,
            max_num_batched_tokens=self.model_settings.nanovllm_max_num_batched_tokens,
            max_num_seqs=self.model_settings.nanovllm_max_num_seqs,
            max_model_len=self.model_settings.nanovllm_max_model_len,
            gpu_memory_utilization=self.model_settings.nanovllm_gpu_memory_utilization,
        )
        await self._server.wait_for_ready()
        self._model_info = _model_info_dict(await self._server.get_model_info())
        await self._warmup_async()

    async def _synthesize_async(
        self,
        *,
        request: ResponseRequest,
        started_at: float,
        text: str,
        control: str,
        reference_wav: bytes | None,
        reference_metadata: dict[str, Any],
        reference_prepare_ms: float,
    ) -> EngineResult:
        server = self._server
        if server is None:
            raise RuntimeError("NanoVLLM VoxCPM model is not loaded")

        reference_latents: bytes | None = None
        reference_encode_ms = 0.0
        if reference_wav is not None:
            encode_started = time.perf_counter()
            reference_latents = await server.encode_latents(reference_wav, "wav")
            reference_encode_ms = (time.perf_counter() - encode_started) * 1000.0

        generation = request.generation.nanovllm_voxcpm
        cfg_value = (
            self.model_settings.nanovllm_cfg_value
            if generation.cfg_value is None
            else float(generation.cfg_value)
        )
        temperature = (
            self.model_settings.nanovllm_temperature
            if generation.temperature is None
            else float(generation.temperature)
        )
        max_generate_length = (
            self.model_settings.nanovllm_max_generate_length
            if generation.max_generate_length is None
            else int(generation.max_generate_length)
        )

        import numpy as np

        chunks: list[Any] = []
        first_chunk_ms: float | None = None
        generate_started = time.perf_counter()
        async for chunk in server.generate(
            target_text=text,
            max_generate_length=max_generate_length,
            temperature=temperature,
            cfg_value=cfg_value,
            ref_audio_latents=reference_latents,
        ):
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - generate_started) * 1000.0
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        generate_wall_ms = (time.perf_counter() - generate_started) * 1000.0
        if not chunks:
            raise ValueError("NanoVLLM VoxCPM returned no audio chunks")

        encode_started = time.perf_counter()
        sample_rate_hz = int(self._model_info.get("sample_rate") or 0)
        if sample_rate_hz <= 0:
            raise ValueError("NanoVLLM VoxCPM model did not expose a valid sample rate")
        wav_bytes, duration_ms, audio_seconds = _wav_bytes(np.concatenate(chunks), sample_rate_hz=sample_rate_hz)
        wav_encode_ms = (time.perf_counter() - encode_started) * 1000.0
        total_wall_ms = (time.perf_counter() - started_at) * 1000.0
        metadata = {
            "engine": "nanovllm_voxcpm",
            "model_id": self.model_settings.nanovllm_model_id,
            "language": request.language,
            "devices": list(self.model_settings.nanovllm_devices),
            "cfg_value": cfg_value,
            "temperature": temperature,
            "inference_timesteps": self.model_settings.nanovllm_inference_timesteps,
            "max_generate_length": max_generate_length,
            "control": control,
            **reference_metadata,
        }
        return EngineResult(
            audio=wav_bytes,
            mime_type="audio/wav",
            sample_rate_hz=sample_rate_hz,
            duration_ms=duration_ms,
            metrics={
                "nanovllm_total_wall_ms": total_wall_ms,
                "nanovllm_reference_prepare_wall_ms": reference_prepare_ms,
                "nanovllm_reference_encode_wall_ms": reference_encode_ms,
                "nanovllm_generate_wall_ms": generate_wall_ms,
                "nanovllm_first_chunk_wall_ms": first_chunk_ms,
                "nanovllm_wav_encode_ms": wav_encode_ms,
                "nanovllm_chunks": len(chunks),
                "input_chars": len(request.input),
                "output_audio_seconds": audio_seconds,
                "realtime_factor": (total_wall_ms / 1000.0) / audio_seconds if audio_seconds > 0 else 0.0,
            },
            metadata=metadata,
        )

    def _control_for_request(self, request: ResponseRequest) -> str:
        fragments: list[str] = []
        preset = str(request.voice.preset or "").strip()
        if not preset:
            preset = self.model_settings.voice_presets.get(str(request.language or "").strip(), "configured")
        if preset:
            if preset not in VOICE_PRESETS:
                raise ValueError(f"unsupported nanovllm_voxcpm voice preset: {preset}")
            _append_unique(fragments, VOICE_PRESETS[preset])
        _append_unique(fragments, str(request.voice.instructions or "").strip())
        if request.voice.reference_audio is not None and request.voice.reference_audio_match == "voice_and_pace":
            _append_unique(fragments, REFERENCE_CONTROL)
        return " ".join(fragments).strip()

    def _reference_wav(self, request: ResponseRequest) -> tuple[bytes | None, dict[str, Any]]:
        reference_audio = request.voice.reference_audio
        if reference_audio is None:
            return None, {"reference_audio": False}
        with tempfile.TemporaryDirectory(prefix="tts-pool-nanovllm-voxcpm-ref-") as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            source_path = tmpdir / "source.wav"
            source_path.write_bytes(decode_reference_audio(reference_audio))
            original_duration_ms = _wav_duration_ms(source_path)
            max_duration_s = (
                self.model_settings.nanovllm_reference_max_duration_s
                if reference_audio.max_duration_s is None
                else max(1.0, min(60.0, float(reference_audio.max_duration_s)))
            )
            metadata: dict[str, Any] = {
                "reference_audio": True,
                "reference_mime_type": reference_audio.mime_type,
                "reference_audio_match": request.voice.reference_audio_match,
                "reference_max_duration_s": max_duration_s,
                "reference_source_duration_ms": original_duration_ms,
            }
            if original_duration_ms <= int(max_duration_s * 1000):
                metadata["reference_clipped"] = False
                metadata["reference_duration_ms"] = original_duration_ms
                return source_path.read_bytes(), metadata
            clipped_path = tmpdir / "reference.wav"
            clipped_duration_ms = _copy_wav_tail(source_path, clipped_path, max_duration_s=max_duration_s)
            metadata["reference_clipped"] = True
            metadata["reference_duration_ms"] = clipped_duration_ms
            return clipped_path.read_bytes(), metadata

    async def _warmup_async(self) -> None:
        if not self.model_settings.nanovllm_warmup_enabled:
            return
        server = self._server
        if server is None:
            raise RuntimeError("NanoVLLM VoxCPM model is not loaded")

        cases = self.model_settings.nanovllm_warmup_cases or DEFAULT_WARMUP_CASES
        LOGGER.info("NanoVLLM VoxCPM warmup started model=%s cases=%s", self.model_name, len(cases))
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="tts-pool-nanovllm-warmup-") as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            reference_wavs: dict[float, bytes] = {}
            for index, case in enumerate(cases, start=1):
                reference_latents = None
                reference_encode_ms = 0.0
                if case.reference_audio:
                    reference_wav = reference_wavs.get(case.reference_duration_s)
                    if reference_wav is None:
                        reference_path = tmpdir / f"reference-{len(reference_wavs) + 1}.wav"
                        _write_warmup_reference_wav(reference_path, duration_s=case.reference_duration_s)
                        reference_wav = reference_path.read_bytes()
                        reference_wavs[case.reference_duration_s] = reference_wav
                    encode_started = time.perf_counter()
                    reference_latents = await server.encode_latents(reference_wav, "wav")
                    reference_encode_ms = (time.perf_counter() - encode_started) * 1000.0

                cfg_value = (
                    self.model_settings.nanovllm_cfg_value if case.cfg_value is None else float(case.cfg_value)
                )
                temperature = (
                    self.model_settings.nanovllm_temperature if case.temperature is None else float(case.temperature)
                )
                max_generate_length = (
                    self.model_settings.nanovllm_max_generate_length
                    if case.max_generate_length is None
                    else int(case.max_generate_length)
                )
                generate_started = time.perf_counter()
                chunks = 0
                async for _chunk in server.generate(
                    target_text=_voxcpm2_text(case.text, _control_for_warmup_case(case)),
                    max_generate_length=max_generate_length,
                    temperature=temperature,
                    cfg_value=cfg_value,
                    ref_audio_latents=reference_latents,
                ):
                    chunks += 1
                generate_ms = (time.perf_counter() - generate_started) * 1000.0
                LOGGER.info(
                    "NanoVLLM VoxCPM warmup case completed model=%s index=%s/%s name=%s "
                    "reference_audio=%s reference_audio_match=%s temperature=%.3f "
                    "max_generate_length=%s reference_encode_ms=%.1f generate_ms=%.1f chunks=%s",
                    self.model_name,
                    index,
                    len(cases),
                    case.name or f"case_{index}",
                    case.reference_audio,
                    case.reference_audio_match,
                    temperature,
                    max_generate_length,
                    reference_encode_ms,
                    generate_ms,
                    chunks,
                )
        LOGGER.info(
            "NanoVLLM VoxCPM warmup completed model=%s cases=%s wall_ms=%.1f",
            self.model_name,
            len(cases),
            (time.perf_counter() - started) * 1000.0,
        )

    def _start_loop(self) -> None:
        with self._loop_lock:
            if self._loop is not None:
                return
            loop = asyncio.new_event_loop()
            ready = threading.Event()
            thread = threading.Thread(
                target=_run_loop,
                args=(loop, ready),
                name=f"tts-pool-nanovllm-loop-{self.model_name}",
                daemon=True,
            )
            self._loop = loop
            self._loop_thread = thread
            thread.start()
            ready.wait()

    def _stop_loop(self) -> None:
        with self._loop_lock:
            loop = self._loop
            thread = self._loop_thread
            self._loop = None
            self._loop_thread = None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5.0)

    def _run_on_loop(self, coroutine: Any) -> Any:
        loop = self._loop
        if loop is None:
            raise RuntimeError("NanoVLLM VoxCPM event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future.result()


def _run_loop(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    asyncio.set_event_loop(loop)
    ready.set()
    try:
        loop.run_forever()
    finally:
        loop.close()


def _model_info_dict(model_info: Any) -> dict[str, Any]:
    if isinstance(model_info, dict):
        return dict(model_info)
    model_dump = getattr(model_info, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump())
    if hasattr(model_info, "_asdict"):
        return dict(model_info._asdict())
    return dict(getattr(model_info, "__dict__", {}))


def _control_for_warmup_case(case: NanoVLLMWarmupCase) -> str:
    fragments: list[str] = []
    preset = str(case.voice_preset or "configured").strip() or "configured"
    if preset not in VOICE_PRESETS:
        raise ValueError(f"unsupported nanovllm_voxcpm warmup voice preset: {preset}")
    _append_unique(fragments, VOICE_PRESETS[preset])
    reference_match = str(case.reference_audio_match or "voice").strip() or "voice"
    if reference_match not in {"voice", "voice_and_pace"}:
        raise ValueError(f"unsupported nanovllm_voxcpm warmup reference_audio_match: {reference_match}")
    if case.reference_audio and reference_match == "voice_and_pace":
        _append_unique(fragments, REFERENCE_CONTROL)
    return " ".join(fragments).strip()
