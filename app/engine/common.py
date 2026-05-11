from __future__ import annotations

from dataclasses import dataclass
import base64
import logging
from pathlib import Path
import subprocess
from typing import Any
from typing import Literal

from app.config import ModelSettings
from app.schemas import ReferenceAudio


LOGGER = logging.getLogger("tts_pool.engine")
ModelLifecycle = Literal["unloaded", "loading", "loaded", "unloading", "failed"]


class UnknownModelError(Exception):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.model_name = model_name


class ModelStateError(Exception):
    def __init__(self, model_name: str, code: str) -> None:
        super().__init__(f"{model_name}: {code}")
        self.model_name = model_name
        self.code = code


@dataclass
class ModelRuntimeState:
    resolved_backend: str
    configured_enabled: bool
    lifecycle: ModelLifecycle = "unloaded"
    inflight_requests: int = 0
    configured_target_inflight: int = 1
    effective_target_inflight: int = 1
    last_error: str | None = None


def exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def decode_reference_audio(reference_audio: ReferenceAudio) -> bytes:
    mime_type = str(reference_audio.mime_type or "").strip().lower()
    if mime_type not in {"audio/wav", "audio/x-wav", "audio/wave"}:
        raise ValueError(f"unsupported reference_audio mime_type: {reference_audio.mime_type!r}")
    try:
        return base64.b64decode(reference_audio.data_base64, validate=True)
    except Exception as exc:
        raise ValueError("reference_audio.data_base64 is not valid base64") from exc


def estimate_model_artifact_size_mib(model_path: str) -> int | None:
    path_value = str(model_path or "").strip()
    if not path_value:
        return None
    path = Path(path_value)
    try:
        if path.is_file():
            total_bytes = path.stat().st_size
        elif path.is_dir():
            total_bytes = 0
            for candidate in path.rglob("*"):
                if candidate.is_file():
                    total_bytes += candidate.stat().st_size
        else:
            return None
    except OSError:
        return None
    if total_bytes <= 0:
        return None
    return max(1, int(total_bytes / (1024 * 1024)))


def query_gpu_memory() -> tuple[list[dict[str, object]], str | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return [], "nvidia-smi not found"
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
        return [], message or "nvidia-smi failed"

    gpus: list[dict[str, object]] = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            used_mib = int(parts[2])
            total_mib = int(parts[3])
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "used_mib": used_mib,
                "total_mib": total_mib,
                "used_over_total": f"{used_mib}MiB / {total_mib}MiB",
            }
        )
    return gpus, None


def model_definition_payload(model_settings: ModelSettings, *, resolved_backend: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": model_settings.backend,
        "model_path": model_settings.model_path,
        "enabled": model_settings.enabled,
        "target_inflight": model_settings.target_inflight,
    }
    if resolved_backend == "kokoro":
        payload.update(
            {
                "device": model_settings.device,
                "voice_presets": dict(model_settings.voice_presets),
                "kokoro_speed": model_settings.kokoro_speed,
            }
        )
    elif resolved_backend == "voxcpm2":
        payload.update(
            {
                "voxcpm2_model_id": model_settings.voxcpm2_model_id,
                "voxcpm2_load_denoiser": model_settings.voxcpm2_load_denoiser,
                "voxcpm2_optimize": model_settings.voxcpm2_optimize,
                "voxcpm2_cfg_value": model_settings.voxcpm2_cfg_value,
                "voxcpm2_inference_timesteps": model_settings.voxcpm2_inference_timesteps,
                "voxcpm2_normalize": model_settings.voxcpm2_normalize,
                "voxcpm2_denoise": model_settings.voxcpm2_denoise,
                "voxcpm2_reference_max_duration_s": model_settings.voxcpm2_reference_max_duration_s,
                "voxcpm2_warmup_enabled": model_settings.voxcpm2_warmup_enabled,
                "voxcpm2_warmup_cases": [vars(case) for case in model_settings.voxcpm2_warmup_cases],
            }
        )
    elif resolved_backend == "nanovllm_voxcpm":
        payload.update(
            {
                "nanovllm_model_id": model_settings.nanovllm_model_id,
                "nanovllm_devices": list(model_settings.nanovllm_devices),
                "nanovllm_max_num_seqs": model_settings.nanovllm_max_num_seqs,
                "nanovllm_max_num_batched_tokens": model_settings.nanovllm_max_num_batched_tokens,
                "nanovllm_max_model_len": model_settings.nanovllm_max_model_len,
                "nanovllm_gpu_memory_utilization": model_settings.nanovllm_gpu_memory_utilization,
                "nanovllm_inference_timesteps": model_settings.nanovllm_inference_timesteps,
                "nanovllm_max_generate_length": model_settings.nanovllm_max_generate_length,
                "nanovllm_temperature": model_settings.nanovllm_temperature,
                "nanovllm_cfg_value": model_settings.nanovllm_cfg_value,
                "nanovllm_reference_max_duration_s": model_settings.nanovllm_reference_max_duration_s,
                "nanovllm_warmup_enabled": model_settings.nanovllm_warmup_enabled,
                "nanovllm_warmup_cases": [vars(case) for case in model_settings.nanovllm_warmup_cases],
            }
        )
    return payload


def _voxcpm2_languages() -> list[str]:
    return [
        "Arabic",
        "Chinese",
        "Danish",
        "Dutch",
        "English",
        "Finnish",
        "French",
        "German",
        "Greek",
        "Hebrew",
        "Hindi",
        "Indonesian",
        "Italian",
        "Japanese",
        "Khmer",
        "Korean",
        "Lao",
        "Malay",
        "Norwegian",
        "Polish",
        "Portuguese",
        "Russian",
        "Spanish",
        "Swahili",
        "Swedish",
        "Tagalog",
        "Thai",
        "Turkish",
    ]


def capabilities_for_backend(backend: str, model_settings: ModelSettings) -> dict[str, Any]:
    normalized = backend.strip().lower()
    if normalized == "stub":
        return {
            "languages": ["Dutch", "English"],
            "output_formats": ["wav"],
            "voice_presets": False,
            "voice_instructions": False,
            "reference_audio": False,
            "streaming": False,
        }
    if normalized == "kokoro":
        return {
            "languages": [
                "English",
                "British English",
                "Spanish",
                "French",
                "Hindi",
                "Italian",
                "Portuguese",
                "Brazilian Portuguese",
                "Japanese",
                "Chinese",
            ],
            "output_formats": ["wav"],
            "voice_presets": True,
            "voice_instructions": False,
            "reference_audio": False,
            "request_generation": {
                "kokoro.speed": {"kind": "number", "minimum": 0.25, "maximum": 4.0, "default": model_settings.kokoro_speed},
            },
            "streaming": False,
        }
    if normalized == "voxcpm2":
        return {
            "languages": _voxcpm2_languages(),
            "output_formats": ["wav"],
            "voice_presets": False,
            "voice_instructions": True,
            "reference_audio": True,
            "reference_audio_mime_types": ["audio/wav"],
            "reference_audio_max_duration_s": model_settings.voxcpm2_reference_max_duration_s,
            "request_generation": {
                "voxcpm2.cfg_value": {"kind": "number", "minimum": 0.1, "maximum": 10.0, "default": model_settings.voxcpm2_cfg_value},
                "voxcpm2.inference_timesteps": {
                    "kind": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "default": model_settings.voxcpm2_inference_timesteps,
                },
                "voxcpm2.normalize": {"kind": "boolean", "default": model_settings.voxcpm2_normalize},
                "voxcpm2.denoise": {"kind": "boolean", "default": model_settings.voxcpm2_denoise},
            },
            "streaming": False,
        }
    if normalized == "nanovllm_voxcpm":
        return {
            "languages": _voxcpm2_languages(),
            "output_formats": ["wav"],
            "voice_presets": False,
            "voice_instructions": True,
            "reference_audio": True,
            "reference_audio_mime_types": ["audio/wav"],
            "reference_audio_max_duration_s": model_settings.nanovllm_reference_max_duration_s,
            "request_generation": {
                "nanovllm_voxcpm.cfg_value": {"kind": "number", "minimum": 0.1, "maximum": 10.0, "default": model_settings.nanovllm_cfg_value},
                "nanovllm_voxcpm.temperature": {
                    "kind": "number",
                    "minimum": 0.01,
                    "maximum": 5.0,
                    "default": model_settings.nanovllm_temperature,
                },
                "nanovllm_voxcpm.max_generate_length": {
                    "kind": "integer",
                    "minimum": 1,
                    "maximum": 4096,
                    "default": model_settings.nanovllm_max_generate_length,
                },
            },
            "streaming": False,
        }
    return {"output_formats": ["wav"], "streaming": False}
