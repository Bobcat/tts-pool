from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.json"
DEFAULT_LOCAL_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "local.json"
VOXCPM2_DEFAULT_MODEL_ID = "openbmb/VoxCPM2"


@dataclass(frozen=True)
class VoxCPM2WarmupCase:
    name: str = ""
    text: str = "Let's see if this works."
    reference_audio: bool = False
    reference_duration_s: float = 4.0
    cfg_value: float | None = None
    inference_timesteps: int | None = None


@dataclass(frozen=True)
class NanoVLLMWarmupCase:
    name: str = ""
    text: str = "Let's see if this works."
    reference_audio: bool = False
    reference_duration_s: float = 2.0
    cfg_value: float | None = None
    temperature: float | None = None
    max_generate_length: int | None = None


@dataclass(frozen=True)
class GrpcSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8021
    max_receive_message_bytes: int = 16_777_216
    max_send_message_bytes: int = 16_777_216
    stream_buffer_chunks: int = 4
    stream_buffer_bytes: int = 1_048_576
    stalled_consumer_timeout_s: float = 2.0
    shutdown_grace_s: float = 5.0


@dataclass(frozen=True)
class ServiceSettings:
    host: str = "127.0.0.1"
    port: int = 8020
    log_level: str = "info"
    grpc: GrpcSettings = field(default_factory=GrpcSettings)


@dataclass(frozen=True)
class ModelSettings:
    backend: str | None = None
    model_path: str = ""
    enabled: bool = True
    target_inflight: int = 1
    device: str | None = None
    voice_presets: dict[str, str] = field(default_factory=dict)
    kokoro_speed: float = 1.0
    kokoro_warmup_enabled: bool = False
    kokoro_warmup_languages: tuple[str, ...] = field(default_factory=tuple)
    voxcpm2_model_id: str = VOXCPM2_DEFAULT_MODEL_ID
    voxcpm2_load_denoiser: bool = False
    voxcpm2_optimize: bool = False
    voxcpm2_cfg_value: float = 2.0
    voxcpm2_inference_timesteps: int = 10
    voxcpm2_normalize: bool = False
    voxcpm2_denoise: bool = False
    voxcpm2_reference_max_duration_s: float = 8.0
    voxcpm2_warmup_enabled: bool = False
    voxcpm2_warmup_cases: tuple[VoxCPM2WarmupCase, ...] = field(default_factory=tuple)
    nanovllm_model_id: str = VOXCPM2_DEFAULT_MODEL_ID
    nanovllm_devices: tuple[int, ...] = (0,)
    nanovllm_max_num_seqs: int = 4
    nanovllm_max_num_batched_tokens: int = 4096
    nanovllm_max_model_len: int = 2048
    nanovllm_gpu_memory_utilization: float = 0.85
    nanovllm_inference_timesteps: int = 10
    nanovllm_max_generate_length: int = 256
    nanovllm_temperature: float = 1.0
    nanovllm_cfg_value: float = 2.0
    nanovllm_reference_max_duration_s: float = 8.0
    nanovllm_warmup_enabled: bool = False
    nanovllm_warmup_cases: tuple[NanoVLLMWarmupCase, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FairnessSettings:
    default_weight: float = 1.0
    weights: dict[str, float] = field(default_factory=dict)
    soft_max_inflight_per_key: int = 1
    max_pending_per_key: int = 4
    max_pending_per_executor: int = 8
    idle_state_ttl_s: float = 300.0


@dataclass(frozen=True)
class EngineSettings:
    backend: str = "stub"
    models: dict[str, ModelSettings] = field(default_factory=dict)
    fairness: FairnessSettings = field(default_factory=FairnessSettings)


@dataclass(frozen=True)
class AppSettings:
    service: ServiceSettings = field(default_factory=ServiceSettings)
    engine: EngineSettings = field(default_factory=EngineSettings)


def load_settings(path: str | Path | None = None) -> AppSettings:
    settings_path = _resolve_settings_path(path)
    payload: dict[str, Any] = {}
    if settings_path.exists():
        loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded

    local_settings_path = _resolve_local_settings_path(settings_path)
    if local_settings_path.exists():
        loaded = json.loads(local_settings_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = _merge_dicts(payload, loaded)

    service_payload = _dict(payload.get("service"))
    grpc_payload = _dict(service_payload.get("grpc"))
    engine_payload = _dict(payload.get("engine"))
    models_payload = _dict(engine_payload.get("models"))
    fairness_value = engine_payload.get("fairness", {})
    if not isinstance(fairness_value, dict):
        raise ValueError("engine.fairness must be an object")
    fairness_payload = fairness_value

    models: dict[str, ModelSettings] = {}
    for model_name, model_payload_raw in models_payload.items():
        model_payload = _dict(model_payload_raw)
        backend_value = model_payload.get("backend")
        backend = None
        if backend_value is not None:
            parsed_backend = str(backend_value).strip().lower()
            if parsed_backend:
                backend = parsed_backend
        models[str(model_name)] = ModelSettings(
            backend=backend,
            model_path=str(model_payload.get("model_path", "") or "").strip(),
            enabled=bool(model_payload.get("enabled", True)),
            target_inflight=max(1, int(model_payload.get("target_inflight", 1))),
            device=_optional_str(model_payload.get("device")),
            voice_presets=_str_dict(model_payload.get("voice_presets")),
            kokoro_speed=max(0.25, float(model_payload.get("kokoro_speed", 1.0))),
            kokoro_warmup_enabled=bool(model_payload.get("kokoro_warmup_enabled", False)),
            kokoro_warmup_languages=_str_tuple(model_payload.get("kokoro_warmup_languages")),
            voxcpm2_model_id=str(model_payload.get("voxcpm2_model_id", VOXCPM2_DEFAULT_MODEL_ID) or "").strip()
            or VOXCPM2_DEFAULT_MODEL_ID,
            voxcpm2_load_denoiser=bool(model_payload.get("voxcpm2_load_denoiser", False)),
            voxcpm2_optimize=bool(model_payload.get("voxcpm2_optimize", False)),
            voxcpm2_cfg_value=max(0.1, float(model_payload.get("voxcpm2_cfg_value", 2.0))),
            voxcpm2_inference_timesteps=max(1, int(model_payload.get("voxcpm2_inference_timesteps", 10))),
            voxcpm2_normalize=bool(model_payload.get("voxcpm2_normalize", False)),
            voxcpm2_denoise=bool(model_payload.get("voxcpm2_denoise", False)),
            voxcpm2_reference_max_duration_s=_bounded_float(
                model_payload.get("voxcpm2_reference_max_duration_s", 8.0),
                minimum=1.0,
                maximum=60.0,
            ),
            voxcpm2_warmup_enabled=bool(model_payload.get("voxcpm2_warmup_enabled", False)),
            voxcpm2_warmup_cases=_voxcpm2_warmup_cases(model_payload.get("voxcpm2_warmup_cases")),
            nanovllm_model_id=str(model_payload.get("nanovllm_model_id", VOXCPM2_DEFAULT_MODEL_ID) or "").strip()
            or VOXCPM2_DEFAULT_MODEL_ID,
            nanovllm_devices=_int_tuple(model_payload.get("nanovllm_devices"), default=(0,)),
            nanovllm_max_num_seqs=max(1, int(model_payload.get("nanovllm_max_num_seqs", 4))),
            nanovllm_max_num_batched_tokens=max(1, int(model_payload.get("nanovllm_max_num_batched_tokens", 4096))),
            nanovllm_max_model_len=max(1, int(model_payload.get("nanovllm_max_model_len", 2048))),
            nanovllm_gpu_memory_utilization=_bounded_float(
                model_payload.get("nanovllm_gpu_memory_utilization", 0.85),
                minimum=0.1,
                maximum=1.0,
            ),
            nanovllm_inference_timesteps=max(1, int(model_payload.get("nanovllm_inference_timesteps", 10))),
            nanovllm_max_generate_length=max(1, int(model_payload.get("nanovllm_max_generate_length", 256))),
            nanovllm_temperature=_bounded_float(model_payload.get("nanovllm_temperature", 1.0), minimum=0.01, maximum=5.0),
            nanovllm_cfg_value=max(0.1, float(model_payload.get("nanovllm_cfg_value", 2.0))),
            nanovllm_reference_max_duration_s=_bounded_float(
                model_payload.get("nanovllm_reference_max_duration_s", 8.0),
                minimum=1.0,
                maximum=60.0,
            ),
            nanovllm_warmup_enabled=bool(model_payload.get("nanovllm_warmup_enabled", False)),
            nanovllm_warmup_cases=_nanovllm_warmup_cases(model_payload.get("nanovllm_warmup_cases")),
        )

    service_port = _port(service_payload.get("port", 8020), "service.port")
    grpc_port = _port(grpc_payload.get("port", 8021), "service.grpc.port")
    grpc_enabled = bool(grpc_payload.get("enabled", False))
    if grpc_enabled and grpc_port == service_port:
        raise ValueError("service.grpc.port must differ from service.port")

    return AppSettings(
        service=ServiceSettings(
            host=_nonempty_str(service_payload.get("host", "127.0.0.1"), "service.host"),
            port=service_port,
            log_level=_nonempty_str(service_payload.get("log_level", "info"), "service.log_level"),
            grpc=GrpcSettings(
                enabled=grpc_enabled,
                host=_nonempty_str(grpc_payload.get("host", "127.0.0.1"), "service.grpc.host"),
                port=grpc_port,
                max_receive_message_bytes=_positive_int(
                    grpc_payload.get("max_receive_message_bytes", 16_777_216),
                    "service.grpc.max_receive_message_bytes",
                ),
                max_send_message_bytes=_positive_int(
                    grpc_payload.get("max_send_message_bytes", 16_777_216),
                    "service.grpc.max_send_message_bytes",
                ),
                stream_buffer_chunks=_positive_int(
                    grpc_payload.get("stream_buffer_chunks", 4),
                    "service.grpc.stream_buffer_chunks",
                ),
                stream_buffer_bytes=_positive_int(
                    grpc_payload.get("stream_buffer_bytes", 1_048_576),
                    "service.grpc.stream_buffer_bytes",
                ),
                stalled_consumer_timeout_s=_positive_float(
                    grpc_payload.get("stalled_consumer_timeout_s", 2.0),
                    "service.grpc.stalled_consumer_timeout_s",
                ),
                shutdown_grace_s=_positive_float(
                    grpc_payload.get("shutdown_grace_s", 5.0),
                    "service.grpc.shutdown_grace_s",
                ),
            ),
        ),
        engine=EngineSettings(
            backend=str(engine_payload.get("backend", "stub")).strip().lower() or "stub",
            models=models,
            fairness=FairnessSettings(
                default_weight=_positive_float(
                    fairness_payload.get("default_weight", 1.0),
                    "engine.fairness.default_weight",
                ),
                weights=_fairness_weights(fairness_payload.get("weights", {})),
                soft_max_inflight_per_key=_positive_int(
                    fairness_payload.get("soft_max_inflight_per_key", 1),
                    "engine.fairness.soft_max_inflight_per_key",
                ),
                max_pending_per_key=_positive_int(
                    fairness_payload.get("max_pending_per_key", 4),
                    "engine.fairness.max_pending_per_key",
                ),
                max_pending_per_executor=_positive_int(
                    fairness_payload.get("max_pending_per_executor", 8),
                    "engine.fairness.max_pending_per_executor",
                ),
                idle_state_ttl_s=_positive_float(
                    fairness_payload.get("idle_state_ttl_s", 300.0),
                    "engine.fairness.idle_state_ttl_s",
                ),
            ),
        ),
    )


def _resolve_settings_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env_value = os.environ.get("TTS_POOL_SETTINGS_PATH", "").strip()
    if env_value:
        return Path(env_value)
    return DEFAULT_SETTINGS_PATH


def _nonempty_str(value: object, name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    return normalized


def _port(value: object, name: str) -> int:
    parsed = _positive_int(value, name)
    if parsed > 65_535:
        raise ValueError(f"{name} must be at most 65535")
    return parsed


def _resolve_local_settings_path(settings_path: Path) -> Path:
    env_value = os.environ.get("TTS_POOL_LOCAL_SETTINGS_PATH", "").strip()
    if env_value:
        return Path(env_value)
    if settings_path == DEFAULT_SETTINGS_PATH:
        return DEFAULT_LOCAL_SETTINGS_PATH
    return settings_path.with_name("local.json")


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _str_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if str(key).strip() and str(item).strip()}


def _str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        parsed = value.strip()
        return (parsed,) if parsed else ()
    if not isinstance(value, list):
        return ()
    parsed: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        parsed.append(text)
    return tuple(parsed)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None


def _positive_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number greater than 0") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field_name} must be a finite number greater than 0")
    return parsed


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer greater than 0")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer greater than 0") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be an integer greater than 0")
    if parsed <= 0:
        raise ValueError(f"{field_name} must be an integer greater than 0")
    return parsed


def _fairness_weights(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("engine.fairness.weights must be an object of {fairness_key: weight}")
    parsed: dict[str, float] = {}
    for raw_key, raw_weight in value.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("engine.fairness.weights keys must not be blank")
        if len(key) > 128:
            raise ValueError("engine.fairness.weights keys must be at most 128 characters")
        if key in parsed:
            raise ValueError(f"duplicate normalized fairness weight key: {key}")
        parsed[key] = _positive_float(
            raw_weight,
            f"engine.fairness.weights['{key}']",
        )
    return parsed


def _int_tuple(value: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if not isinstance(value, list):
        return default
    parsed: list[int] = []
    for item in value:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(parsed) or default


def _bounded_float(value: Any, *, minimum: float, maximum: float) -> float:
    parsed = float(value)
    return max(minimum, min(maximum, parsed))


def _voxcpm2_warmup_cases(value: Any) -> tuple[VoxCPM2WarmupCase, ...]:
    if not isinstance(value, list):
        return ()
    cases: list[VoxCPM2WarmupCase] = []
    for item in value:
        payload = _dict(item)
        if not payload:
            continue
        cfg_value = payload.get("cfg_value")
        inference_timesteps = payload.get("inference_timesteps")
        cases.append(
            VoxCPM2WarmupCase(
                name=str(payload.get("name", "") or "").strip(),
                text=str(payload.get("text", "Let's see if this works.") or "").strip()
                or "Let's see if this works.",
                reference_audio=bool(payload.get("reference_audio", False)),
                reference_duration_s=_bounded_float(payload.get("reference_duration_s", 4.0), minimum=0.2, maximum=60.0),
                cfg_value=None if cfg_value is None else max(0.1, float(cfg_value)),
                inference_timesteps=None if inference_timesteps is None else max(1, int(inference_timesteps)),
            )
        )
    return tuple(cases)


def _nanovllm_warmup_cases(value: Any) -> tuple[NanoVLLMWarmupCase, ...]:
    if not isinstance(value, list):
        return ()
    cases: list[NanoVLLMWarmupCase] = []
    for item in value:
        payload = _dict(item)
        if not payload:
            continue
        cfg_value = payload.get("cfg_value")
        temperature = payload.get("temperature")
        max_generate_length = payload.get("max_generate_length")
        cases.append(
            NanoVLLMWarmupCase(
                name=str(payload.get("name", "") or "").strip(),
                text=str(payload.get("text", "Let's see if this works.") or "").strip()
                or "Let's see if this works.",
                reference_audio=bool(payload.get("reference_audio", False)),
                reference_duration_s=_bounded_float(payload.get("reference_duration_s", 2.0), minimum=0.2, maximum=60.0),
                cfg_value=None if cfg_value is None else max(0.1, float(cfg_value)),
                temperature=None if temperature is None else _bounded_float(temperature, minimum=0.01, maximum=5.0),
                max_generate_length=None if max_generate_length is None else max(1, int(max_generate_length)),
            )
        )
    return tuple(cases)
