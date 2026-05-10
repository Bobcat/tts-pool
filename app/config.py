from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.json"
DEFAULT_LOCAL_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "local.json"
VOXCPM2_DEFAULT_MODEL_ID = "openbmb/VoxCPM2"


@dataclass(frozen=True)
class ServiceSettings:
    host: str = "127.0.0.1"
    port: int = 8020
    log_level: str = "info"


@dataclass(frozen=True)
class ModelSettings:
    backend: str | None = None
    model_path: str = ""
    enabled: bool = True
    target_inflight: int = 1
    device: str | None = None
    voice_presets: dict[str, str] = field(default_factory=dict)
    kokoro_speed: float = 1.0
    voxcpm2_model_id: str = VOXCPM2_DEFAULT_MODEL_ID
    voxcpm2_load_denoiser: bool = False
    voxcpm2_optimize: bool = False
    voxcpm2_cfg_value: float = 2.0
    voxcpm2_inference_timesteps: int = 10
    voxcpm2_normalize: bool = False
    voxcpm2_denoise: bool = False
    voxcpm2_reference_max_duration_s: float = 8.0


@dataclass(frozen=True)
class EngineSettings:
    backend: str = "stub"
    models: dict[str, ModelSettings] = field(default_factory=dict)


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
    engine_payload = _dict(payload.get("engine"))
    models_payload = _dict(engine_payload.get("models"))

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
        )

    return AppSettings(
        service=ServiceSettings(
            host=str(service_payload.get("host", "127.0.0.1")),
            port=int(service_payload.get("port", 8020)),
            log_level=str(service_payload.get("log_level", "info")),
        ),
        engine=EngineSettings(
            backend=str(engine_payload.get("backend", "stub")).strip().lower() or "stub",
            models=models,
        ),
    )


def _resolve_settings_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env_value = os.environ.get("TTS_POOL_SETTINGS_PATH", "").strip()
    if env_value:
        return Path(env_value)
    return DEFAULT_SETTINGS_PATH


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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None


def _bounded_float(value: Any, *, minimum: float, maximum: float) -> float:
    parsed = float(value)
    return max(minimum, min(maximum, parsed))
