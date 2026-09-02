from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


MAX_REFERENCE_AUDIO_BYTES = 12_582_912


class ReferenceAudio(BaseModel):
    mime_type: str = "audio/wav"
    data: bytes = Field(max_length=MAX_REFERENCE_AUDIO_BYTES)
    max_duration_s: float | None = Field(default=None, gt=0.0, le=60.0)
    # Transcript of the reference audio. When provided, VoxCPM-family
    # engines switch from reference-only to "ultimate cloning": the WAV
    # is sent as the prompt audio, paired with this transcript, so the
    # model anchors voice identity to a known (audio, text) pair before
    # generating the target text.
    prompt_text: str | None = None
    # Ultimate-cloning sub-mode: when prompt_text is set, also pass the
    # same WAV as the reference_wav (voice-isolation prefix). VoxCPM2
    # docs call this "optional, for better similarity" — default True
    # picks that recommendation. Set False for the UC1 prompt-only path
    # (continuation seed without the extra ref-isolation prefix), which
    # is a few percent cheaper and sometimes preferred by advanced
    # users tuning by ear.
    also_use_as_reference: bool = True

    def decoded_bytes(self) -> bytes:
        return self.data


class VoiceSpec(BaseModel):
    preset: str | None = None
    instructions: str | None = None
    reference_audio: ReferenceAudio | None = None


class KokoroGenerationParams(BaseModel):
    speed: float | None = Field(default=None, ge=0.25, le=4.0)


class VoxCPM2GenerationParams(BaseModel):
    cfg_value: float | None = Field(default=None, ge=0.1, le=10.0)
    inference_timesteps: int | None = Field(default=None, ge=1, le=200)
    normalize: bool | None = None
    denoise: bool | None = None


class NanoVLLMVoxCPMGenerationParams(BaseModel):
    cfg_value: float | None = Field(default=None, ge=0.1, le=10.0)
    temperature: float | None = Field(default=None, ge=0.01, le=5.0)
    max_generate_length: int | None = Field(default=None, ge=1, le=4096)


class GenerationParams(BaseModel):
    kokoro: KokoroGenerationParams = Field(default_factory=KokoroGenerationParams)
    voxcpm2: VoxCPM2GenerationParams = Field(default_factory=VoxCPM2GenerationParams)
    nanovllm_voxcpm: NanoVLLMVoxCPMGenerationParams = Field(default_factory=NanoVLLMVoxCPMGenerationParams)


class ResponseRequest(BaseModel):
    model: str
    input: str
    language: str
    fairness_key: str | None = Field(default=None, max_length=128)
    voice: VoiceSpec = Field(default_factory=VoiceSpec)
    generation: GenerationParams = Field(default_factory=GenerationParams)

    @field_validator("model", "language", mode="before")
    @classmethod
    def _normalize_required_label(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("input", mode="before")
    @classmethod
    def _validate_input(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("input must be a string")
        if not value.strip():
            raise ValueError("input must not be blank")
        return value

    @field_validator("fairness_key", mode="before")
    @classmethod
    def _normalize_fairness_key(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("fairness_key must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("fairness_key must not be blank")
        return normalized


class AdminLoadRequest(BaseModel):
    target_inflight: int | None = Field(default=None, ge=1)


class AdminFairnessKeyEntry(BaseModel):
    fairness_key: str | None = None
    pending: int = Field(default=0, ge=0)
    active: int = Field(default=0, ge=0)
    weight: float = Field(default=1.0, gt=0.0)
    score: float = Field(default=0.0, ge=0.0)
    rejected_per_key_limit: int = Field(default=0, ge=0)
    rejected_executor_limit: int = Field(default=0, ge=0)


class AdminFairnessSnapshot(BaseModel):
    rejected_per_key_limit: int = Field(default=0, ge=0)
    rejected_executor_limit: int = Field(default=0, ge=0)
    keys: list[AdminFairnessKeyEntry] = Field(default_factory=list)


class AdminModelEntry(BaseModel):
    name: str
    resolved_backend: str
    configured_enabled: bool
    runtime_state: Literal["unloaded", "loading", "loaded", "unloading", "failed"]
    is_loaded: bool
    inflight_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    runtime_inflight: int = Field(default=0, ge=0)
    configured_target_inflight: int = Field(default=1, ge=1)
    effective_target_inflight: int = Field(default=1, ge=1)
    accepting_new_requests: bool = False
    fairness: AdminFairnessSnapshot = Field(default_factory=AdminFairnessSnapshot)
    last_error: str | None = None
    vram_estimate_mib: int | None = Field(default=None, ge=0)
    vram_estimate_source: Literal["observed_load_delta", "model_artifact_size", "unavailable"] = "unavailable"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    definition: dict[str, Any] = Field(default_factory=dict)


class AdminModelsEnvelope(BaseModel):
    models: list[AdminModelEntry] = Field(default_factory=list)


class AdminGpuMemoryDevice(BaseModel):
    index: int
    name: str
    used_mib: int = Field(ge=0)
    total_mib: int = Field(ge=0)
    used_over_total: str


class AdminGpuMemoryModelEstimate(BaseModel):
    name: str
    runtime_state: Literal["unloaded", "loading", "loaded", "unloading", "failed"]
    is_loaded: bool
    vram_estimate_mib: int | None = Field(default=None, ge=0)
    vram_estimate_source: Literal["observed_load_delta", "model_artifact_size", "unavailable"] = "unavailable"


class AdminGpuMemoryEnvelope(BaseModel):
    gpus: list[AdminGpuMemoryDevice] = Field(default_factory=list)
    models: list[AdminGpuMemoryModelEstimate] = Field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class EngineResult:
    audio: bytes
    mime_type: str = "audio/wav"
    sample_rate_hz: int | None = None
    duration_ms: int | None = None
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
