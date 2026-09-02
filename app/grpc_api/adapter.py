"""Conversion between the versioned protobuf contract and engine types."""

from __future__ import annotations

from app.schemas import GenerationParams
from app.schemas import KokoroGenerationParams
from app.schemas import NanoVLLMVoxCPMGenerationParams
from app.schemas import ReferenceAudio
from app.schemas import ResponseRequest
from app.schemas import VoiceSpec
from app.schemas import VoxCPM2GenerationParams

from .v1 import tts_pb2


def request_from_proto(message: tts_pb2.SynthesisRequest) -> ResponseRequest:
    if message.output_encoding not in {
        tts_pb2.AUDIO_ENCODING_UNSPECIFIED,
        tts_pb2.AUDIO_ENCODING_PCM_S16LE,
    }:
        raise ValueError("output_encoding must be PCM_S16LE")

    reference_audio: ReferenceAudio | None = None
    if message.voice.HasField("reference_audio"):
        source = message.voice.reference_audio
        reference_audio = ReferenceAudio(
            mime_type=source.mime_type or "audio/wav",
            data=source.data,
            max_duration_s=source.max_duration_s if source.HasField("max_duration_s") else None,
            prompt_text=source.prompt_text or None,
            also_use_as_reference=(
                source.also_use_as_reference
                if source.HasField("also_use_as_reference")
                else True
            ),
        )

    return ResponseRequest(
        model=message.model,
        input=message.input,
        language=message.language,
        fairness_key=message.fairness_key or None,
        voice=VoiceSpec(
            preset=message.voice.preset or None,
            instructions=message.voice.instructions or None,
            reference_audio=reference_audio,
        ),
        generation=GenerationParams(
            kokoro=KokoroGenerationParams(
                speed=_optional(message.generation.kokoro, "speed"),
            ),
            voxcpm2=VoxCPM2GenerationParams(
                cfg_value=_optional(message.generation.voxcpm2, "cfg_value"),
                inference_timesteps=_optional(message.generation.voxcpm2, "inference_timesteps"),
                normalize=_optional(message.generation.voxcpm2, "normalize"),
                denoise=_optional(message.generation.voxcpm2, "denoise"),
            ),
            nanovllm_voxcpm=NanoVLLMVoxCPMGenerationParams(
                cfg_value=_optional(message.generation.nanovllm_voxcpm, "cfg_value"),
                temperature=_optional(message.generation.nanovllm_voxcpm, "temperature"),
                max_generate_length=_optional(
                    message.generation.nanovllm_voxcpm,
                    "max_generate_length",
                ),
            ),
        ),
    )


def _optional(message: object, field_name: str) -> object | None:
    has_field = getattr(message, "HasField")
    if not has_field(field_name):
        return None
    return getattr(message, field_name)
