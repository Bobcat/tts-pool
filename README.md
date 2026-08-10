# tts-pool

FastAPI service for serving configured text-to-speech models through one
JSON-over-HTTP response API. It supports Kokoro, VoxCPM2, and
NanoVLLM-VoxCPM backends, with runtime model administration, per-model
scheduling, timing metrics, and GPU memory inspection.

## Index

- [Overview](#overview)
- [HTTP API](#http-api)
- [Synthesis Example](#synthesis-example)
- [Request Fields](#request-fields)
- [Voice And Reference Audio](#voice-and-reference-audio)
- [Local Overrides](#local-overrides)
- [Timing Metrics](#timing-metrics)
- [Deployment Notes](#deployment-notes)
- [Test](#test)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

- `POST /v1/responses` synthesizes text to WAV audio for any loaded model id.
- Configured model ids can use `kokoro`, `voxcpm2`, or `nanovllm_voxcpm`.
- Responses include base64 WAV audio, timing metrics, and backend metadata.
- Admin endpoints expose model state, runtime load/unload, queue state, and GPU memory.
- Each loaded model has its own scheduler with configurable inflight limits.

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/responses` | Synthesize text to WAV audio through a loaded TTS model. |
| `GET /v1/models` | List currently loaded model ids. |
| `GET /v1/admin/models` | List configured model ids plus runtime state, queue state, capabilities, and definitions. |
| `GET /v1/admin/gpu-memory` | Return current GPU memory usage plus per-model artifact estimates. |
| `POST /v1/admin/models/{model_name}/load` | Load one configured model at runtime. |
| `POST /v1/admin/models/{model_name}/unload` | Gracefully unload one loaded model. |

See [docs/api.md](docs/api.md) for shorter API notes.
See [docs/voxcpm2-optimization.md](docs/voxcpm2-optimization.md) for the current
VoxCPM2 optimization findings and warmup configuration.
See [docs/nanovllm-voxcpm-spike.md](docs/nanovllm-voxcpm-spike.md) for the
current NanoVLLM-VoxCPM notes.

## Synthesis Example

Example request:

```json
{
  "model": "voxcpm2",
  "input": "Let's see if this works.",
  "language": "English",
  "voice": {
    "instructions": "Speak in English. Use a clear, natural voice.",
    "reference_audio": {
      "mime_type": "audio/wav",
      "data_base64": "...",
      "max_duration_s": 4
    }
  },
  "format": {
    "type": "wav"
  },
  "generation": {
    "voxcpm2": {
      "cfg_value": 2.0,
      "inference_timesteps": 10,
      "normalize": false,
      "denoise": false
    }
  },
  "stream": false
}
```

Example response shape:

```json
{
  "id": "ttsresp_123",
  "object": "tts_response",
  "model": "voxcpm2",
  "audio": {
    "mime_type": "audio/wav",
    "data_base64": "...",
    "sample_rate_hz": 48000,
    "duration_ms": 1440
  },
  "metrics": {
    "engine_queue_wait_ms": 0.0,
    "backend_synthesis_wall_ms": 420.5,
    "engine_total_wall_ms": 421.1,
    "pool_total_wall_ms": 421.8,
    "voxcpm2_generate_wall_ms": 410.2,
    "output_audio_seconds": 1.44,
    "realtime_factor": 0.29
  },
  "metadata": {
    "engine": "voxcpm2",
    "device": "cuda",
    "reference_audio": true
  }
}
```

`stream: true` is intentionally rejected in the current version. The response
contains base64 WAV audio so callers can stay stateless and remote-friendly.

## Request Fields

Currently supported API request fields:

| Field | Type | Required | Default if omitted | Notes |
| --- | --- | --- | --- | --- |
| `model` | `string` | yes | none | Must match a currently loaded model id. |
| `input` | `string` | yes | none | Text to synthesize. |
| `language` | `string` | yes | none | Language label passed to the selected backend. |
| `voice` | `object` | no | `{}` | Optional backend-specific voice id, instructions, and reference audio. |
| `format.type` | `"wav"` | no | `"wav"` | WAV is the only supported output format. |
| `generation` | `object` | no | `{}` | Backend-specific generation overrides. |
| `stream` | `boolean` | no | `false` | `true` currently returns `400 stream_unsupported`. |

Kokoro generation fields:

| Field | Type | Notes |
| --- | --- | --- |
| `generation.kokoro.speed` | `float \| null` | Optional temporary speed override for the request. |

VoxCPM2 generation fields:

| Field | Type | Notes |
| --- | --- | --- |
| `generation.voxcpm2.cfg_value` | `float \| null` | Classifier-free guidance value. |
| `generation.voxcpm2.inference_timesteps` | `int \| null` | Diffusion sampling steps. |
| `generation.voxcpm2.normalize` | `boolean \| null` | Request-level text normalization toggle. |
| `generation.voxcpm2.denoise` | `boolean \| null` | Request-level denoise toggle for prompt/reference audio when denoiser support is loaded. |

NanoVLLM-VoxCPM generation fields:

| Field | Type | Notes |
| --- | --- | --- |
| `generation.nanovllm_voxcpm.cfg_value` | `float \| null` | Classifier-free guidance value. |
| `generation.nanovllm_voxcpm.temperature` | `float \| null` | Sampling temperature. |
| `generation.nanovllm_voxcpm.max_generate_length` | `int \| null` | Maximum generated token length. |

## Voice And Reference Audio

`voice.preset` is only for backends with native voice ids. Kokoro expects a
voice id such as `af_heart`.

VoxCPM2 and NanoVLLM-VoxCPM do not define service-level voice presets. Clients
own the prompt wording and send the final control text through
`voice.instructions`. Kokoro ignores `voice.instructions`.

VoxCPM2 and NanoVLLM-VoxCPM support `voice.reference_audio`:

```json
{
  "mime_type": "audio/wav",
  "data_base64": "...",
  "max_duration_s": 8
}
```

Without `prompt_text`, the service clips reference WAV audio to the
configured/requested maximum before passing it to the selected backend. When
`prompt_text` is present, the service keeps the complete WAV so the transcript
stays aligned with the audio. In that mode, `max_duration_s` does not limit the
reference audio. If a client wants the model to follow the reference pace or
articulation, that instruction belongs in `voice.instructions`.

## Local Overrides

Shared defaults live in `config/settings.json`. Machine-local overrides belong
in ignored `config/local.json`. When present, `local.json` is merged over
`settings.json`.

Settings files can also be selected explicitly:

- `TTS_POOL_SETTINGS_PATH`: base settings file path
- `TTS_POOL_LOCAL_SETTINGS_PATH`: local override file path

Example local override:

```json
{
  "service": {
    "host": "127.0.0.1",
    "port": 8020
  },
  "engine": {
    "models": {
      "kokoro": {
        "model_path": "/path/to/kokoro",
        "enabled": false
      },
      "voxcpm2": {
        "enabled": true,
        "target_inflight": 1,
        "voxcpm2_model_id": "openbmb/VoxCPM2",
        "voxcpm2_optimize": true,
        "voxcpm2_reference_max_duration_s": 8.0,
        "voxcpm2_warmup_enabled": true
      },
      "nanovllm_voxcpm": {
        "enabled": false,
        "target_inflight": 2,
        "nanovllm_model_id": "openbmb/VoxCPM2",
        "nanovllm_max_num_seqs": 2,
        "nanovllm_max_num_batched_tokens": 2048,
        "nanovllm_max_model_len": 1024,
        "nanovllm_gpu_memory_utilization": 0.10,
        "nanovllm_warmup_enabled": true
      }
    }
  }
}
```

Notes:

- Models without a `backend` field use the global `engine.backend`.
- `enabled` controls whether a model is loaded at service startup.
- A configured model with `enabled: false` may still be loaded later through the
  admin API.
- `voxcpm2_warmup_enabled` runs a bounded default warmup suite after VoxCPM2
  loads. Custom `voxcpm2_warmup_cases` can be added in `local.json`.
- `nanovllm_warmup_enabled` runs a bounded request-shape warmup suite after
  NanoVLLM-VoxCPM loads. Custom `nanovllm_warmup_cases` can be added in
  `local.json`.
- `target_inflight` is configured per model id and applied through the scheduler.
- The base dependencies include the VoxCPM2 backend package.
- Kokoro dependencies are available through `pip install -e '.[kokoro]'`.

## Timing Metrics

The response `metrics` payload uses nested timers:

- `backend_synthesis_wall_ms`
  total wall time spent inside the selected TTS runtime
- `engine_total_wall_ms`
  backend synthesis plus queueing, scheduling, and other engine work around it
- `pool_total_wall_ms`
  total time spent inside the `tts-pool` request handler

The payload may also include runtime-specific counters and sub-timers:

- `engine_queue_wait_ms`
  time spent waiting in the per-model scheduler queue
- `engine_outside_backend_wall_ms`
  engine time not spent inside backend synthesis
- `input_chars`
  input text length
- `output_audio_seconds`
  generated audio duration
- `realtime_factor`
  synthesis wall time divided by output audio duration
- `voxcpm2_generate_wall_ms`
  VoxCPM2 model generation time
- `voxcpm2_wav_encode_ms`
  WAV encoding time after VoxCPM2 generation
- `nanovllm_reference_prepare_wall_ms`
  local reference WAV decode/clip/copy time before NanoVLLM encoding
- `nanovllm_reference_encode_wall_ms`
  NanoVLLM reference WAV latent encoding time
- `nanovllm_generate_wall_ms`
  NanoVLLM-VoxCPM async generation-loop time
- `nanovllm_first_chunk_wall_ms`
  time to first generated audio chunk, comparable to TTFT for streaming LLMs
- `nanovllm_wav_encode_ms`
  WAV encoding time after NanoVLLM generation
- `kokoro_pipeline_wall_ms`
  Kokoro pipeline consumption time

Some fields are backend-dependent and may be omitted.

## Deployment Notes

The service can be run directly:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8020
```

The `deploy/systemd` directory contains a user-service example. The checked-in
files assume this checkout layout:

```bash
~/projects/tts-pool
```

For a different layout, edit the unit and start script or provide a user-level
systemd drop-in that sets the working directory, settings path, host, port, and
virtualenv path.

Useful systemd commands after adapting the paths:

```bash
systemctl --user status tts-pool.service
journalctl --user -u tts-pool.service -f
systemctl --user restart tts-pool.service
```

## Test

```bash
python3 -m unittest discover -s tests
```

Additional checks used during development:

```bash
python3 -m py_compile app/main.py app/config.py app/schemas.py app/engine/common.py app/engine/router.py app/engine/scheduler.py app/engine/stub.py app/engine/kokoro.py app/engine/voxcpm2.py app/engine/nanovllm_voxcpm.py
python3 -m pip check
git diff --check
```

## Acknowledgments

This pool builds on a number of upstream projects:

- FastAPI
- Uvicorn
- Pydantic
- Kokoro
- VoxCPM2
- NanoVLLM-VoxCPM

## License

Apache License 2.0. See [LICENSE](LICENSE).
