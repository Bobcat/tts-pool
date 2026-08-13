# tts-pool

Text-to-speech service with a streaming gRPC synthesis data plane and an HTTP
control plane. It supports Kokoro, VoxCPM2, and NanoVLLM-VoxCPM backends, with
runtime model administration, per-model scheduling, timing metrics, and GPU
memory inspection. The scheduler shares model capacity between clients and
limits how much work may wait in a queue.

## Index

- [Overview](#overview)
- [gRPC Synthesis](#grpc-synthesis)
- [HTTP Control Plane](#http-control-plane)
- [Local Overrides](#local-overrides)
- [Client Fairness](#client-fairness)
- [Timing Metrics](#timing-metrics)
- [Deployment Notes](#deployment-notes)
- [Test](#test)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

- `tts.v1.TTSService/Synthesize` streams raw PCM audio for any loaded model id.
- Configured model ids can use `kokoro`, `voxcpm2`, or `nanovllm_voxcpm`.
- Streams contain ordered audio chunks followed by timing metrics and backend metadata.
- Admin endpoints expose model state, runtime load/unload, queue state, and GPU memory.
- Each loaded model has its own scheduler with weighted client fairness,
  configurable inflight capacity, and bounded waiting queues.

## gRPC Synthesis

The versioned contract lives in [`proto/tts/v1/tts.proto`](proto/tts/v1/tts.proto).
One `SynthesisRequest` produces a server stream with:

1. one `Started` event that defines the PCM format;
2. ordered `AudioChunk` events with PCM S16LE bytes;
3. one `Completed` event with final counts, metrics, and metadata.

NanoVLLM-VoxCPM emits chunks while inference is running. Complete-only engines
use the same RPC and stream their completed waveform in chunks. Reference audio
is sent as binary WAV data, not base64. Clients that need fair scheduling send
a stable opaque `fairness_key` for the same principal.

There is no HTTP synthesis route. See [docs/api.md](docs/api.md) for request
fields, error statuses, message limits, and binding guidance.

## HTTP Control Plane

| Endpoint | Purpose |
| --- | --- |
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
    "port": 8020,
    "grpc": {
      "enabled": true,
      "host": "127.0.0.1",
      "port": 8021
    }
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
- Additional Japanese and Chinese Kokoro language dependencies are available
  through `pip install -e '.[kokoro]'`.

## Client Fairness

Several services can send work to the same loaded TTS model. Without fairness,
one busy client could keep filling every available place while other clients
wait.

Each request may include a stable `fairness_key`, such as:

- `interactive-client`
- `batch-worker`
- `workbench`

Requests with the same key keep their arrival order. Requests without a key
share one anonymous queue. A key's name is only a label; words such as
`interactive` do not grant priority by themselves.

An active slot is one request currently running for a model. If a model has four
slots, fairness behaves like this:

- If only batch requests are waiting, they may use all four slots.
- If an interactive request arrives, running batch requests are not interrupted.
- The interactive request gets the next slot that becomes free.
- If no other key is waiting, batch work may continue to use every slot.

Fairness only compares clients while they need the same model at the same time.
A client may use all available capacity while nobody else is waiting. When
another client joins under its own key, the available capacity is shared
according to the configured weights.

The scheduler records how long requests from each key occupy runtime slots.
Configured weights control the long-term split when several queues stay busy. A
key with weight `2.0` gets about twice as much slot-time as a key with weight
`1.0`. Requests cannot choose their own weight.

Configure fairness under `engine.fairness`:

```json
{
  "engine": {
    "fairness": {
      "default_weight": 1.0,
      "weights": {
        "interactive-client": 1.0,
        "batch-worker": 1.0,
        "workbench": 1.0
      },
      "soft_max_inflight_per_key": 1,
      "max_pending_per_key": 4,
      "max_pending_per_executor": 8,
      "idle_state_ttl_s": 300
    }
  }
}
```

The settings mean:

| Setting | Meaning |
| --- | --- |
| `default_weight` | Weight for anonymous or unlisted keys. |
| `weights` | Configured weight for known keys. |
| `soft_max_inflight_per_key` | Preferred number of active requests per key while another key waits. It is not a hard limit. |
| `max_pending_per_key` | Hard limit on waiting requests for one key. |
| `max_pending_per_executor` | Hard limit on all waiting requests for one model. |
| `idle_state_ttl_s` | How long the scheduler remembers used slot-time after a key has no active or waiting requests. |

When two keys have weight `1.0` and both queues stay busy, they receive roughly
equal slot-time. A client can still use all slots when no other key is waiting.
This is weighted fair sharing, not strict priority or a hard per-key concurrency
limit.

Queue limits count waiting work, not active requests. Rejections use gRPC
`RESOURCE_EXHAUSTED`. The `tts-error-code` trailing metadata contains
`fairness_key_queue_full` or `executor_queue_full`. Binary reference audio is
limited to 12,582,912 bytes before a request can enter a queue.

`GET /v1/admin/models` shows active and waiting counts, configured weights, and
rejection counters per key. Each key receives a separate share, so trusted
callers should keep the set of keys small and stable. If untrusted callers can
reach the API, an authenticated gateway should assign or validate the key.

See [docs/scheduler-fairness.md](docs/scheduler-fairness.md) for the scheduling
policy and limitations. See
[docs/tts-scheduler-load-test.md](docs/tts-scheduler-load-test.md) for the live
NanoVLLM capacity and fairness measurements.

## Timing Metrics

The `Completed.metrics` payload uses nested timers:

- `backend_synthesis_wall_ms`
  total wall time spent inside the selected TTS runtime
- `engine_total_wall_ms`
  backend synthesis plus queueing, scheduling, and other engine work around it
The payload may also include runtime-specific counters and sub-timers:

- `engine_queue_wait_ms`
  time spent waiting in the per-model scheduler queue
- `engine_outside_backend_wall_ms`
  engine time not spent inside backend synthesis
- `grpc_first_chunk_wall_ms`
  time from RPC receipt until the first PCM chunk is handed to the transport
- `grpc_stream_wall_ms`
  time from RPC receipt until the completed stream event
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

The live NanoVLLM benchmark exercises concurrent synthesis and client fairness
through the gRPC data plane:

```bash
.venv/bin/python scripts/tts_pool_fairness_bench.py \
  --concurrencies 1,2 \
  --repeats 3 \
  --fairness-capacity 2 \
  --output /tmp/tts-pool-fairness-capacity2.json
```

The benchmark requires a running service with Kokoro and NanoVLLM-VoxCPM
loaded. It does not edit configuration or restart the service. See the
[load-test report](docs/tts-scheduler-load-test.md) for the 4/4 and weighted
test procedure.

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
