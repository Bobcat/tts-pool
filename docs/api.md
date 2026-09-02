# TTS Pool APIs

TTS synthesis uses the versioned gRPC service in
`proto/tts/v1/tts.proto`. HTTP is a control plane only.

## Synthesis

`tts.v1.TTSService/Synthesize` accepts one `SynthesisRequest` and returns a
server stream of `SynthesisEvent` messages:

1. one `Started` event defines the PCM format;
2. one or more ordered `AudioChunk` events carry raw PCM S16LE bytes;
3. one `Completed` event reports final counts, metrics, and metadata.

NanoVLLM VoxCPM emits audio while inference is running. Complete-only runtimes
use the same RPC and emit their completed waveform as one or more PCM chunks.
There is no HTTP synthesis route or base64 audio envelope.

The request includes:

- model, text, and language;
- an optional opaque `fairness_key` with a maximum of 128 characters;
- optional voice instructions, preset, and binary WAV reference audio;
- backend-specific generation settings;
- PCM S16LE as the output encoding.

Reference audio is capped at 12,582,912 raw bytes before a request enters a
model queue. The gRPC server also applies independent total receive and send
message limits.

Trusted clients should reuse one stable, opaque fairness key for the same
principal. Requests without a key share one anonymous queue. The selected TTS
backend never receives the key.

Queue admission failures use gRPC `RESOURCE_EXHAUSTED`. The trailing metadata
field `tts-error-code` distinguishes `fairness_key_queue_full`,
`executor_queue_full`, and `stream_consumer_stalled`.

| Condition | gRPC status | `tts-error-code` |
| --- | --- | --- |
| Unknown model | `NOT_FOUND` | `unknown_model` |
| Model unavailable or changing state | `FAILED_PRECONDITION` | Model-state error code |
| Invalid request | `INVALID_ARGUMENT` | `invalid_request` |
| Queue limit or stalled consumer | `RESOURCE_EXHAUSTED` | Admission or stall code |
| Cancelled synthesis | `CANCELLED` | `synthesis_cancelled` |
| Internal synthesis failure | `INTERNAL` | `synthesis_failed` |

Internal exception text is logged by the service. The client receives the
sanitized text `TTS synthesis failed` for an internal failure.

Cancelling the RPC removes queued work or propagates cancellation to an active
runtime. Each stream has bounded chunk and byte buffers. A client that stops
consuming for longer than the configured timeout receives
`stream_consumer_stalled`.

## Metrics

The terminal `Completed.metrics` structure contains scheduler and runtime
timings. Common fields are:

- `engine_queue_wait_ms`: time waiting for fair scheduler admission;
- `backend_synthesis_wall_ms`: time occupying a runtime slot;
- `engine_total_wall_ms`: queue and synthesis time combined;
- `engine_outside_backend_wall_ms`: engine time outside backend synthesis;
- `grpc_first_chunk_wall_ms`: time from RPC receipt to the first PCM chunk;
- `grpc_stream_wall_ms`: time from RPC receipt to the completed stream event.

Runtime-specific fields include audio duration, realtime factor, reference
preparation, and model-generation timings. Complete-only runtimes preserve
their backend metrics when their WAV output is converted to streamed PCM.

## Health

The server registers the standard `grpc.health.v1.Health` service. Both the
empty service name and `tts.v1.TTSService` report `SERVING` while the gRPC
server is ready. They change to `NOT_SERVING` during controlled shutdown.

## HTTP Control Plane

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/models` | List loaded public model ids. |
| `GET /v1/admin/models` | List configured models, runtime state, capabilities, and scheduler state. |
| `GET /v1/admin/gpu-memory` | Return GPU memory usage and model artifact estimates. |
| `POST /v1/admin/models/{model}/load` | Load one configured model. |
| `POST /v1/admin/models/{model}/unload` | Stop admission, cancel work, and unload one model. |

`GET /v1/admin/models` exposes aggregate queue and active counts. Its
`fairness` object lists active or pending keys, configured weight, scheduler
score, and rejection counters. Keep this control plane on a trusted network.

## Binding And Generation

The tracked gRPC bind address is `127.0.0.1`. A deployment that serves remote
trusted clients must override it locally and protect the port at the network
boundary.

Regenerate the checked-in Python server bindings after editing the proto:

```bash
.venv/bin/python scripts/generate_grpc_stubs.py
```
