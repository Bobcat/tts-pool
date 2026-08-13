# TTS-pool Low-Latency Streaming Architecture

Status: implementation contract. TTS synthesis uses `grpc.aio` exclusively.

## Goal

TTS-pool must serve clients that need audio before synthesis has completed.
The first usable NanoVLLM chunk should leave TTS-pool without waiting for a
complete WAV.

The refactor must preserve:

- the existing fair scheduler;
- one capacity boundary per loaded model;
- model load, unload, and inspection behavior;
- support for runtimes that only produce complete audio.

The new interface adds binary reference input, incremental PCM output,
backpressure, and cancellation.

Trusted clients synthesize exclusively through gRPC. `POST /v1/responses`, its
base64 audio envelope, and the HTTP `stream` field have been removed. HTTP
remains the control plane for model inspection and administration; it is not a
second synthesis transport.

## Measured Opportunity

The loaded NanoVLLM-VoxCPM runtime yields waveform chunks during generation.
The gRPC path forwards each chunk after its PCM conversion.

Warm measurements for about 10.24 seconds of generated audio were:

| Active sequences | First NanoVLLM chunk | Complete synthesis |
| ---: | ---: | ---: |
| 1 | 62.8-65.4 ms | 938.4-938.9 ms |
| 2 | 62.5-69.6 ms | 1,060.0-1,084.4 ms |
| 4 | 67.3-75.9 ms | 1,290.3-1,324.0 ms |

Streaming exposes roughly 0.9-1.25 seconds of audio-production time that was
hidden behind complete-response assembly. See
[Scheduler and NanoVLLM Load Test](tts-scheduler-load-test.md).

## Service Shape

One TTS-pool process owns:

- the loaded runtimes;
- the model executors;
- the fairness scheduler state;
- the FastAPI control-plane server;
- one `grpc.aio` server for low-latency synthesis.

Both network servers must use the same `TTSRouterEngine` instance. Starting a
separate process for gRPC would create a second scheduler and load another model
instance, so that is not the target architecture.

The FastAPI lifespan starts and stops the gRPC server. Uvicorn serves the HTTP
control plane. The gRPC server listens on its own configured port. Both ports
remain part of one systemd service.

```text
gRPC synthesis request ─> TTSRouterEngine ─> LoadedModelExecutor ─> runtime
                                                   │
                                                   └─> fair pending queue

HTTP model/admin request ─> TTSRouterEngine control plane
```

## Transport

Use one unary-request, server-streaming gRPC method:

```text
TTSService.Synthesize(SynthesisRequest) returns (stream SynthesisEvent)
```

`grpc.aio` supplies HTTP/2 multiplexing, per-stream flow control, deadlines,
cancellation, and binary messages. A client can send concurrent RPCs over one
long-lived channel.

The transport handler consumes synthesis events asynchronously. A queued or
active RPC must not reserve one thread merely to wait for its next event.

The transport measurements are recorded in
[TTS Streaming Transport Benchmark](tts-transport-benchmark.md).

## Protobuf Contract

The versioned schema lives under `proto/tts/v1/`. Generated Python modules are
checked in so service startup does not require a compiler.

### Request

`SynthesisRequest` contains:

- model name;
- target text;
- language;
- trusted `fairness_key`;
- voice preset or instructions;
- optional reference WAV as `bytes`;
- optional reference transcript;
- reference-mode fields already supported by the runtime;
- model-specific generation settings;
- requested output format.

Reference audio is binary protobuf data. It is not base64 text.

One decoded-byte cap applies to reference audio on every internal entry path.
The gRPC receive limit caps the complete protobuf message and may be larger
than that field limit. Increasing the transport limit must not increase the
allowed reference-audio size.

The first streaming version accepts one complete request message. Client-side
request streaming is not needed because reference WAVs are bounded and must be
available before reference encoding starts.

### Events

`SynthesisEvent` has one of three payloads:

1. `Started`
2. `AudioChunk`
3. `Completed`

`Started` contains:

- response ID;
- model name;
- sample rate;
- channel count;
- sample encoding;
- scheduler queue wait.

`AudioChunk` contains:

- monotonically increasing sequence number;
- first sample position;
- signed 16-bit little-endian PCM bytes.

The initial stream format is mono PCM at the runtime sample rate. Each event is
self-delimiting through protobuf and gRPC framing. TTS-pool does not need a WAV
header before it sends the first chunk.

`Completed` contains:

- total sample count;
- duration;
- chunk count;
- queue, reference, generation, and transport timings;
- bounded model metadata.

Exactly one `Started` event precedes audio. Exactly one `Completed` event ends a
successful stream. Failures terminate the RPC with a non-OK gRPC status rather
than an in-band error event. Stable error codes use trailing metadata. Internal
exceptions and tracebacks stay in service logs and are never returned to
clients.

## Error Mapping

| TTS-pool condition | gRPC status |
| --- | --- |
| Unknown model | `NOT_FOUND` |
| Model loading, unloading, or unavailable | `FAILED_PRECONDITION` |
| Invalid synthesis input | `INVALID_ARGUMENT` |
| Per-key or executor queue full | `RESOURCE_EXHAUSTED` |
| Deadline exceeded while queued or active | `DEADLINE_EXCEEDED` |
| Caller cancelled | `CANCELLED` |
| Runtime synthesis failure | `INTERNAL` |

Trailing metadata contains the existing stable machine-readable error code. It
does not contain audio or internal exception details.

## Shared Generation Core

Every network synthesis request produces the same internal event stream. Do
not maintain separate network-facing complete and streaming implementations.

```text
runtime chunk iterator
    └─> gRPC: encode each chunk as PCM and emit it
```

The shared stream yields typed internal events:

- runtime started;
- audio chunk;
- runtime completed.

Models with a native incremental iterator emit chunks as generation proceeds.
A runtime that only returns complete audio still uses the gRPC contract: the
executor converts its completed WAV to PCM and emits one `AudioChunk`, followed
by `Completed`. Its capability reports native streaming as unavailable, so a
client knows its first chunk cannot arrive before synthesis completes.

## NanoVLLM Runtime Refactor

`NanoVLLMVoxCPMTTSRuntime` processes values returned by `server.generate()` in
these steps:

1. validate and normalize the request;
2. prepare and encode reference audio;
3. start `server.generate()`;
4. convert and emit each returned waveform chunk;
5. finalize metrics and metadata.

Float waveform values are clipped to `[-1.0, 1.0]` and converted once to
signed 16-bit PCM.

The runtime keeps its dedicated event-loop thread. A bounded stream bridge
moves events from that loop to the executor consumer. The bridge must support a
thread-safe cancellation signal.

## Scheduler Integration

One synthesis request remains one scheduler job. Streaming does not create a
second admission path or bypass fairness.

The executor admits synthesis handles containing:

- a bounded event stream;
- terminal completion state;
- a cancellation method;
- scheduler and runtime timestamps.

The result-only future and HTTP-completion path are removed. The fair pending
queue continues to select jobs by normalized
`fairness_key`, virtual service, weight, active work, and activation order.

Service accounting starts when the executor dispatches the job. It ends only
when the runtime iterator exits. A slow or cancelled network consumer cannot
release a model slot while NanoVLLM is still using the sequence.

## Backpressure

Every synthesis handle has a bounded event buffer. Configure both a chunk limit
and a byte limit.

Each streaming runtime reports its maximum emitted PCM chunk size when loaded.
Model load fails if the configured byte buffer cannot hold one chunk. This is a
server configuration error, not consumer backpressure.

When the buffer is full:

1. runtime-to-transport forwarding waits for a short configured interval;
2. continued saturation marks the consumer as stalled;
3. the synthesis is cancelled;
4. the runtime slot remains charged until generation actually stops.

TTS-pool must never build an unbounded PCM queue. One stalled stream must not
consume memory or block transport progress for unrelated streams.

Do not apply gRPC compression to PCM chunks. PCM is not usefully compressed by
generic message compression, and compression adds latency and CPU work.

## Cancellation

Cancellation has two paths. It wakes both synchronous producer waits and async
transport consumers immediately.

### Queued job

Remove the job from its fairness bucket, complete its handle as cancelled, and
never dispatch it to the runtime.

### Active job

Set the handle's cancellation signal, stop forwarding output, and close the
NanoVLLM async generator. The scheduler charges the active service time up to
the point at which the generator exits.

An implementation test must prove that closing the generator releases the
active NanoVLLM sequence promptly. Until that test passes, TTS-pool may stop
delivery but must not claim that active cancellation saves GPU work.

Model unload and service shutdown use the same mechanism. Pending jobs cancel
first. Active streams receive a bounded grace period. The operation fails
cleanly when a runtime does not release within that bound; it must not wait
forever.

## HTTP Control Plane

HTTP exposes model listing, capabilities, load, unload, and inspection. It does
not accept synthesis input and does not return audio. The removed
`POST /v1/responses` route has no fallback or compatibility alias.

## Configuration

Add a `service.grpc` section:

```json
{
  "service": {
    "grpc": {
      "enabled": true,
      "host": "127.0.0.1",
      "port": 8021,
      "max_receive_message_bytes": 16777216,
      "max_send_message_bytes": 16777216,
      "stream_buffer_chunks": 4,
      "stream_buffer_bytes": 1048576,
      "stalled_consumer_timeout_s": 2.0,
      "shutdown_grace_s": 5.0
    }
  }
}
```

The tracked default binds gRPC to loopback. A deployment on a trusted private
network may widen the bind address in its local configuration. The gRPC API is
not authenticated; callers can supply `fairness_key`, so the port must not be
publicly reachable.

Startup rejects invalid ports and non-positive transport limits. Model load
rejects a buffer too small for one maximum-size runtime chunk. Reference-audio
validation remains independent of the total message limit.

## Observability

Use one response ID from admission through termination. Emit one structured
terminal log entry for a completed, failed, stalled, or cancelled request.
Counters distinguish gRPC status, queued versus active cancellation, stalled
consumers, and native versus complete-runtime synthesis.

Terminal timings include:

- scheduler queue wait;
- reference preparation and encoding time;
- backend time to first chunk;
- first-chunk transport handoff time;
- generation wall time;
- chunks and PCM bytes emitted;
- maximum buffered chunks and bytes;
- cancellation-to-runtime-release time;

Do not log raw `fairness_key` values or use them as metric labels. Hash them
only when per-principal correlation is needed.

## Delivery Order

### Phase 1: contract and server lifecycle

- add the protobuf schema and generated modules;
- add gRPC configuration;
- start and stop `grpc.aio` inside the FastAPI lifespan;
- add health and clean-shutdown tests.

### Phase 2: async delivery and generation core

- define internal stream events and the synthesis handle;
- make transport consumption awaitable without a thread per RPC;
- let the scheduler dispatch synthesis handles;
- adapt complete-only runtimes to one PCM event;
- preserve fairness.

### Phase 3: NanoVLLM chunks

- expose each `server.generate()` result immediately;
- convert each result to PCM once;
- emit terminal metrics after the iterator completes;
- verify that the streamed PCM reproduces the runtime waveform.

### Phase 4: backpressure and cancellation

- enforce bounded stream buffers;
- remove cancelled queued jobs;
- propagate active cancellation to NanoVLLM;
- measure sequence-release latency;
- cover stalled and disconnected consumers.

### Phase 5: client contract and HTTP removal

- publish the versioned contract and reproducible Python bindings;
- require clients to reuse long-lived gRPC channels;
- require trusted clients to send stable opaque fairness identities;
- remove `POST /v1/responses`, base64 audio schemas, and obsolete tests.

### Phase 6: live validation

- measure loopback and dc1-to-dc2 first-chunk latency;
- test active sequence counts 1, 2, and 4;
- repeat fair-progress and weighted-service checks;
- run a long-lived channel soak test;
- verify service restart and model unload behavior.

## Acceptance Criteria

The refactor is complete when:

- the first NanoVLLM chunk is emitted before full generation completes;
- one channel carries concurrent synthesis RPCs;
- every request passes through the existing fair scheduler;
- queued cancellation prevents runtime dispatch;
- active cancellation has measured sequence-release behavior;
- a stalled consumer cannot create unbounded buffering;
- a slow stream does not stop unrelated streams;
- HTTP model and admin endpoints remain green;
- model unload and process shutdown terminate every stream cleanly;
- live first-chunk metrics separate queue, reference, backend, and transport
  time.

## Recorded Live Validation

The first dc1-to-dc2 validation used the real NanoVLLM runtime over Wi-Fi:

- 24 sequential requests reused one gRPC channel with no failures;
- median first chunk was 69.8 ms and p95 was 72.8 ms;
- four overlapping requests on one channel made fair progress across two
  fairness keys at an effective runtime capacity of two;
- binary reference audio produced its first PCM chunk after 167 ms and
  completed after 410 ms;
- cancelling after the first chunk released client and runtime accounting in
  15.5 ms;
- controlled service shutdown completed an active RPC within the configured
  grace period.

These numbers describe that dc1/dc2 deployment, not a wired-network latency
guarantee. Automated transport, slow-consumer, and active-unload tests remain
part of the implementation gate.
