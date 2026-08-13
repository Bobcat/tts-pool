# TTS-pool Low-Latency Streaming Architecture

Status: implementation design. The transport is `grpc.aio`.

## Goal

TTS-pool must serve clients that need audio before synthesis has completed.
The first usable NanoVLLM chunk should leave TTS-pool without waiting for a
complete WAV.

The refactor must preserve:

- the existing fair scheduler;
- one capacity boundary per loaded model;
- the existing complete-response HTTP API;
- model load, unload, and inspection behavior;
- complete WAV output for non-streaming callers.

The new interface adds binary reference input, incremental PCM output,
backpressure, and cancellation.

## Measured Opportunity

The loaded NanoVLLM-VoxCPM runtime already yields waveform chunks during
generation. TTS-pool currently collects every chunk before it encodes and
returns a WAV.

Warm measurements for about 10.24 seconds of generated audio were:

| Active sequences | First NanoVLLM chunk | Complete synthesis |
| ---: | ---: | ---: |
| 1 | 62.8-65.4 ms | 938.4-938.9 ms |
| 2 | 62.5-69.6 ms | 1,060.0-1,084.4 ms |
| 4 | 67.3-75.9 ms | 1,290.3-1,324.0 ms |

Streaming can expose roughly 0.9-1.25 seconds of audio-production time that is
currently hidden behind complete-response assembly. See
[Scheduler and NanoVLLM Load Test](tts-scheduler-load-test.md).

## Service Shape

One TTS-pool process owns:

- the loaded runtimes;
- the model executors;
- the fairness scheduler state;
- the existing FastAPI HTTP server;
- one `grpc.aio` server for low-latency synthesis.

Both network servers must use the same `TTSRouterEngine` instance. Starting a
separate process for gRPC would create a second scheduler and load another model
instance, so that is not the target architecture.

The FastAPI lifespan starts and stops the gRPC server. Uvicorn continues to
serve the current HTTP API. The gRPC server listens on its own configured port.
Both ports remain part of one systemd service.

```text
HTTP complete request ─┐
                      ├─> TTSRouterEngine ─> LoadedModelExecutor ─> runtime
gRPC stream request ──┘                              │
                                                    └─> fair pending queue
```

## Transport

Use one unary-request, server-streaming gRPC method:

```text
TTSService.Synthesize(SynthesisRequest) returns (stream SynthesisEvent)
```

`grpc.aio` supplies HTTP/2 multiplexing, per-stream flow control, deadlines,
cancellation, and binary messages. A client can send concurrent RPCs over one
long-lived channel.

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
- scheduler queue wait;
- normalized generation identity needed for logging.

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
- the bounded model metadata returned by the complete API.

Exactly one `Started` event precedes audio. Exactly one `Completed` event ends a
successful stream. Failures terminate the RPC with a non-OK gRPC status rather
than an in-band error event.

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

The trailing status details contain the existing stable machine-readable error
code. They do not contain audio.

## Shared Generation Core

Streaming and complete responses must consume one runtime generation source.
Do not maintain separate synthesis implementations.

```text
runtime chunk iterator
    ├─> gRPC: encode each chunk as PCM and emit it
    └─> HTTP: collect the same PCM, add a WAV header, return the WAV
```

The shared stream yields typed internal events:

- runtime started;
- audio chunk;
- runtime completed.

The complete path becomes a stream consumer. This guarantees that streaming
and complete output use the same reference handling, generation parameters,
chunk conversion, and metadata.

Models without a native incremental iterator remain complete-response only.
Their advertised capabilities must report streaming as unavailable. The gRPC
method rejects them with `FAILED_PRECONDITION`; it does not emulate streaming
by waiting for a complete WAV.

## NanoVLLM Runtime Refactor

`NanoVLLMVoxCPMTTSRuntime` currently appends arrays returned by
`server.generate()` and concatenates them after generation. Split this path
into these steps:

1. validate and normalize the request;
2. prepare and encode reference audio;
3. start `server.generate()`;
4. convert and emit each returned waveform chunk;
5. finalize metrics and metadata.

Float waveform values are clipped to `[-1.0, 1.0]` and converted once to
signed 16-bit PCM. The complete path assembles those PCM bytes into its WAV.

The runtime keeps its dedicated event-loop thread. A bounded stream bridge
moves events from that loop to the executor consumer. The bridge must support a
thread-safe cancellation signal.

## Scheduler Integration

One synthesis request remains one scheduler job. Streaming does not create a
second admission path or bypass fairness.

The executor changes from a result-only future to a synthesis handle containing:

- a bounded event stream;
- terminal completion state;
- a cancellation method;
- scheduler and runtime timestamps.

The fair pending queue continues to select jobs by normalized
`fairness_key`, virtual service, weight, active work, and activation order.
HTTP and gRPC jobs compete in the same queue.

Service accounting starts when the executor dispatches the job. It ends only
when the runtime iterator exits. A slow or cancelled network consumer cannot
release a model slot while NanoVLLM is still using the sequence.

## Backpressure

Every synthesis handle has a bounded event buffer. Configure both a chunk limit
and a byte limit.

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

Cancellation has two paths.

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
first. Active streams receive a bounded grace period before forced shutdown.

## Complete HTTP API

`POST /v1/responses` remains the complete-response API. It returns the current
WAV envelope after consuming the shared generation stream.

Streaming belongs exclusively to gRPC. The HTTP `stream` request field is not
part of the target contract and should be removed when the gRPC contract lands.
TTS-pool should not expose two streaming protocols.

## Configuration

Add a `service.grpc` section:

```json
{
  "service": {
    "grpc": {
      "enabled": true,
      "host": "0.0.0.0",
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

The message limits must remain at least as strict as the current reference-audio
limit. Startup rejects invalid ports, non-positive limits, and a buffer too
small for one maximum-size runtime chunk.

The gRPC port is intended for a trusted private network. TLS can be added at
the transport boundary when traffic leaves that network; it is not required
for the initial dc1-to-dc2 deployment.

## Observability

Use one response ID from admission through the terminal event. Record:

```text
request_received
job_enqueued
job_dispatched
reference_started
reference_completed
backend_first_chunk
grpc_first_chunk_written
generation_completed
stream_completed
stream_cancelled
runtime_released
```

Derived metrics include:

- scheduler queue wait;
- reference preparation and encoding time;
- backend time to first chunk;
- first-chunk transport handoff time;
- generation wall time;
- chunks and PCM bytes emitted;
- maximum buffered chunks and bytes;
- stalled-consumer cancellations;
- queued and active cancellations;
- cancellation-to-runtime-release time;
- gRPC status counts;
- complete-response versus streaming request counts.

Do not use raw `fairness_key` values as metric labels. Existing bounded or
hashed log correlation rules continue to apply.

## Delivery Order

### Phase 1: contract and server lifecycle

- add the protobuf schema and generated modules;
- add gRPC configuration;
- start and stop `grpc.aio` inside the FastAPI lifespan;
- add health and clean-shutdown tests.

### Phase 2: shared generation core

- define internal stream events and the synthesis handle;
- let the scheduler dispatch streaming handles;
- make the complete HTTP path consume the same events;
- preserve fairness and complete WAV output.

### Phase 3: NanoVLLM chunks

- expose each `server.generate()` result immediately;
- convert each result to PCM once;
- emit terminal metrics after the iterator completes;
- verify streamed PCM and complete WAV equivalence.

### Phase 4: backpressure and cancellation

- enforce bounded stream buffers;
- remove cancelled queued jobs;
- propagate active cancellation to NanoVLLM;
- measure sequence-release latency;
- cover stalled and disconnected consumers.

### Phase 5: live validation

- measure loopback and dc1-to-dc2 first-chunk latency;
- test active sequence counts 1, 2, and 4;
- repeat fair-progress and weighted-service checks;
- run a long-lived channel soak test;
- verify service restart and model unload behavior.

## Acceptance Criteria

The refactor is complete when:

- the first NanoVLLM chunk is emitted before full generation completes;
- gRPC and HTTP output decode to the same PCM samples;
- one channel carries concurrent synthesis RPCs;
- every request passes through the existing fair scheduler;
- queued cancellation prevents runtime dispatch;
- active cancellation has measured sequence-release behavior;
- a stalled consumer cannot create unbounded buffering;
- a slow stream does not stop unrelated streams;
- complete HTTP behavior and admin endpoints remain green;
- model unload and process shutdown terminate every stream cleanly;
- live first-chunk metrics separate queue, reference, backend, and transport
  time.
