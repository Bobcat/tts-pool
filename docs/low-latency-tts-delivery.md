# Low-Latency TTS Delivery

Status: design proposal. The transport benchmark must select the streaming
mechanism before implementation starts.

## Goal

Make speech start as soon as possible after a user presses `Speak`.

The design has two complementary paths:

1. prepare likely speech before the click and buffer it in the browser;
2. stream a cache miss from NanoVLLM to the browser without waiting for a
   complete WAV.

Prepared audio should normally start from a local browser buffer. A cache miss
should start after the first usable model audio chunk plus transport and
playback buffering. The complete WAV remains useful for replay and caching, but
must not block first audio.

This is an end-to-end property. TTS-pool, the app backend, and the frontend each
own part of the latency path.

## Evidence

NanoVLLM-VoxCPM already produces waveform chunks during generation. The current
TTS-pool backend collects those chunks before returning a response.

The live scheduler test measured these warm results for output containing about
10.24 seconds of audio:

| Runtime load | First internal chunk | Complete backend request |
| --- | ---: | ---: |
| One request | 62.8-65.4 ms | 938.4-938.9 ms |
| Two requests | 62.5-69.6 ms | 1,060.0-1,084.4 ms |
| Four requests | 67.3-75.9 ms | 1,290.3-1,324.0 ms |

Reference encoding was 7.5-17.2 ms in the paired-reference tests. See
[Scheduler and NanoVLLM Load Test](tts-scheduler-load-test.md).

The backend therefore has usable audio roughly 0.9-1.25 seconds before the
current API returns the complete result. Streaming can expose that interval to
the caller. Pre-generation can remove the generation interval from the click
path entirely.

## Current Path

### Browser and app

Voice sessions use one WebSocket in both directions:

- the browser sends binary PCM microphone audio;
- the browser sends JSON controls such as `speak_now`, `speak_part`, and
  `tts_playback_complete`;
- the app pushes transcript, turn, status, and `tts_clip_ready` JSON events.

There is no frontend polling in the voice workflow. After `tts_clip_ready`, the
browser retrieves the complete WAV from an app HTTP artifact URL and queues it
for playback.

### App and TTS-pool

The app sends one synchronous HTTP `POST /v1/responses` per speech bubble. The
request is JSON. Optional reference audio is base64 text inside that JSON.

TTS-pool currently:

1. admits the request through the model executor scheduler;
2. waits for every NanoVLLM waveform chunk;
3. concatenates the chunks;
4. encodes a complete WAV;
5. base64-encodes the WAV in a JSON response.

The app decodes the response, writes the WAV to disk, pushes `tts_clip_ready`,
and lets the browser retrieve the artifact. The `stream` request field exists,
but `stream=true` is rejected.

## Target Flow

### Prepared bubble

```text
final translated bubble
    -> app submits TTS before Speak
    -> TTS-pool generates audio
    -> app assembles and caches the WAV
    -> app tells the browser that audio is prepared
    -> browser fetches and buffers the audio
    -> user presses Speak
    -> browser starts the local buffer
```

Generation and browser transfer happen before the user action. The click still
notifies the app so the server can close the ASR scope, update the turn state,
and track playback completion.

### Cache miss

```text
user presses Speak
    -> app starts or joins synthesis
    -> TTS-pool emits the first NanoVLLM chunk
    -> app forwards the chunk
    -> browser starts buffered PCM playback
    -> later chunks continue while audio plays
    -> app also assembles the complete WAV
```

No layer waits for a complete WAV before forwarding the first chunk.

### Several bubbles

Synthesis completion order does not determine playback order. The app and
frontend retain the order of the selected turn parts.

The first prepared bubble may start immediately while later bubbles are still
being generated or transferred. Each later bubble must be ready before the
previous bubble finishes to avoid an audible gap.

## Prepared-Audio Identity

Prepared audio is valid only for the exact synthesis input that produced it.
An app-side cache key must include:

- turn ID and part ID;
- exact target text;
- target language;
- selected TTS model;
- normalized TTS generation settings;
- voice mode and instructions;
- reference-audio identity or digest;
- reference prompt text when present.

The app stores a generation number with each part. A newer translation or TTS
configuration invalidates older work even if it completes later.

The frontend receives the same turn ID, part ID, and generation number with a
prepared-audio notification. It must remove a buffered artifact when a later
turn update invalidates that identity.

## Preparation Policy

The first implementation prepares closed bubbles whose target translation is
final. It does not prepare a changing preview.

The app controls preparation with configurable limits:

- maximum active preparations per principal;
- maximum queued preparations per principal;
- maximum active preparations for one app process;
- maximum prepared bytes and artifacts per voice session;
- artifact lifetime after a turn or session closes.

These are app resource controls. They are not fields that a browser may choose.

When the user presses `Speak`, the app follows this order:

1. use the exact prepared generation if the browser already has it;
2. join the exact in-flight preparation instead of submitting a duplicate;
3. start an on-demand stream on a cache miss.

Preparation failures stay silent in the UI. An explicit `Speak` action may
retry on demand. Overload must not start a preparation retry loop.

## Principal Fairness

The app derives one stable opaque `fairness_key` from its resolved principal
and stores it with the voice session. Every TTS request for that principal uses
the same key. Opening more browser or voice sessions must not create more
scheduler shares.

The browser cannot supply or override the key.

TTS-pool continues to apply its existing weighted least-served scheduler. One
principal may borrow unused runtime capacity. A contending principal receives
fair progress when a slot becomes free. The streaming result path does not
change scheduler selection or service accounting.

Fairness is local to the configured TTS-pool. Product credits, quotas, and app
request limits remain app concerns.

## App-to-Pool Streaming Contract

The production transport will expose one logical synthesis stream per request.
The app client presents one stable interface regardless of the selected wire
mechanism:

```python
async for event in tts_client.synthesize_stream(request):
    ...
```

One app process initially owns one long-lived connection or channel to its
configured TTS-pool. Concurrent synthesis requests use independent logical
streams on that connection. The synthesis API does not expose connection
selection to callers.

The request contains the existing synthesis data plus the trusted
`fairness_key`. Reference audio should use binary bytes in the streaming
transport rather than base64 text.

### Stream events

The logical response contains ordered events:

1. `started`;
2. zero or more `audio` events;
3. exactly one `completed` or `error` terminal event.

`started` contains:

- output encoding;
- sample rate;
- channel count;
- request and model identity required for correlation.

Each `audio` event contains:

- a monotonically increasing sequence number;
- binary PCM payload;
- enough sample-position information to detect a gap or duplicate.

The initial output format is mono signed 16-bit little-endian PCM at the model
sample rate. TTS-pool does not need a final WAV header before emitting PCM.

`completed` contains:

- final sample count and duration;
- pool, queue, backend, and first-chunk timings;
- model metadata already returned by the non-streaming response.

An error before `started` is a normal request error. An error after audio has
started terminates the stream and marks the partial audio unusable for replay.
The frontend may finish audio it already buffered, but must not treat it as a
complete artifact.

### `stream=false`

The non-streaming path remains supported. Both response modes use one backend
generation source:

```text
NanoVLLM chunk iterator
    -> stream=true: emit chunks as they arrive
    -> stream=false: collect chunks and return the current WAV envelope
```

This prevents output differences between the streaming and non-streaming
paths.

### Backpressure and cancellation

Writing an audio event must respect transport backpressure. TTS-pool must not
build an unbounded per-client chunk queue.

When the client closes or cancels a stream:

- queued work is removed before it enters the runtime;
- active output forwarding stops;
- TTS-pool asks the backend generation to stop;
- the scheduler accounts for runtime time already consumed;
- no completed result is reported.

The transport benchmark must verify whether cancelling the NanoVLLM async
generator releases its active sequence promptly. Until that is proven, active
cancellation cannot be claimed to save GPU work.

## Minimal TTS-Pool Change

The TTS-pool implementation is limited to the streaming data path:

- implement the selected transport;
- make `stream=true` return incremental audio;
- expose NanoVLLM chunks without first collecting the complete result;
- carry stream cancellation to queued and active work where supported;
- preserve the current scheduler, fairness policy, model configuration, and
  capacity boundary;
- keep `stream=false` behavior available from the shared generation source;
- add streaming latency and cancellation tests.

TTS-pool generates and transports audio. It does not own preparation policy,
browser buffering, playback order, app quotas, or product state.

## App Responsibilities

The app owns:

- resolving the principal and deriving the trusted fairness key;
- selecting its configured TTS-pool;
- limiting speculative work;
- starting preparation after a bubble becomes final;
- deduplicating an explicit request against in-flight preparation;
- invalidating stale generations;
- assembling streamed PCM into a replayable WAV;
- retaining prepared artifacts for a bounded lifetime;
- releasing parts in turn order;
- forwarding cache-miss audio to the browser;
- recording end-to-end latency metrics.

A voice session stays assigned to one configured TTS-pool. Pool selection is
outside the synthesis protocol.

## Frontend Responsibilities

The frontend owns:

- retrieving prepared artifacts before `Speak`;
- holding a bounded local audio buffer keyed by part identity and generation;
- discarding buffered audio after invalidation or session cleanup;
- starting prepared playback directly from the user gesture;
- receiving and buffering streamed cache-miss PCM;
- preserving part order;
- reporting playback completion to the app;
- handling browser autoplay restrictions without regenerating audio.

Prepared complete artifacts can continue to use an app HTTP artifact URL. They
are fetched before the click, so this transfer is outside the critical path.

For cache misses, the existing voice WebSocket carries stream control and
binary audio frames from the app to the browser. Each binary frame identifies
the clip and sequence. The browser feeds PCM to an `AudioWorklet` or equivalent
streaming playback buffer. The app keeps transcript and turn events as JSON.

## Failure Semantics

| Condition | Required behavior |
| --- | --- |
| Prepared artifact is ready | Play the exact buffered generation |
| Preparation is still running | Join it; do not submit duplicate work |
| Preparation failed | Generate on demand after `Speak` |
| Translation or settings changed | Invalidate older app and browser entries |
| User changes turn or lane | Stop playback and discard stale entries |
| Browser disconnects | Stop forwarding and release session buffers |
| Pool rejects preparation | Keep the UI usable; do not retry in a loop |
| Pool rejects explicit speech | Show a concise playback error |
| Stream fails after partial audio | Do not cache the partial clip for replay |
| Playback is blocked by the browser | Keep prepared audio and show the resume control |

## Transport Selection Benchmark

Benchmark two small streaming prototypes before fixing the production wire
contract:

1. binary streaming HTTP using the service's Python async stack;
2. `grpc.aio` with one request and a server-streamed event response.

Both prototypes must consume the same synthetic or NanoVLLM chunk iterator and
send the same event information. The benchmark compares transport behavior,
not model output.

Run tests on loopback and on the real app-to-pool network route. Cover:

- cold and warm connections;
- with and without reference audio;
- active NanoVLLM concurrency 1, 2, and 4;
- synthetic concurrent streams at 10, 100, 500, and 1,000;
- one slow consumer beside normal consumers;
- cancellation while queued and while active;
- a disconnected client;
- simultaneous preparation and on-demand streams.

Measure:

- request start to backend start;
- backend first chunk to server transport write;
- backend first chunk to client receipt;
- client receipt to first playable browser buffer;
- p50, p95, and p99 time to first audio;
- inter-chunk gap and jitter;
- client-side waiting before a stream is opened;
- CPU time and memory per active stream;
- bytes transferred and number of copies;
- cancellation-to-slot-release time;
- effect of a slow stream on unrelated streams.

The selected transport must:

- expose the first chunk without waiting for completion;
- add no more than one network round trip plus 10 ms at p95 between backend
  first chunk and client receipt on the real route;
- keep unrelated streams progressing when one consumer is slow;
- propagate cancellation without leaked queue or stream state;
- support independent concurrent streams over one warm client connection;
- carry binary reference and output audio without base64;
- have bounded memory under the synthetic concurrency test.

When both candidates meet the limits, choose the smaller operational and code
surface.

## Observability

Use one correlation identity through browser, app, and TTS-pool. Record these
timestamps or durations:

```text
bubble_final
preparation_submitted
pool_enqueued
backend_started
backend_first_chunk
app_first_chunk_received
browser_first_chunk_received
speak_clicked
playback_started
generation_completed
playback_completed
```

Derived metrics include:

- preparation hit, in-flight join, or cache miss;
- click-to-playback-start;
- pool queue wait;
- backend time to first chunk;
- transport time for the first chunk;
- browser startup-buffer time;
- prepared bytes retained per session;
- stale preparations and completed stale work;
- audible gaps between parts;
- pool rejections and stream failures.

Do not use principal or fairness-key values as unbounded metric labels. Logs may
carry a bounded or hashed correlation value where needed.

## Delivery Phases

### Phase 1: transport decision

- add the two isolated streaming prototypes;
- run the benchmark matrix;
- record the results and select the production transport.

No app behavior changes in this phase.

### Phase 2: TTS-pool streaming

- implement the selected streaming contract;
- connect it to the NanoVLLM chunk iterator;
- implement `stream=true` and keep `stream=false` on the shared source;
- preserve scheduler and fairness behavior;
- verify output equivalence and cancellation cleanup.

### Phase 3: app preparation

- attach the resolved principal fairness key to voice sessions;
- prepare final closed bubbles with bounded concurrency;
- deduplicate and invalidate by exact generation identity;
- assemble and retain complete WAV artifacts;
- expose preparation state to the frontend.

At this point, prepared clicks can be fast even before live browser streaming is
enabled.

### Phase 4: frontend prefetch

- fetch prepared artifacts before the user action;
- retain bounded local buffers;
- play the exact local artifact on `Speak`;
- preserve the existing playback-complete lifecycle.

### Phase 5: cache-miss streaming

- forward TTS-pool PCM events over the voice WebSocket;
- add streaming browser playback;
- assemble the same PCM into a replayable app artifact;
- measure click-to-audio and inter-part gaps end to end.

### Phase 6: tuning

- tune preparation limits from measured contention and stale-work rates;
- tune browser startup buffering from measured jitter;
- test larger deployed model capacities without changing the protocol.

## Acceptance Criteria

The complete feature is ready when:

- a prepared bubble starts from the browser buffer without a network wait;
- a cache miss begins playback from incremental PCM before full generation
  completes;
- several selected bubbles play in their original order;
- stale audio never plays after text, settings, lane, turn, or account changes;
- retries and explicit speech do not duplicate an in-flight preparation;
- one principal cannot gain scheduler share by opening more sessions;
- a slow or disconnected browser does not create unbounded buffering;
- the existing non-streaming API still produces equivalent complete audio;
- latency metrics separate model, pool queue, transport, app, and browser time.
