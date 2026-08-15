# Code Review: Low-Latency Streaming Implementation

Review of the uncommitted implementation of
[Low-Latency TTS Delivery](low-latency-tts-delivery.md).

- Scope: all uncommitted modifications and all untracked files
  (`app/grpc_api/`, `proto/`, `app/engine/streaming.py`,
  `scripts/tts_transport_bench.py`, `tests/test_grpc_api.py`,
  `tests/test_streaming.py`). No code was changed during the review.
- Method: read the full diff against the design document, traced the
  concurrency paths by hand, ran the test suite.
- Test run: `python -m unittest discover -s tests` — 77 tests, all pass.

## Overall verdict

The implementation is solid and close to the design. The shared generation
core, the scheduler integration, the bounded stream buffer, and the
cancellation chain are implemented the way the design describes, and the new
concurrency primitives come with meaningful tests. Phases 1-4 of the delivery
order are essentially covered. The findings below are deviations and edge
cases, not structural problems. Several acceptance criteria (phase 5) remain
unproven and are listed at the end.

## What is done well

- **Shared generation core.** `synthesize()` and `synthesize_stream()` both
  run through `_generate_pcm_async` (app/engine/nanovllm_voxcpm.py:210); the
  complete HTTP path is a stream consumer that collects PCM chunks and adds a
  WAV header. The PCM conversion (clip to `[-1.0, 1.0]`, `* 32767.0`,
  truncating `<i2`) is elementwise-identical to the old whole-array
  `_wav_bytes` conversion, so per-chunk conversion is bit-equivalent to the
  previous output. `test_stream_emits_pcm_before_completed_event` asserts
  streamed PCM equals the complete WAV frames.
- **Scheduler integration.** One request remains one job; streaming jobs go
  through the same `_FairPendingQueue`, the same admission limits, and the
  same capacity boundary (`effective_target_inflight`). Queued-job removal
  (app/engine/scheduler.py:140) does not touch virtual service, so a cancelled
  queued job does not distort fairness. Service time is charged from dispatch
  until the runtime iterator exits — including for cancelled jobs — which is
  exactly what the design requires ("a slow or cancelled network consumer
  cannot release a model slot while NanoVLLM is still using the sequence").
- **Backpressure.** The buffer is bounded on both chunks and bytes, a stalled
  consumer is cancelled after `stalled_consumer_timeout_s`, the producer is
  woken immediately on cancel, and high-water marks
  (`stream_max_buffered_chunks`/`_bytes`) land in the terminal metrics. No
  unbounded PCM queue exists anywhere.
- **Cancellation chain.** Cancel is checked before dispatch
  (scheduler.py:422), before generation, after reference encoding, and between
  chunks; the NanoVLLM async generator is closed in a `finally`
  (nanovllm_voxcpm.py:297-300). `test_stream_cancellation_closes_runtime_generator`
  proves the close reaches the generator promptly (against a fake server).
- **Error mapping.** Matches the design table, including the stable
  machine-readable code in trailing metadata (`tts-error-code`), verified by
  `test_error_code_is_exposed_as_trailing_metadata`. `DEADLINE_EXCEEDED` has no
  explicit branch, but that is fine: gRPC transport cancellation delivers it
  to the client without server code.
- **Configuration.** Port collision, non-positive limits, blank hosts, and
  `nan` are all rejected at startup; gRPC is disabled by default; the config
  tests are thorough.
- **Lifecycle.** The gRPC server starts and stops inside the FastAPI lifespan
  and is stopped *before* `engine.close()`; a failed bind fails startup
  loudly. The health service is registered. No gRPC compression is configured,
  per the design.
- **Packaging.** `grpcio`, `grpcio-health-checking`, and `protobuf` pins match
  the installed versions and the generated-code headers (protobuf 7.35.1,
  grpcio 1.83.0); `app.grpc_api` and `app.grpc_api.v1` were added to
  `packages`.

## Findings

### Medium

**1. Binary reference audio bypasses the reference-size limit.**
HTTP caps `data_base64` at `MAX_REFERENCE_AUDIO_BASE64_CHARS` = 16,777,216
chars, i.e. ~12.6 MiB decoded (app/schemas.py:15,20). The gRPC path stores raw
bytes via `ReferenceAudio.from_bytes` (app/schemas.py:38-56,
app/grpc_api/adapter.py:26-36) with no per-field size check; the only bound is
`max_receive_message_bytes` (16 MiB for the whole message). Two consequences:
gRPC accepts roughly 30% larger reference audio than HTTP, and the design's
rule "the message limits must remain at least as strict as the current
reference-audio limit" is not validated anywhere — an operator can set a
64 MiB message limit and startup will not complain. Suggest enforcing a
decoded-byte cap in `from_bytes`/`decoded_bytes` and adding the startup
cross-check from the design's Configuration section.

**2. No startup rejection of a buffer smaller than one runtime chunk.**
The design explicitly requires startup to reject "a buffer too small for one
maximum-size runtime chunk". Today that misconfiguration only surfaces at
runtime, as `StreamConsumerStalled` (app/engine/streaming.py:140-141), which
`_status_for_error` maps to `RESOURCE_EXHAUSTED` toward the client
(app/grpc_api/service.py:150-151) — a server configuration error reported as a
client fault, and only after synthesis has started. The maximum chunk size is
model knowledge, so if a startup check is impractical, at minimum derive the
bound at model load time (sample rate is known there) or document the required
relationship between `stream_buffer_bytes` and the runtime chunk size.

**3. Observability is below the design spec.**
Implemented: one JSON log line per completed stream with a rich metrics dict
(app/grpc_api/service.py:66-78) and two cancellation log lines. The design's
`request_received → … → runtime_released` event sequence is not emitted, and
none of the listed counters exist: stalled-consumer cancellations, queued vs
active cancellations, cancellation-to-runtime-release latency, gRPC status
counts, and complete-vs-streaming request counts. Most raw values already
exist in the `Completed` metrics, so this is aggregation work, not
instrumentation from scratch.

### Low

**4. `read_event` does not wake on `cancel()`.**
The wait condition only observes `_events` and `_terminal`
(app/engine/streaming.py:189-202); `cancel()` sets `_cancelled` and notifies,
but a blocked reader re-checks the condition and goes back to sleep. Today
this is masked because termination always follows: the queued path fails the
handle via the cancel callback, and the active path ends when the runtime
raises `SynthesisCancelled`. Still, a client-cancelled gRPC handler leaves its
`asyncio.to_thread(handle.read_event)` thread parked until the runtime exits
(app/grpc_api/service.py:47), and any future runtime that does not honor
cancellation leaks that thread. Consider including `_cancelled` in the
`read_event` condition (raise `SynthesisCancelled`) or documenting that
cancellation only reaches readers indirectly.

**5. `Started` lacks the "normalized generation identity" from the design.**
The proto's `Started` carries response ID, model, format, and queue wait
(proto/tts/v1/tts.proto:63-70) but no generation identity for logging.
Generation parameters only appear in `Completed.metadata`. Either extend
`Started` or amend the design document.

**6. The HTTP `stream` field and the docs were not updated.**
The design says the HTTP `stream` field "should be removed when the gRPC
contract lands". It is still present (app/schemas.py:108) and still rejected
with a 400 (app/main.py:139-140 — pre-existing behavior). Keeping the explicit
rejection is defensible for API compatibility, but then
docs/api.md:44-48 ("`stream: true` is intentionally rejected in the first
version") is now stale, and the new gRPC API is not documented anywhere —
neither in docs/ nor in the README. The proto contract and an error-mapping
table deserve a public doc page.

**7. Proto regeneration is undocumented and fragile.**
`app/grpc_api/v1/tts_pb2_grpc.py` imports `from . import tts_pb2`, which plain
`grpc_tools.protoc` does not generate — a naive regeneration produces a
top-level `import tts_pb2` and breaks the package. There is no script or note
describing the generation command. Add one (the `dev` extra already carries
`grpcio-tools`).

**8. Terminal callbacks swallow all exceptions.**
`_run_terminal_callbacks` silently drops every exception
(app/engine/streaming.py:231-236). The router's inflight accounting runs in
one of these callbacks; a bug there would be invisible. Log at least.

**9. Model unload can block indefinitely on a stuck stream.**
`unload_model` waits for `inflight_requests` with no timeout
(app/engine/router.py:262-265 — a pre-existing pattern, now also relevant for
streams). `begin_shutdown` cancels active handles immediately rather than
granting the design's "bounded grace period", and if a runtime generator ever
fails to honor cancellation, unload (and therefore `close()`) hangs forever.
Bounded in practice by NanoVLLM's per-chunk cancel checks, but the bound is
convention, not mechanism.

### Nits

- Cancel race between enqueue and `set_cancel_callback`
  (app/engine/scheduler.py:331-337): a cancel in that window leaves the job
  queued until dispatch. The pre-dispatch cancelled check still prevents any
  runtime work, so correctness holds; only queue-depth accounting lingers. A
  comment would help.
- `StreamConsumerStalled` names two different root causes: one oversized chunk
  (streaming.py:141) and an undrained buffer (streaming.py:151). Same
  client-facing code, different operator action.
- `scripts/tts_transport_bench.py` needs `h2`, which is in no dependency
  extra; the script does print an install hint.
- `first_chunk_written_at` is stamped when the generator resumes after the
  `yield` (app/grpc_api/service.py:62-64) — a reasonable approximation of
  "written to transport", worth knowing when reading
  `grpc_first_chunk_wall_ms`.

## Test coverage vs the acceptance criteria

Covered by unit tests: PCM/WAV equivalence, queued cancellation preventing
runtime dispatch, active cancellation releasing the runtime slot and closing
the generator (fake server), bounded buffers and stalled-consumer cancel,
error-code mapping, health and lifespan start/stop, config validation,
existing fairness and HTTP behavior (all pre-existing tests still pass).

Not yet covered, per the design's own acceptance list:

- **Concurrent RPCs over one channel** — no test exercises two overlapping
  streams on one channel.
- **A slow stream does not stop unrelated streams** — no test runs a stalled
  and a healthy stream through one executor.
- **Measured sequence-release behavior** — the design gates any "active
  cancellation saves GPU work" claim on a measurement; the current proof is
  against a fake server. Keep that claim out of docs until the live
  measurement exists.
- **Unload/shutdown with an active stream** — covered by construction
  (`begin_shutdown` cancels active handles) but not by a dedicated test.
- **Phase 5 live validation** — the checked-in
  `scripts/tts_transport_bench.py` is a synthetic raw-HTTP/2-vs-gRPC transport
  comparison, not an end-to-end TTS streaming benchmark; the loopback and
  dc1-to-dc2 first-chunk measurements remain open.

## Suggested follow-ups (in order)

1. Enforce the decoded-size cap for binary reference audio and the startup
   cross-check for message limits (finding 1).
2. Add the buffer-vs-chunk-size validation or document the relationship
   (finding 2).
3. Add the missing counters and the event sequence, or amend the design's
   Observability section to what is actually emitted (finding 3).
4. Add the three missing concurrency tests (concurrent channel, slow-stream
   isolation, unload with active stream).
5. Update docs/api.md and add public gRPC API documentation; decide on the
   HTTP `stream` field (finding 6).
6. Document proto regeneration (finding 7).
