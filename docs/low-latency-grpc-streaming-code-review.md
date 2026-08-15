# Code Review: Low-Latency gRPC Streaming

Reviews the uncommitted implementation of
[Low-Latency TTS Delivery](low-latency-tts-delivery.md) on branch
`feature/low-latency-grpc-streaming`.

- **Scope** — all modified tracked files plus the untracked
  `app/engine/streaming.py`, `app/grpc_api/`, `proto/`,
  `scripts/tts_transport_bench.py`, `tests/test_grpc_api.py`,
  `tests/test_streaming.py`. No code was changed.
- **Method** — read the full diff against the design, traced the threading and
  cancellation paths by hand, verified specific claims by running code.
- **Test run** — `python -m unittest discover -s tests`: 77 tests, all pass.

## Verdict

The architecture matches the design. One generation core feeds both the
complete and the streaming path, one request is still one scheduler job, the
buffer is bounded on chunks and bytes, and cancellation reaches the NanoVLLM
generator. Phases 1-4 are functionally there.

The findings below are one scaling defect, one interop defect, one bypassed
input limit, and a set of design deviations. Nothing requires restructuring.

## Verified as correct

These were checked, not assumed:

- **PCM equivalence.** The per-chunk conversion (clip to `[-1.0, 1.0]`,
  `* 32767.0`, truncate to `<i2`) is elementwise identical to the old
  whole-array conversion in `_wav_bytes` (app/engine/voxcpm2.py:355-356). The
  `duration_ms` formula is unchanged: `int(samples / sample_rate * 1000)`. The
  complete path is byte-identical to before this change.
- **Slot accounting.** Service time is charged from dispatch until the runtime
  iterator exits, including for cancelled and failed jobs
  (app/engine/scheduler.py:445-458). A stalled consumer blocks the producer
  thread, so it cannot release a model slot while NanoVLLM still holds the
  sequence — which is what the design requires.
- **Fairness is untouched.** Queued-job removal (app/engine/scheduler.py:140)
  does not adjust virtual service, so cancelling a queued stream does not skew
  the key's score. HTTP and gRPC jobs share one `_FairPendingQueue`.
- **Cancel coverage.** Checked before dispatch (scheduler.py:422), before
  generation, after reference encoding, and between chunks; the async generator
  is closed in a `finally` (app/engine/nanovllm_voxcpm.py:297-300).
- **Lifecycle order.** The gRPC server is stopped before `engine.close()`
  (app/main.py:60-65), and a failed bind raises instead of starting degraded.
- **Version pins.** `grpcio>=1.83` and `protobuf>=7.35.1` match both the
  installed runtimes (1.83.0 / 7.35.1) and the gencode headers. The protobuf
  runtime check only rejects a runtime *older* than the gencode, so the
  unbounded upper pin is safe here.

## Findings

### 1. Each open RPC parks a thread in asyncio's default executor

`app/grpc_api/service.py:47` reads events with
`await asyncio.to_thread(handle.read_event)`, and `read_event`
(app/engine/streaming.py:189-202) blocks until an event arrives or the handle
terminates. One RPC therefore holds one pool thread from admission until its
next event — for a queued stream, that is the entire queue wait.

`asyncio.to_thread` uses the loop's default `ThreadPoolExecutor`, sized
`min(32, cpu_count + 4)`. Admission allows `max_pending_per_executor` (8) plus
`effective_target_inflight` concurrent streams *per model*.

Failure mode when concurrent RPCs exceed the pool: the excess RPCs never get a
thread, so nobody drains their handle. The buffer fills at
`stream_buffer_chunks` (4), the producer waits `stalled_consumer_timeout_s`
(2.0 s), and the synthesis dies with `StreamConsumerStalled` →
`RESOURCE_EXHAUSTED` — against a perfectly healthy client. It is a server-side
thread shortage reported as consumer backpressure.

On the current 28-core host the pool is 32 and the per-model cap is about 12,
so it needs several loaded models or a small container to bite. Uvicorn's sync
endpoints use anyio's own thread limiter, so there is no contention with the
HTTP path.

Fix options: give the read loop a dedicated executor sized to the admission
cap, or make `SynthesisHandle` awaitable so the transport never blocks a
thread.

### 2. Generated descriptors carry the wrong file name

`app/grpc_api/v1/tts_pb2.py` registers its descriptor as `tts.proto`
(verified: `tts_pb2.DESCRIPTOR.name == 'tts.proto'`). The canonical source is
`proto/tts/v1/tts.proto`, so protoc was run from inside the leaf directory.

Any client generated from the canonical layout — `protoc -I proto
tts/v1/tts.proto` — registers `tts/v1/tts.proto` with the same `tts.v1`
package and the same message names. Importing both modules into one Python
process makes the default descriptor pool fail:

```text
TypeError: Couldn't build proto file into descriptor pool:
duplicate symbol 'tts.v1.ReferenceAudio'
```

That is exactly the shape of a client library that vendors its own stubs and
also imports something from TTS-pool. Regenerate with `-I proto` so the
descriptor path matches the versioned layout the design specifies.

### 3. Binary reference audio bypasses the size cap

`ReferenceAudio.from_bytes` (app/schemas.py:38-56) sets `data_base64=""` and
stores the payload in a `PrivateAttr`, so the
`Field(max_length=MAX_REFERENCE_AUDIO_BASE64_CHARS)` guard (app/schemas.py:20)
never sees it. Three consequences:

- **Limits diverge.** HTTP caps at 16,777,216 base64 chars, about 12.6 MiB
  decoded. gRPC is bounded only by `max_receive_message_bytes` (16 MiB for the
  whole message) — roughly 30% more reference audio.
- **The design's cross-check is missing.** "The message limits must remain at
  least as strict as the current reference-audio limit" is not validated
  anywhere; a 64 MiB message limit passes startup.
- **The model has an inconsistent invariant.** `data_base64` and `_data_bytes`
  can disagree, and `model_dump()` drops the private attr. Nothing round-trips
  the request through pydantic serialization today, but if anything does, the
  reference audio silently becomes empty.

Enforce a decoded-byte cap in `from_bytes`, and add the startup cross-check.

### 4. The default config exposes gRPC wider than HTTP

`config/settings.json` ships `service.host: "127.0.0.1"` but
`service.grpc.host: "0.0.0.0"`, served through `add_insecure_port`
(app/grpc_api/server.py:39) with no authentication. `fairness_key` is
client-supplied (app/grpc_api/adapter.py:42) and is the scheduler's quota
identity — the design calls it "trusted".

This follows the design ("intended for a trusted private network"), so it is a
deployment question, not a code defect. But the committed default now binds
every interface while the HTTP API stays on loopback. Either default the gRPC
host to `127.0.0.1` and let the deployment widen it, or state the required
firewall boundary in the config and the docs.

### 5. `read_event` does not wake on cancel

The wait condition observes only `_events` and `_terminal`
(app/engine/streaming.py:191). `cancel()` sets `_cancelled` and calls
`notify_all()`, but a blocked reader re-evaluates the same condition and goes
back to sleep.

It works today because termination always follows: the queued path fails the
handle from the cancel callback, and the active path ends when the runtime
raises `SynthesisCancelled` at its next chunk boundary. The cost is that a
client-cancelled RPC leaves its pool thread parked until the runtime notices —
which is finding 1's pool under extra pressure precisely when clients are
disconnecting. Include `_cancelled` in the condition.

### 6. No startup validation of buffer size against chunk size

The design requires startup to reject "a buffer too small for one
maximum-size runtime chunk". Today an oversized chunk raises
`StreamConsumerStalled` (app/engine/streaming.py:140-141) mapped to
`RESOURCE_EXHAUSTED` (app/grpc_api/service.py:150-151): a server
misconfiguration, surfaced mid-synthesis, reported to the client as its own
fault.

Maximum chunk size is model knowledge, so a load-time check is more realistic
than a startup one — the sample rate is known by then. Failing that, document
the required relationship between `stream_buffer_bytes` and runtime chunk
size.

### 7. Observability is well short of the design

Implemented: one JSON line per completed stream with the full metrics dict
(app/grpc_api/service.py:66-78), plus two cancellation lines. Missing:

- the whole `request_received` → `runtime_released` event sequence;
- every listed counter — stalled-consumer cancellations, queued versus active
  cancellations, cancellation-to-release latency, gRPC status counts,
  complete-versus-streaming request counts.

Separately, the gRPC log line carries no `fairness_key`, so streaming traffic
cannot be attributed to a tenant at all — while the HTTP line logs the raw key
(app/main.py:43). Pick one rule and apply it to both transports.

Most raw values already reach `Completed.metrics`, so this is aggregation, not
new instrumentation.

### 8. Proto regeneration is not reproducible

There is no script, Makefile target, or documented command. It also is not a
one-liner: `tts_pb2_grpc.py` uses `from . import tts_pb2`, which plain
`grpc_tools.protoc` does not emit, so a naive regeneration produces a
top-level `import tts_pb2` and breaks the package. The `dev` extra adds
`grpcio-tools` with nothing in the repo that uses it. Add the exact command,
including the `-I proto` from finding 2 and the import rewrite.

### 9. Empty proto3 scalars pass through unchecked

`request_from_proto` (app/grpc_api/adapter.py:38-47) forwards `model`,
`input`, and `language` without validation. proto3 has no presence for plain
strings, so an omitted field arrives as `""`. An empty `input` then fails deep
in the runtime and maps to `INTERNAL`, where the same omission over HTTP gets
a 422 from pydantic. Reject blank `input`, `model`, and `language` in the
adapter so they map to `INVALID_ARGUMENT`.

## Design deviations worth a decision

- **HTTP `stream` field.** The design says remove it when the gRPC contract
  lands. It is still in the schema (app/schemas.py:108) and still 400s
  (app/main.py:139-140). Keeping the explicit rejection is defensible, but
  then docs/api.md:48 and README.md:113,129 are stale.
- **No gRPC documentation.** Neither docs/api.md nor the README mentions the
  new transport, the proto contract, or the error-mapping table.
- **`Started` has no generation identity.** The design lists "normalized
  generation identity needed for logging"; the proto carries response ID,
  model, format, and queue wait only (proto/tts/v1/tts.proto:63-70).
  Generation parameters appear only in `Completed.metadata`, which arrives
  after the audio.
- **Shutdown grace.** The design gives active streams "a bounded grace period
  before forced shutdown". `begin_shutdown` cancels them immediately
  (app/engine/scheduler.py:365-366), and `unload_model` then waits on
  `inflight_requests` with no timeout (app/engine/router.py:262-265). Bounded
  in practice by NanoVLLM's per-chunk cancel checks — by convention, not by
  mechanism.

## Nits

- `app/grpc_api/service.py:30` logs to `uvicorn.error`; every other module uses
  a `tts_pool.*` logger. Lines are lost if the process runs without uvicorn.
- `router.stream(response_id=...)` is never passed by any caller — dead
  parameter. Response-ID generation now exists in two places
  (app/main.py:142, app/engine/router.py:129).
- `_run_terminal_callbacks` swallows every exception
  (app/engine/streaming.py:231-236). The router's inflight accounting runs in
  one of those callbacks, so a bug there is invisible and would hang
  `unload_model`.
- `_FairPendingQueue.remove` uses `deque.remove(job)`, i.e. dataclass
  `__eq__`. Field order puts `request` first, so cancelling a queued job
  deep-compares pydantic `ResponseRequest` objects — reference audio included —
  for each preceding job, holding `_cond`. Not a correctness bug: `job_id`
  makes false matches impossible. Matching on `job_id` avoids the work.
- Cancel between `enqueue` and `set_cancel_callback`
  (app/engine/scheduler.py:331-337) leaves the job queued until dispatch. The
  pre-dispatch check still prevents runtime work; only queue depth lingers.
- `StreamConsumerStalled` covers two different operator problems under one
  client-facing code: one oversized chunk (streaming.py:141) and an undrained
  buffer (streaming.py:151).
- The complete path computes `nanovllm_total_wall_ms` and `realtime_factor`
  twice and discards the inner pair (app/engine/nanovllm_voxcpm.py:120-128 vs
  307,333).
- `scripts/tts_transport_bench.py` needs `h2`, which is in no extra. It does
  print an install hint.
- `first_chunk_written_at` is stamped after the generator resumes past the
  `yield` (app/grpc_api/service.py:62-64) — a fair approximation of "handed to
  transport", worth knowing when reading `grpc_first_chunk_wall_ms`.

## Test coverage against the acceptance criteria

Covered: PCM/WAV equivalence, queued cancellation preventing dispatch, active
cancellation releasing the slot and closing the generator, bounded buffers,
stalled-consumer cancellation, error-code trailing metadata, health and
lifespan start/stop, config validation. All pre-existing fairness and HTTP
tests still pass.

Not covered:

- **Anything crossing the transport under load.** Both gRPC tests use a handle
  that `RecordingEngine.stream` fills completely before returning
  (tests/test_grpc_api.py:25-47), so the servicer never blocks in `read_event`.
  Backpressure, client cancellation, and finding 1 are all invisible to the
  suite.
- **Concurrent RPCs over one channel** — an explicit acceptance criterion.
- **A slow stream not stopping unrelated streams** — an explicit acceptance
  criterion.
- **Unload and shutdown with an active stream** — correct by construction, not
  by test.
- **Measured sequence-release behaviour.** The design gates any "active
  cancellation saves GPU work" claim on a live measurement; the current proof
  runs against a fake server. Keep that claim out of the docs until measured.
- **Phase 5 live validation.** `scripts/tts_transport_bench.py` is a synthetic
  raw-HTTP/2-versus-gRPC comparison, not end-to-end TTS. Loopback and
  dc1-to-dc2 first-chunk numbers are still open.

## Suggested order

1. Fix the executor sizing behind `read_event` (finding 1) and add the
   `_cancelled` wake (finding 5) — they share a fix surface.
2. Regenerate the stubs with `-I proto` and check in the command (findings 2
   and 8).
3. Cap decoded reference bytes and add the message-limit cross-check
   (finding 3).
4. Decide the gRPC bind default and write down the trust boundary (finding 4).
5. Add the three missing concurrency tests — concurrent channel, slow-stream
   isolation, unload with an active stream.
6. Close the observability gap, or amend the design's Observability section to
   what is actually emitted (finding 7).
7. Adapter input validation (finding 9), then the doc updates.
