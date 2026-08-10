# TTS Scheduler Fairness

Status: design proposal, 2026-08-10.

## Goal

Let one caller use idle synthesis capacity without allowing that caller to keep
every runtime slot when other callers are waiting.

The first implementation should:

- preserve FIFO order within one stable scheduling identity;
- share runtime slot-time fairly between contending identities;
- let one identity borrow every free slot while it is alone;
- keep the executor work-conserving;
- bound pending work and reject overload explicitly;
- expose enough state to explain queue decisions;
- keep backend runtimes unaware of scheduling identity.

This policy is needed before an app submits several WAV requests for one user in
parallel. It allocates pool capacity. It does not implement app-side prefetch,
audio caching, stale-work cancellation, or playback ordering.

## Reference Policy

This design adapts the per-model client fairness implemented in llm-pool at
commit `6e66eaa`. That scheduler replaced one global FIFO with per-key FIFO
buckets, weighted least-served slot-time, a work-conserving soft inflight cap,
bounded pending queues, and per-key observability.

The two pools have the same relevant boundary: a model executor admits work up
to an effective runtime capacity. TTS uses a different request and result type
and currently has no model replicas. The implementation should therefore port
the fairness policy and its deterministic tests, not copy the llm-pool
scheduler module or its replica layer.

TTS-specific choices remain local to this service. In particular, pending queue
sizes must account for base64 reference audio, and Nano-vLLM capacity must stay
bounded by `max_num_seqs`.

## Current Behavior

Each loaded model has one `LoadedModelExecutor`. The executor currently owns:

- one global FIFO `deque`;
- one configured `target_inflight`;
- one runtime capability;
- one worker thread per admitted synthesis request.

The effective capacity is already bounded correctly:

```text
effective_target_inflight = min(target_inflight, runtime_capability)
```

For NanoVLLM-VoxCPM, `runtime_capability` is `max_num_seqs`. The TTS executor
therefore controls how many requests may enter the runtime concurrently.

The global FIFO has no caller identity. A burst submitted first can fill every
runtime slot and every early queue position. A later request from another
caller waits behind that burst. The pending queue is also unbounded.

## Scope

Fairness applies inside one loaded model executor.

- Requests for different public model ids have independent queues and scores.
- The policy does not arbitrate between models sharing one GPU.
- The policy does not change Nano-vLLM's internal sequence scheduler.
- The policy does not reserve or preempt a running runtime slot.
- The policy is independent of subscription plans, quotas, and rate limits.

The pool schedules stable identities, not application users directly. A trusted
caller may map one app principal to one stable identity. The pool does not need
principal, tenant, session, or product vocabulary.

## Request Identity

Add one optional field to `ResponseRequest`:

```python
fairness_key: str | None = None
```

A supplied key must:

- be a string;
- contain non-whitespace text;
- be trimmed before use;
- contain at most 128 characters after trimming.

An omitted key belongs to one anonymous bucket. The anonymous bucket is a
normal participant with the default weight. This keeps the contract useful for
internal callers that do not need per-caller attribution.

The key is scheduling metadata only. It must not be forwarded as a VoxCPM,
Kokoro, or other backend generation parameter.

### Trust boundary

Requests declare their identity but not their scheduling weight. This is safe
while callers are trusted internal services.

The Omni Translate app should derive an opaque key from its resolved principal
and retain it with the voice session. The browser must not choose or override
that key. Multiple sessions for the same principal should use the same key, so
opening more sessions does not create a larger aggregate share.

If untrusted callers later reach `tts-pool` directly, an authenticated gateway
must assign or validate the key. Otherwise a caller could evade fairness by
minting a fresh key for every request.

## Pool-Owned Settings

Add one pool-wide section under `engine.fairness`:

| Setting | Initial direction |
| --- | --- |
| `default_weight` | `1.0` |
| `weights` | Empty mapping; reserved for trusted named workloads |
| `soft_max_inflight_per_key` | `1` |
| `max_pending_per_key` | Decide after request-memory and load measurement |
| `max_pending_per_executor` | Decide after request-memory and load measurement |
| `idle_state_ttl_s` | Candidate: llm-pool's 300 seconds; verify TTS key cardinality |

Weights must be finite and greater than zero. Requests must not contain a
weight or priority field. A key missing from `weights` uses `default_weight`.

The queue limits need TTS-specific measurements. A pending TTS request can hold
base64 reference audio and is materially larger than a normal text-only LLM
request. Do not copy llm-pool queue depths without measuring this payload cost.

The same fairness settings may be read by every model executor. Each executor
keeps independent state. Per-model weight overrides are not needed in the first
version.

## Scheduling Policy

Use weighted least-served scheduling over backend slot-time. This is the small
policy port from llm-pool. Do not port its replica layer.

### Per-key state

Each model executor keeps:

- one FIFO `deque` of pending jobs per normalized key;
- completed normalized slot-time, `virtual_service_ms`;
- start times for active jobs;
- active job count;
- deterministic selection order for tie-breaking;
- an idle timestamp;
- per-key queue rejection counters.

Aggregate pending depth remains the sum of all key buckets.

### Service unit

Charge elapsed time spent inside the executor's backend call:

```text
service_ms = backend_finished_at - backend_started_at
charge = service_ms / weight
```

For the current NanoVLLM-VoxCPM runtime, this interval includes reference-audio
preparation inside the runtime, reference latent encoding, model generation,
and WAV assembly. It is runtime-slot occupancy, not exact GPU kernel time.

Do not charge queue wait or `engine_total_wall_ms`. A request receives no
service while it is pending.

Charge failed backend calls in `finally`. A failure still occupied a runtime
slot. A job abandoned before its worker starts receives no charge.

With four concurrent jobs, four milliseconds of slot-time can be charged per
millisecond of wall time. This is intentional. Nano-vLLM may batch those jobs in
one GPU step, but each request consumes one TTS admission slot and can become
one Nano sequence.

### Selection score

Completed service alone reacts too late while several jobs are active. Include
their elapsed service in the selection score:

```text
score(key, now) =
    virtual_service_ms[key]
    + sum(now - active_started_at) / weight[key]
```

Use one consistent time unit in the implementation.

When a slot becomes free:

1. collect keys with pending jobs;
2. prefer keys below `soft_max_inflight_per_key`;
3. if none are below the soft cap, consider every pending key;
4. calculate each candidate's score;
5. choose the lowest score;
6. break equal scores with deterministic round-robin order;
7. pop the oldest job from that key;
8. record its active start time;
9. start the existing worker path.

Step 3 makes the policy work-conserving. The soft cap is an anti-monopoly
preference, not a hard per-key concurrency limit.

On completion:

1. remove the active job from its key;
2. add `service_ms / weight` to `virtual_service_ms`;
3. notify the executor loop that capacity and scores changed;
4. complete the existing result future.

### New and returning keys

A new key must not start at zero while current contenders have positive scores.
Initialize it at the minimum score of the active or queued keys. Use zero only
when no other key is active or queued.

Keep idle score state for a bounded TTL. This prevents a bursty caller from
resetting its history between adjacent bursts. Expire the state after the TTL
to bound retained per-key state. A returning key then enters at the current
minimum score.

Do not add score decay in the first version.

## Capacity Behavior

Assume `effective_target_inflight = 4` and a soft per-key cap of one.

- If only key A has work, A may borrow all four slots.
- If A and B are both queued while slots are available, each gets a slot before
  either key borrows solely because the other key is at its soft cap.
- If A already occupies all four slots when B arrives, B gets the next slot
  released by A.
- Running work is never interrupted.

This provides fair progress and full utilization. It does not guarantee a
maximum wait for a newly arriving key. A strict guarantee would require a hard
cap, reserved capacity, or preemption. Those policies would either leave idle
capacity unused or require runtime cancellation semantics.

`target_inflight` should normally be aligned with Nano-vLLM `max_num_seqs` when
the full runtime capacity is intended for use. The existing `min(...)` boundary
must remain authoritative when they differ.

## Queue Limits And Rejections

Bound pending work in two places:

- `max_pending_per_key` limits one identity's waiting jobs;
- `max_pending_per_executor` limits all waiting jobs for one model.

Active jobs do not count as pending. Check the per-key limit first, followed by
the executor total.

Reject before enqueue with HTTP `429`:

| Condition | Error code |
| --- | --- |
| Per-key pending limit reached | `fairness_key_queue_full` |
| Executor pending limit reached | `executor_queue_full` |

Add a small `RequestAdmissionError` carrying `status_code`, `code`, and
`message`. The API layer maps it without converting overload into a synthesis
failure.

Fair scheduling is not rate limiting. A retry storm can still consume its fair
share continuously. Product quotas, request rates, and IP controls remain
separate layers.

## Observability

Keep the existing aggregate executor fields:

- `queue_depth`;
- `runtime_inflight`;
- `configured_target_inflight`;
- `effective_target_inflight`;
- `accepting_new_requests`.

Extend the bounded `/v1/admin/models` entry with:

```json
{
  "fairness": {
    "rejected_per_key_limit": 0,
    "rejected_executor_limit": 0,
    "keys": [
      {
        "fairness_key": "opaque-app-key",
        "pending": 2,
        "active": 1,
        "weight": 1.0,
        "score": 1842.5,
        "rejected_per_key_limit": 0,
        "rejected_executor_limit": 0
      }
    ]
  }
}
```

Only active or pending keys need to appear in the admin response. Idle score
history can remain internal until it expires.

Include the normalized key in the compact inference log. The app should send
an opaque value rather than an email address or display name. Do not use
fairness keys as unbounded time-series metric labels.

The existing timing metrics remain valid. In particular:

- `engine_queue_wait_ms` measures time before fair admission;
- `backend_synthesis_wall_ms` is the charged service interval on success;
- backend-specific Nano-vLLM metrics explain reference encoding and generation.

## Lifecycle

Model unload keeps the existing graceful boundary:

1. stop accepting new jobs;
2. drain every pending key bucket;
3. fail drained futures with `model_unloading`;
4. allow active calls to leave the runtime;
5. discard all fairness state with the executor;
6. close the runtime.

Worker-start failure must remove the job from active state, release the executor
slot, and fail its future. It must not leave a phantom active charge.

## Implementation Boundary

The scheduler feature should remain one scoped vertical slice:

| File | Change |
| --- | --- |
| `app/schemas.py` | Optional normalized `fairness_key`; bounded admin fairness schemas |
| `app/config.py` | `FairnessSettings` and `engine.fairness` validation |
| `app/engine/common.py` | Stable admission error type |
| `app/engine/scheduler.py` | Per-key queue, scoring, limits, completion charge, snapshots |
| `app/engine/router.py` | Pass settings into each executor and expose snapshots |
| `app/engine/__init__.py` | Export the admission error for the API boundary |
| `app/main.py` | Map admission errors to `429`; log normalized key |
| `config/settings.json` | Pool-owned policy values after the queue-size decision |
| `docs/api.md` | Document the request field and overload responses |
| `tests/` | Contract, scheduler, config, API, router, and lifecycle coverage |

Do not introduce a generic cross-repository scheduler package. Port the policy
and tests from llm-pool, then keep TTS-specific result handling and its single
runtime executor shape.

Do not add replicas as part of this work. If TTS model replicas are introduced
later, one public model should keep one shared fairness queue across them.

## Test Contract

Add deterministic tests for:

- fairness-key trimming, length, blank rejection, and anonymous default;
- FIFO order within one key;
- round-robin tie-breaking between equal keys;
- weighted long-run slot-time with unequal synthesis durations;
- active elapsed time affecting selection before completion;
- one key borrowing every slot while alone;
- a waiting key receiving the next released slot;
- no idle slot when every queued key is at its soft cap;
- a new key entering at the current minimum score;
- idle-state expiry;
- failed synthesis receiving a slot-time charge;
- worker-start failure abandoning active state without a charge;
- per-key and executor queue-limit `429` responses;
- aggregate queue depth across key buckets;
- admin snapshot contents and bounded key visibility;
- inference logging of the normalized key;
- unload draining every pending bucket;
- effective capacities 1, 2, and 4.

Use fake runtimes and controlled events for scheduler tests. The deterministic
suite must not require CUDA or a loaded TTS model.

After the scheduler tests pass, run a separate NanoVLLM-VoxCPM load test with
representative reference audio. Compare concurrency 1, 2, and 4 using:

- queue wait;
- reference-encode time;
- time to first internal audio chunk;
- total synthesis wall time;
- realtime factor;
- completed audio per wall-clock second;
- peak GPU memory;
- failures and timeouts.

Use that measurement to choose the pending queue limits and whether live
`target_inflight` should move beyond two.

## Rollout Order

1. Fast-forward the local TTS checkout to origin before implementation.
2. Implement and test the optional fairness contract in `tts-pool`.
3. Deploy it with existing callers in the anonymous bucket and inspect admin
   state under controlled concurrent load.
4. Let the app assign one stable, opaque key per resolved principal.
5. Verify that two principals share capacity and one principal can still borrow
   idle slots.
6. Only then let the app dispatch several WAV requests for one principal.
7. Treat pre-generation before the play control as a later latency feature.

## Non-Goals

The first scheduler slice does not add:

- app authentication or principal resolution inside `tts-pool`;
- subscription entitlements, usage accounting, or rate limiting;
- fairness across model executors or GPUs;
- hard per-key active limits;
- reserved interactive slots or caller-selected priorities;
- preemption of active synthesis;
- Nano-vLLM scheduler changes;
- backend-native response streaming;
- app-side WAV prefetch or caching;
- queued or active request cancellation;
- playback-order state;
- a TTS replica architecture.

## Decisions Before Code

The scheduling algorithm and request identity shape are settled by this design.
Choose these operational values before changing `config/settings.json`:

1. `max_pending_per_key`;
2. `max_pending_per_executor`;
3. `idle_state_ttl_s`;
4. the first deployed `target_inflight` and Nano-vLLM `max_num_seqs` pair.

Base those choices on measured request memory and concurrency behavior. They are
runtime controls, not product entitlements.
