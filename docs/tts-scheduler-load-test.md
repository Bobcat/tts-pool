# Scheduler and NanoVLLM Load Test

Date: 2026-08-10

## Result

The scheduler passed live progress and weighted-service checks against the
loaded NanoVLLM-VoxCPM runtime. NanoVLLM completed all 60 measured capacity
requests with valid WAV output. No HTTP failure, service warning, CUDA
out-of-memory error, or unexpected service restart occurred.

Increasing `target_inflight` and `nanovllm_max_num_seqs` together from 2 to 4
raised throughput at full capacity by about 64%. Median per-request backend
time increased by about 22%, which is the expected latency-throughput tradeoff
from batching four sequences instead of two. The test did not find a persistent
VRAM cost for the larger sequence limit.

The service was restored to its original 2/2 local configuration after the
test. No benchmark weights remain configured.

## Test Environment

- Host GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB
- NVIDIA driver: 580.126.09
- Model: `openbmb/VoxCPM2` through `nanovllm_voxcpm`
- Base Git revision: `0027a5c80a50cbf4e5646e94c3db53e66d7a99d6`
- Fairness implementation: uncommitted working-tree changes on that revision
- Service endpoint: `http://127.0.0.1:8020`
- `nanovllm_max_num_batched_tokens`: 2048
- `nanovllm_max_model_len`: 1024
- `nanovllm_gpu_memory_utilization`: 0.1
- Normal benchmark generation limit: 64
- Fairness-check generation limit: 96
- Temperature: 0.01

The script first generated a 5.125-second reference WAV with Kokoro
`af_heart`. It warmed both NanoVLLM request shapes, then tested requests without
a reference and requests paired with that reference. Each capacity result below
is the median of three batches. Request timings are the median request value in
the median batch.

The target text produced 10.24 seconds of audio per request. Throughput is the
sum of output-audio seconds divided by batch wall time; `10x`, for example,
means ten seconds of audio completed per second of wall time.

## Capacity Results

### Without Reference Audio

| Runtime config | Concurrent requests | Batch wall | Throughput | Backend/request | First chunk | Queue wait | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2/2 | 1 | 943.5 ms | 10.854x | 938.4 ms | 65.4 ms | 0.029 ms | 0.092 |
| 2/2 | 2 | 1,067.5 ms | 19.184x | 1,060.0 ms | 65.6 ms | 0.077 ms | 0.104 |
| 4/4 | 1 | 944.0 ms | 10.847x | 938.9 ms | 62.8 ms | 0.036 ms | 0.092 |
| 4/4 | 2 | 1,067.8 ms | 19.179x | 1,060.2 ms | 62.5 ms | 0.117 ms | 0.104 |
| 4/4 | 4 | 1,300.8 ms | 31.488x | 1,290.3 ms | 67.3 ms | 0.238 ms | 0.126 |

At one and two concurrent requests, the 2/2 and 4/4 configurations performed
the same. Four concurrent requests increased throughput by 64.2% over the
2/2 full-capacity result while increasing median backend time by 21.7%.

### With Paired Reference Audio

| Runtime config | Concurrent requests | Batch wall | Throughput | Backend/request | Reference encode | First chunk | Queue wait | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2/2 | 1 | 953.9 ms | 10.735x | 947.9 ms | 7.5 ms | 64.1 ms | 0.030 ms | 0.093 |
| 2/2 | 2 | 1,087.9 ms | 18.826x | 1,078.5 ms | 11.2 ms | 67.3 ms | 0.151 ms | 0.105 |
| 4/4 | 1 | 960.2 ms | 10.665x | 953.8 ms | 7.8 ms | 65.7 ms | 0.026 ms | 0.093 |
| 4/4 | 2 | 1,093.3 ms | 18.732x | 1,084.4 ms | 11.2 ms | 69.6 ms | 0.107 ms | 0.106 |
| 4/4 | 4 | 1,337.0 ms | 30.636x | 1,324.0 ms | 17.2 ms | 75.9 ms | 0.167 ms | 0.129 |

Four concurrent paired-reference requests increased throughput by 62.7% over
the 2/2 full-capacity result. Median backend time increased by 22.8%.
Reference encoding remained a small part of total request time.

## Scheduler Results

### Progress Between Services

For each runtime capacity, service A submitted twice as many jobs as available
slots. Service B submitted one job only after A occupied every slot and had
queued work. B entered the next dispatch wave when a slot became free:

| Capacity | Active A jobs | Queued A jobs | B queue wait | Difference from first queued A start | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 2 | 2 | 1,585.6 ms | 0.3 ms | pass |
| 4 | 4 | 4 | 1,952.1 ms | 0.4 ms | pass |

The sub-millisecond differences are based on client submission time plus the
server-reported queue wait, so they should be read as the same dispatch wave.
The result shows that A's existing queue did not starve the later B request.

### Weighted Service

The 4/4 run temporarily configured these trusted weights:

```json
{
  "bench-weight-a": 2.0,
  "bench-weight-b": 1.0
}
```

Both keys submitted four jobs while the runtime was saturated. The first six
backend starts contained four A jobs and two B jobs, matching the configured
2:1 share. The admin endpoint also reported the expected weights for both
active keys. The check passed.

This is a short deterministic contention check, not a statistical proof of a
long-running traffic ratio. Unit tests cover the score calculation and
tie-breaking rules separately.

## GPU Memory and Service Health

| Runtime config | Service load delta | Total GPU memory during test | Transient range |
| --- | ---: | ---: | ---: |
| 2/2 | 10,369 MiB | 77,367–77,593 MiB | 226 MiB |
| 4/4 | 10,373 MiB | 77,371–77,599 MiB | 228 MiB |

The service load delta is measured by tts-pool around model loading. Total GPU
memory comes from `nvidia-smi` and includes unrelated processes on the shared
GPU. The near-identical values show no measurable persistent or transient VRAM
penalty from changing `max_num_seqs` from 2 to 4 for this workload.

After restoring 2/2 and restarting, NanoVLLM reported `loaded`, effective
capacity 2, `max_num_seqs` 2, and `accepting_new_requests=true`. The service had
no warning-or-higher journal entries and `NRestarts=0`.

## Reproduce the Test

The benchmark uses only the Python standard library. Run it while tts-pool is
loaded and idle enough for meaningful timing results.

Test the current 2/2 configuration:

```bash
.venv/bin/python scripts/tts_pool_fairness_bench.py \
  --concurrencies 1,2 \
  --repeats 3 \
  --fairness-capacity 2 \
  --output /tmp/tts-pool-fairness-capacity2.json
```

To repeat the 4/4 and weighted test, temporarily set both
`engine.models.nanovllm_voxcpm.target_inflight` and
`engine.models.nanovllm_voxcpm.nanovllm_max_num_seqs` to `4`. Also configure
the two weights shown above under `engine.fairness.weights`, then restart
tts-pool and verify that the admin endpoint reports effective capacity 4.

```bash
systemctl --user restart tts-pool.service

.venv/bin/python scripts/tts_pool_fairness_bench.py \
  --concurrencies 1,2,4 \
  --repeats 3 \
  --fairness-capacity 4 \
  --weighted-check \
  --output /tmp/tts-pool-fairness-capacity4.json
```

Restore the original local configuration and restart the service after the
test. The script does not edit configuration or control the service.

The JSON output contains configuration snapshots, per-request metrics, batch
summaries, GPU-memory samples, and the evidence used by both scheduler checks.
It exits nonzero on HTTP errors, response-format errors, or a failed scheduler
check. A failed scheduler check is also recorded as `passed: false` in the
corresponding JSON section.

## Limits

- This was one host, one GPU, and one short-text workload.
- The test validates response structure and nonempty WAV data, not subjective
  speech quality.
- It does not measure isolated per-process VRAM while other GPU workloads are
  changing.
- It does not prove long-duration thermal or memory stability at 4/4.
- The 4/4 setting remains a tuning choice. These results support it for higher
  throughput, but a longer soak test should precede making it the local default.
