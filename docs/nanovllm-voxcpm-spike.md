# NanoVLLM-VoxCPM Spike

Status: 2026-05-11.

This note captures the phase-1 findings for adding NanoVLLM-VoxCPM as a
separate `tts-pool` backend.

## Goal

Evaluate whether NanoVLLM-VoxCPM is a good backend for concurrent VoxCPM2
serving while keeping the existing official `voxcpm` backend available as the
quality/reference baseline.

## Upstream Summary

Official VoxCPM documentation says the standard `VoxCPM.generate` path with
`torch.compile` uses CUDA Graphs and should not be used as a multi-threaded
concurrent runtime. Their recommended high-throughput path is NanoVLLM-VoxCPM.

Relevant upstream docs:

- <https://voxcpm.readthedocs.io/en/latest/faq.html#cuda-graphs-and-multi-threading>
- <https://voxcpm.readthedocs.io/en/latest/deployment/nanovllm_voxcpm.html>
- <https://github.com/a710128/nanovllm-voxcpm>

## dc1 Environment

- Host: `dc1`
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB VRAM
- Current GPU memory use during inventory: about 2,140 MiB
- NVIDIA driver: 590.48.01
- CPU: Intel Core i7-14700KF
- CPU topology: 20 cores / 28 threads
- RAM: 31 GiB
- `nvcc`: not found in `PATH`
- CUDA toolkit: `/usr/local/cuda-12.8`
- Python headers: `/usr/include/python3.12/Python.h`
- `tts-pool` Python: 3.12
- PyTorch: 2.11.0+cu130
- PyTorch CUDA: 13.0
- GPU capability reported by PyTorch: `(12, 0)`

The local Hugging Face cache already contains the VoxCPM2 snapshot with the
layout NanoVLLM expects:

- `config.json`
- `model.safetensors`
- `audiovae.pth`

Snapshot path:

```text
/home/gunnar/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/bffb3df5a29440629464e5e839f4d214c8714c3d
```

## Package And Dependency Findings

PyPI package:

```text
nano-vllm-voxcpm==2.0.1
```

Declared requirements include:

- `torch!=2.6.*,>=2.5.0`
- `triton>=3.0.0`
- `transformers>=4.51.0`
- `flash-attn`
- `torchcodec`
- `torchaudio`
- `librosa`
- `soundfile>=0.13.1`

Most dependencies are already present in the current `tts-pool` venv. The main
missing dependency was `flash-attn`.

A dry-run install currently fails while preparing `flash-attn` because the
isolated build env cannot import `torch`. That usually means `flash-attn` must
be installed with:

```bash
pip install flash-attn --no-build-isolation
```

The existing `tts-pool` venv uses PyTorch 2.11.0+cu130, while the local CUDA
compiler is 12.8. Building `flash-attn` in that venv fails with a CUDA version
mismatch. For the phase-2 direct test, a separate spike venv was created:

```text
.venv-nanovllm
```

That venv uses PyTorch 2.11.0+cu128 and a prebuilt community
`flash-attn` wheel for CUDA 12.8 / PyTorch 2.11 / Python 3.12. This avoids
changing the current service venv while still testing the NanoVLLM runtime.

## API Shape

NanoVLLM-VoxCPM exposes:

```python
from nanovllm_voxcpm import VoxCPM

server = VoxCPM.from_pretrained(
    model="/path/to/VoxCPM2",
    devices=[0],
    max_num_batched_tokens=8192,
    max_num_seqs=16,
    gpu_memory_utilization=0.95,
)
```

The sync server returns an iterator of waveform chunks:

```python
for chunk in server.generate(target_text="Hello world"):
    ...
```

The async server returns an async generator.

VoxCPM2-specific `generate` supports:

- `target_text`
- `prompt_latents`
- `prompt_text`
- `prompt_id`
- `max_generate_length`
- `temperature`
- `cfg_value`
- `ref_audio_latents`
- `lora_name`

Reference audio is not passed as a WAV file directly. It must first be encoded:

```python
ref_audio_latents = server.encode_latents(wav_bytes, "wav")
```

Then generation receives:

```python
server.generate(target_text=text, ref_audio_latents=ref_audio_latents)
```

## Fit With Current tts-pool Contract

Good fit:

- It can be modelled as a separate backend, for example `nanovllm_voxcpm`.
- It has its own internal scheduler and batching limits.
- It supports concurrent requests via `max_num_seqs`.
- It streams waveform chunks internally, while `tts-pool` can initially collect
  chunks and return the existing base64 WAV response.
- It supports VoxCPM2 reference audio through encoded `ref_audio_latents`.
- It has model metadata including output sample rate.

Resolved in phase 2:

- `nano-vllm-voxcpm==2.0.1` imports and loads with PyTorch 2.11.0+cu128.
- `flash-attn==2.8.3` imports in the spike venv.
- The local VoxCPM2 snapshot loads with NanoVLLM.
- `ref_audio_latents` generation works from a 4s WAV sample.
- Concurrent generation works at concurrency 1, 2, and 4 on the 16GB RTX 5070 Ti.

Open questions:

- Whether text control prompts behave identically enough when passed as part of
  `target_text`, compared with the official `voxcpm` prompt wrapping.
- Whether `ref_audio_latents` gives the same voice cloning quality as the
  official `reference_wav_path` path.
- Best values for `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, and
  `gpu_memory_utilization` on 16GB `dc1` versus 96GB `dc2`.
- Whether the production `tts-pool` venv should move to a CUDA 12.8 PyTorch
  stack or keep the current CUDA 13.0 stack for the official `voxcpm` backend.

## Phase 2 Direct Test

Direct test command:

```bash
.venv-nanovllm/bin/python scripts/nanovllm_direct_bench.py \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.75 \
  --max-generate-length 256 \
  --concurrency 1,2,4
```

Load/config:

| Metric | Value |
| --- | ---: |
| Load wall | 6.25s |
| GPU memory after load | 14,021 MiB / 16,303 MiB |
| 4s reference encode | 1.04s |
| Reference latents | 25,600 bytes |
| Output sample rate | 48,000 Hz |

Smoke results:

| Case | TTFB | Wall | Output audio | RTF after TTFB |
| --- | ---: | ---: | ---: | ---: |
| short, no ref | 390ms | 812ms | 3.36s | 0.13 |
| short, ref 4s | 96ms | 678ms | 4.64s | 0.13 |

Concurrency results, all with 4s reference latents:

| Concurrency | Batch wall | TTFB mean | Request wall mean | Output audio mean | RTF after TTFB mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 649ms | 88ms | 648ms | 4.5s | 0.1 |
| 2 | 953ms | 186ms | 942ms | 5.2s | 0.1 |
| 4 | 1093ms | 93ms | 1011ms | 5.5s | 0.2 |

Interpretation:

- NanoVLLM appears materially faster than the official `voxcpm` path on `dc1`.
- Concurrency 4 works within 16GB VRAM with the conservative settings above.
- The reference encoder is still a visible cost. It should be cached or reused
  when multiple TTS calls use the same speech sample.
- The current results justify adding a separate `nanovllm_voxcpm` backend to
  `tts-pool`.

## Phase 3 Integration

Implemented on 2026-05-11:

1. `tts-pool` has a separate `nanovllm_voxcpm` backend.
2. The existing official `voxcpm2` backend remains available.
3. NanoVLLM scheduling and generation limits are configurable:
   - `nanovllm_max_num_seqs`
   - `nanovllm_max_num_batched_tokens`
   - `nanovllm_max_model_len`
   - `nanovllm_gpu_memory_utilization`
   - `nanovllm_max_generate_length`
4. The runtime capability is derived from `nanovllm_max_num_seqs`.
5. The backend collects NanoVLLM chunks into the existing `/v1/responses` WAV
   response.
6. The ASR Translate TTS app can select the `nanovllm_voxcpm` model through its
   TTS options.

Still out of scope:

- true streaming responses
- reference-latent caching across repeated calls
- automatic per-host tuning for 16GB versus larger GPUs

## dc2 Blackwell Service Tuning

Deployment status on 2026-05-11:

- Host: `dc2`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM
- Runtime venv: `.venv-nanovllm`
- PyTorch: `2.11.0+cu128`
- CUDA visible to PyTorch: `12.8`
- GPU capability reported by PyTorch: `(12, 0)`
- `flash-attn==2.8.3` was built from source on `dc2` with
  `FLASH_ATTN_CUDA_ARCHS=120`; no community binary wheel is used for the dc2
  deployment.

The initial dc2 service config reserved too much VRAM. NanoVLLM's
`gpu_memory_utilization` is the dominant VRAM control; the other capacity
parameters changed reachable concurrency/limits but did not materially lower
reserved VRAM in this setup.

Measured by restarting `tts-pool.service`, waiting for `nanovllm_voxcpm` to
reach `loaded`, then summing `nvidia-smi --query-compute-apps` memory for the
`tts-pool/.venv-nanovllm` worker process.

| Step | Changed setting | Effective config | tts-pool VRAM |
| --- | --- | --- | ---: |
| Baseline | none | `gpu_util=0.35`, `seqs=4`, `target=4`, `batched_tokens=4096`, `model_len=2048` | 33,854 MiB |
| 1 | `gpu_memory_utilization=0.25` | `seqs=4`, `target=4`, `batched_tokens=4096`, `model_len=2048` | 24,206 MiB |
| 2 | `gpu_memory_utilization=0.20` | `seqs=4`, `target=4`, `batched_tokens=4096`, `model_len=2048` | 19,310 MiB |
| 3 | `max_num_seqs=2` | `gpu_util=0.20`, `target=4`, `batched_tokens=4096`, `model_len=2048` | 19,306 MiB |
| 4 | `target_inflight=2` | `gpu_util=0.20`, `seqs=2`, `batched_tokens=4096`, `model_len=2048` | 19,306 MiB |
| 5 | `max_num_batched_tokens=2048` | `gpu_util=0.20`, `seqs=2`, `target=2`, `model_len=2048` | 19,882 MiB |
| 6 | `max_model_len=1024` | `gpu_util=0.20`, `seqs=2`, `target=2`, `batched_tokens=2048` | 19,882 MiB |
| 7 | `gpu_memory_utilization=0.15` | `seqs=2`, `target=2`, `batched_tokens=2048`, `model_len=1024` | 14,986 MiB |
| 8 | `gpu_memory_utilization=0.12` | `seqs=2`, `target=2`, `batched_tokens=2048`, `model_len=1024` | 12,106 MiB |
| 9 | `gpu_memory_utilization=0.10` | `seqs=2`, `target=2`, `batched_tokens=2048`, `model_len=1024` | 10,090 MiB |
| 10 | `gpu_memory_utilization=0.08` | `seqs=2`, `target=2`, `batched_tokens=2048`, `model_len=1024` | 10,090 MiB |

Interpretation:

- `gpu_memory_utilization` controls the reserved VRAM almost linearly until
  roughly the 10 GiB floor seen at `0.10` and `0.08`.
- Lowering `max_num_seqs` and `target_inflight` from 4 to 2 reduces concurrency
  capacity but does not itself release meaningful VRAM after load.
- Lowering `max_num_batched_tokens` and `max_model_len` did not reduce the
  measured reserved VRAM in this test. It may still reduce pathological request
  shapes and should remain conservative.
- The current dc2 default is intentionally conservative:

```json
{
  "target_inflight": 2,
  "nanovllm_max_num_seqs": 2,
  "nanovllm_max_num_batched_tokens": 2048,
  "nanovllm_max_model_len": 1024,
  "nanovllm_gpu_memory_utilization": 0.10
}
```

Smoke results at the conservative setting:

| Case | Wall | Pool wall | Reference encode | Generate | Output audio |
| --- | ---: | ---: | ---: | ---: | ---: |
| no reference | 795ms | 739ms | n/a | n/a | 1.28s |
| 2s reference, first after restart | 1,169ms | 1,109ms | 787ms | 320ms | 3.04s |
| 2s reference, warm repeat | 366ms | 287ms | 3.6ms | 282ms | 2.72s |

Speed check for `gpu_memory_utilization`:

Method: keep `target_inflight=2`, `nanovllm_max_num_seqs=2`,
`nanovllm_max_num_batched_tokens=2048`, and `nanovllm_max_model_len=1024`
fixed; restart the service per value; warm no-reference and 2s-reference
request shapes once; then measure median of three requests. Generation
temperature was set to `0.01` to reduce output variance.

| `gpu_memory_utilization` | tts-pool VRAM | No-ref generate | No-ref first chunk | Ref encode | Ref generate | Ref output audio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 10,364 MiB | 377.1ms | 59.4ms | 4.1ms | 2,527.2ms | 28.64s |
| 0.20 | 20,156 MiB | 377.0ms | 59.2ms | 4.2ms | 2,531.3ms | 28.64s |
| 0.35 | 34,700 MiB | 377.9ms | 59.5ms | 4.3ms | 3,607.6ms | 40.96s |

Interpretation: increasing `gpu_memory_utilization` did not improve warm
single-request generation speed in this test. The slower absolute reference
case at `0.35` produced proportionally longer audio; its generation
real-time-factor is effectively the same. Keep `0.10` unless future concurrency
or long-context tests show cache pressure.

## NanoVLLM Warmup

NanoVLLM-VoxCPM does not expose an `optimize=True` equivalent like the official
VoxCPM2 backend. The useful optimization point is request-shape warmup: run a
few short generations after model load so the first user request does not pay
the cold generate/reference-encode cost.

Implemented settings:

- `nanovllm_warmup_enabled`
- `nanovllm_warmup_cases`

Default warmup cases, used when the setting is enabled without custom cases:

1. short text, no reference audio
2. short text, 2s synthetic reference audio, voice match
3. short text, 2s synthetic reference audio, voice + pace match

Each default case uses `temperature=0.01` and `max_generate_length=64` so warmup
stays bounded. The shared `config/settings.json` default is still disabled;
`dc2` enables it in `config/local.json`.

### Fresh Startup A/B

Measured on `dc2` on 2026-05-11 with the conservative config:
`target_inflight=2`, `nanovllm_max_num_seqs=2`,
`nanovllm_max_num_batched_tokens=2048`, `nanovllm_max_model_len=1024`,
`nanovllm_gpu_memory_utilization=0.10`.

Method: restart `tts-pool.service`, wait until `nanovllm_voxcpm` reports
`loaded`, then send the same three requests sequentially. The reference request
uses a 2s synthetic WAV. Request generation settings were `temperature=0.01`
and `max_generate_length=64`.

| Run | Usable after restart | Case | Wall | First chunk | Reference encode | Generate | Output audio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| warmup off | 9.2s | no reference | 1,498ms | 590.0ms | n/a | 1,491.4ms | 10.24s |
| warmup off | 9.2s | 2s reference, voice | 1,763ms | 77.2ms | 801.0ms | 949.0ms | 10.24s |
| warmup off | 9.2s | 2s reference, voice + pace | 403ms | 62.7ms | 4.3ms | 394.3ms | 4.00s |
| warmup on | 12.7s | no reference | 938ms | 60.7ms | n/a | 930.9ms | 10.24s |
| warmup on | 12.7s | 2s reference, voice | 939ms | 59.5ms | 3.5ms | 930.3ms | 10.24s |
| warmup on | 12.7s | 2s reference, voice + pace | 400ms | 59.8ms | 3.6ms | 391.4ms | 4.00s |

Interpretation:

- Startup-to-usable increased from 9.2s to 12.7s, so this warmup costs about
  3.5s at service start.
- The first no-reference request no longer pays the cold first-chunk penalty:
  590.0ms dropped to 60.7ms.
- The first reference request no longer pays the cold reference encoder cost:
  801.0ms dropped to 3.5ms.
- The third request is similar in both runs because the preceding reference
  request already warmed the reference path in the warmup-off run.

## Current App Routing

The app repo local config is pointed to the dc2 TTS pool:

```json
"tts_pool": {
  "base_url": "http://dc2:8020",
  "timeout_s": 300
}
```

The app loads settings at import time, so the running app process on port
`8003` must be restarted after changing this value.
