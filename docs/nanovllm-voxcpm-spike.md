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

## Current App Routing For Local Spike

The app repo local config has been pointed back to the local `dc1` TTS pool:

```json
"tts_pool": {
  "base_url": "http://127.0.0.1:8020",
  "timeout_s": 300
}
```

The app loads settings at import time, so the running app process on port `8003`
must be restarted before this takes effect.
