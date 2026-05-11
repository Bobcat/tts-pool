# VoxCPM2 Optimization Notes

Status: 2026-05-11.

These notes track what we have learned while tuning VoxCPM2 in `tts-pool`,
especially on `dc2`.

## Measurement Environment

Unless stated otherwise, benchmark numbers in this document were measured on
`dc2` through the running `tts-pool` service.

- Host: `dc2`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM
- NVIDIA driver: 580.126.09
- System CUDA reported by `nvidia-smi`: 13.0
- CPU: AMD Ryzen 9 9950X 16-Core Processor
- CPU topology: 16 cores / 32 threads
- RAM: 60 GiB
- Service Python: 3.12.3
- PyTorch: 2.11.0+cu130
- PyTorch CUDA: 13.0

## Goals

- Reduce perceived latency for VoxCPM2 synthesis with reference WAV input.
- Keep voice/reference quality high enough for live translation use.
- Move expensive compile/warmup work out of the first real user request.

## Findings So Far

Reference WAV upload and clipping are not the main latency source. In controlled
tests, 1s/4s/8s reference audio changed request latency only slightly after
warmup. Internal profiling showed reference encoding at only a few milliseconds
after the first cold path.

The main cost is VoxCPM2 generation. Latency mostly follows generated audio
length and the autoregressive/diffusion loop, not the reference WAV duration.

`inference_timesteps` matters. Lowering it from `10` to `6` or `4` reduces
latency, with a quality tradeoff that must be evaluated by listening.

`optimize=true` helps in steady state, but introduces compile behavior:

- model startup performs a VoxCPM2 warmup;
- first real request shapes can still trigger additional compile latency;
- the compile cache is process/runtime behavior, not a model-weight change;
- running later with `optimize=false` does not use those compiled paths.

Language itself is probably not a compile-shape dimension in the current
integration. The service passes `language` through metadata and voice preset
lookup, but VoxCPM2 receives `text`, optional control text, optional reference
WAV, and generation params. Text length/tokenization and prompt shape matter
more than `Dutch` versus `English` as a field value.

## What We Have Done

- Kept `voxcpm2` loaded on `dc2` for benchmarking.
- Added service-side reference WAV clipping and request/config controls for
  maximum reference duration.
- Measured baseline non-optimized VoxCPM2 latency over HTTP.
- Tested `voxcpm2_optimize=true`.
- Found that `torch.compile` initially failed on `dc2` because Python dev
  headers were missing.
- Installed Python 3.12 dev headers locally under:

  ```text
  ~/.local/python3.12-dev
  ```

- Exposed those local headers to the user service with `CPATH`.
- Found that `optimize=true` worked in a direct single-process script.
- Found that the FastAPI/service path then failed in Torch Inductor CUDA graph
  handling from worker threads.
- Disabled Inductor CUDA graph wrapping for the service while keeping
  `torch.compile` enabled.
- Verified `/v1/admin/models` reports `voxcpm2_optimize: true`.
- Re-ran benchmarks with `optimize=true` and saw meaningful steady-state
  improvement.
- Added a configurable VoxCPM2 warmup suite that runs after model load.

## Current dc2 Runtime Setup

The shared defaults now enable VoxCPM2 optimize and warmup:

```json
{
  "engine": {
    "models": {
      "voxcpm2": {
        "voxcpm2_optimize": true,
        "voxcpm2_warmup_enabled": true
      }
    }
  }
}
```

The current user-service drop-in adds local Python headers and disables Inductor
CUDA graphs:

```ini
[Service]
Environment=CPATH=/home/gunnar/.local/python3.12-dev/usr/include/python3.12:/home/gunnar/.local/python3.12-dev/usr/include
Environment=TORCH_BISECT_BACKEND=inductor
Environment=TORCH_BISECT_SUBSYSTEM=cudagraphs
Environment=TORCH_BISECT_MAX=-1
```

That drop-in lives at:

```text
~/.config/systemd/user/tts-pool.service.d/optimize-headers.conf
```

If system packages are acceptable, the cleaner long-term dependency is probably:

```bash
sudo apt-get install python3.12-dev
```

Then the local `CPATH` workaround may no longer be needed. The CUDA-graph
disablement should stay unless we change the execution model or prove a safer
Torch config.

## Benchmarks

These numbers are indicative, not exact. VoxCPM2 output duration varies between
runs, so `realtime_factor` and repeated same-shape tests are more useful than a
single wall-time number.

Before `optimize=true`:

| Case | Client wall | Output audio | RTF |
| --- | ---: | ---: | ---: |
| short, no ref, steps 10 | 556ms | 1.76s | 0.29 |
| short, ref 4s, steps 10 | 440ms | 1.44s | 0.26 |
| medium, no ref, steps 10 | 2007ms | 7.52s | 0.26 |
| medium, ref 4s, steps 10 | 1733ms | 6.40s | 0.26 |
| medium, ref 4s, steps 6 | 1247ms | 5.76s | 0.19 |
| medium, ref 4s, steps 4 | 868ms | 4.80s | 0.16 |
| long, ref 4s, steps 10 | 3153ms | 11.68s | 0.26 |

After `optimize=true`, with CUDA graphs disabled:

| Case | Client wall | Output audio | RTF |
| --- | ---: | ---: | ---: |
| short, no ref, steps 10 | 307ms | 1.44s | 0.19 |
| short, ref 4s, steps 10 | 379ms | 1.60s | 0.20 |
| medium, no ref, steps 10 | 834ms | 4.00s | 0.19 |
| medium, ref 4s, steps 10 | 1064ms | 4.80s | 0.20 |
| medium, ref 4s, steps 6 | 1053ms | 5.76s | 0.17 |
| medium, ref 4s, steps 4 | 1062ms | 6.08s | 0.16 |
| long, ref 4s, steps 10 | 1950ms | 9.28s | 0.20 |

Before the custom warmup suite, one first real request shape after service start
took about 5.5s because it triggered extra compile work.

With `voxcpm2_optimize=true` and `voxcpm2_warmup_enabled=true`, one measured
restart reached a usable `voxcpm2` model state in about 42.2s. A quick
post-warmup check then returned:

| Case | Client wall | Output audio | RTF |
| --- | ---: | ---: | ---: |
| short, no ref, steps 10 | 351ms | 1.60s | 0.21 |
| medium, ref 4s, voice_and_pace, steps 10 | 721ms | 3.68s | 0.20 |

## Warmup Strategy

Starting `tts-pool` is not enough. VoxCPM2's built-in warmup covers only a small
path, so the service also needs warmup calls that exercise realistic request
shapes.

`tts-pool` now supports a first-class VoxCPM2 warmup switch:

```json
{
  "engine": {
    "models": {
      "voxcpm2": {
        "voxcpm2_warmup_enabled": true
      }
    }
  }
}
```

When enabled without custom cases, the service runs a bounded default suite after
the model loads and before the model is reported as loaded. The default suite
covers short text without reference audio, short text with reference audio, and
medium text with reference audio at `10`, `6`, and `4` inference steps.

Custom cases can be set with `voxcpm2_warmup_cases`:

```json
{
  "engine": {
    "models": {
      "voxcpm2": {
        "voxcpm2_warmup_enabled": true,
        "voxcpm2_warmup_cases": [
          {
            "name": "medium_ref_6",
            "text": "Let's see if this works clearly enough for live translation.",
            "reference_audio": true,
            "reference_audio_match": "voice_and_pace",
            "reference_duration_s": 4.0,
            "voice_preset": "configured",
            "cfg_value": 2.0,
            "inference_timesteps": 6
          }
        ]
      }
    }
  }
}
```

Warmup should be shape-based, not language-based:

- short text without reference audio;
- short text with reference audio;
- medium text with reference audio;
- longer text with reference audio;
- each `inference_timesteps` value we expose or use, likely `10`, `6`, and `4`;
- the prompt/control modes we actually use:
  - `preset=configured`;
  - optional voice description preset;
  - `reference_audio_match=voice`;
  - `reference_audio_match=voice_and_pace` if used by clients.

Language-specific warmup is only useful if the effective control prompt or text
shape differs enough to compile a new path. With the current integration, text
length/tokenization matters more than the `language` metadata field.

Current implementation:

- reads an optional configured warmup list under `engine.models.voxcpm2`;
- runs warmups after model load and before reporting the model as loaded;
- uses synthetic short WAV reference audio generated in-process;
- records warmup timing in logs;
- keeps the default warmup suite bounded so service restart time is predictable.

## Next Optimization Ideas

- Add admin exposure for last warmup timings.
- Add metrics for reference encode time, generated patch count, and first
  request compile/warmup time.
- Decide safe `inference_timesteps` presets after listening tests.
- Test `torch.set_float32_matmul_precision("high")`; PyTorch warns that TF32
  tensor cores are available and not enabled.
- Consider keeping `reference_max_duration_s` low, but do not expect major
  latency gains from that alone.
- Consider prompt/reference cache reuse if we later synthesize multiple chunks
  from the same reference sample.
- Implement streaming or chunked TTS for UX. This is likely more important than
  more raw backend speed, because the current `/v1/responses` path only returns
  once the full WAV is generated.

## Risks And Caveats

- `optimize=true` is runtime behavior. A service restart can trigger compile
  work again.
- New shapes can still compile after startup unless we warm them.
- Disabling CUDA graphs avoids the current FastAPI worker-thread crash, but may
  leave some theoretical speed on the table.
- Benchmarks are noisy because generated audio length varies.
- Lower `inference_timesteps` may sound worse; measure quality by listening,
  not just wall time.
