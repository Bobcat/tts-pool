from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import time
from typing import Any
import wave

import numpy as np

from nanovllm_voxcpm import VoxCPM


DEFAULT_MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/"
    / "bffb3df5a29440629464e5e839f4d214c8714c3d"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct NanoVLLM-VoxCPM smoke and concurrency bench.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--max-generate-length", type=int, default=256)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--ref-duration-s", type=float, default=4.0)
    args = parser.parse_args()

    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")

    _print_event(
        "environment",
        {
            "model": str(model_path),
            "gpu_memory_before_mib": _gpu_memory_mib(),
        },
    )

    load_started = time.perf_counter()
    server = VoxCPM.from_pretrained(
        model=str(model_path),
        devices=[args.device],
        inference_timesteps=args.inference_timesteps,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        await server.wait_for_ready()
        load_wall_ms = (time.perf_counter() - load_started) * 1000.0
        model_info = await server.get_model_info()
        _print_event(
            "loaded",
            {
                "load_wall_ms": round(load_wall_ms, 1),
                "model_info": model_info,
                "gpu_memory_after_load_mib": _gpu_memory_mib(),
            },
        )

        ref_wav = _sine_wav(seconds=args.ref_duration_s, sample_rate_hz=int(model_info["encoder_sample_rate"]))
        encode_started = time.perf_counter()
        ref_latents = await server.encode_latents(ref_wav, "wav")
        _print_event(
            "reference_encoded",
            {
                "encode_wall_ms": round((time.perf_counter() - encode_started) * 1000.0, 1),
                "reference_wav_bytes": len(ref_wav),
                "reference_latents_bytes": len(ref_latents),
                "reference_wav_base64_head": base64.b64encode(ref_wav[:24]).decode("ascii"),
            },
        )

        await _single_smoke(server, args, ref_latents)
        await _concurrency_bench(server, args, ref_latents)
    finally:
        await server.stop()
        _print_event("stopped", {"gpu_memory_after_stop_mib": _gpu_memory_mib()})


async def _single_smoke(server: Any, args: argparse.Namespace, ref_latents: bytes) -> None:
    cases = [
        ("short_no_ref", "Let us see if this local NanoVLLM route works.", None),
        ("short_ref", "Let us see if this local NanoVLLM route works with a speech sample.", ref_latents),
    ]
    for name, text, latents in cases:
        result = await _generate_one(server, args, name=name, text=text, ref_latents=latents)
        _print_event("smoke", result)


async def _concurrency_bench(server: Any, args: argparse.Namespace, ref_latents: bytes) -> None:
    concurrencies = [int(item) for item in str(args.concurrency).split(",") if item.strip()]
    for concurrency in concurrencies:
        started = time.perf_counter()
        tasks = [
            asyncio.create_task(
                _generate_one(
                    server,
                    args,
                    name=f"concurrency_{concurrency}_{index}",
                    text=f"Concurrent NanoVLLM request number {index}. Let us see if batching helps.",
                    ref_latents=ref_latents,
                )
            )
            for index in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        wall_ms = (time.perf_counter() - started) * 1000.0
        _print_event(
            "concurrency",
            {
                "concurrency": concurrency,
                "batch_wall_ms": round(wall_ms, 1),
                "ttfb_ms": _stats([item["ttfb_ms"] for item in results]),
                "wall_ms": _stats([item["wall_ms"] for item in results]),
                "audio_seconds": _stats([item["audio_seconds"] for item in results]),
                "rtf_after_ttfb": _stats([item["rtf_after_ttfb"] for item in results]),
                "gpu_memory_mib": _gpu_memory_mib(),
            },
        )


async def _generate_one(
    server: Any,
    args: argparse.Namespace,
    *,
    name: str,
    text: str,
    ref_latents: bytes | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_chunk_at: float | None = None
    samples = 0
    chunks = 0
    async for chunk in server.generate(
        target_text=text,
        max_generate_length=args.max_generate_length,
        cfg_value=args.cfg_value,
        ref_audio_latents=ref_latents,
    ):
        now = time.perf_counter()
        if first_chunk_at is None:
            first_chunk_at = now
        array = np.asarray(chunk, dtype=np.float32).reshape(-1)
        samples += int(array.size)
        chunks += 1
    finished = time.perf_counter()
    ttfb_ms = ((first_chunk_at or finished) - started) * 1000.0
    wall_ms = (finished - started) * 1000.0
    sample_rate = 16000
    info = await server.get_model_info()
    sample_rate = int(info["sample_rate"])
    audio_seconds = samples / sample_rate if sample_rate > 0 else 0.0
    post_ttfb_seconds = max(0.0, (wall_ms - ttfb_ms) / 1000.0)
    return {
        "name": name,
        "reference": ref_latents is not None,
        "chunks": chunks,
        "samples": samples,
        "sample_rate_hz": sample_rate,
        "audio_seconds": round(audio_seconds, 3),
        "ttfb_ms": round(ttfb_ms, 1),
        "wall_ms": round(wall_ms, 1),
        "rtf_after_ttfb": round(post_ttfb_seconds / audio_seconds, 4) if audio_seconds > 0 else 0.0,
    }


def _sine_wav(*, seconds: float, sample_rate_hz: int) -> bytes:
    buffer = io.BytesIO()
    frame_count = max(1, int(seconds * sample_rate_hz))
    amplitude = 0.12 * 32767.0
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        for index in range(frame_count):
            sample = int(amplitude * math.sin(2.0 * math.pi * 220.0 * index / sample_rate_hz))
            writer.writeframesraw(struct.pack("<h", sample))
    return buffer.getvalue()


def _gpu_memory_mib() -> dict[str, int] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    line = result.stdout.splitlines()[0]
    used, total = [int(part.strip()) for part in line.split(",")[:2]]
    return {"used": used, "total": total}


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(float(min(values)), 1),
        "mean": round(float(sum(values) / len(values)), 1),
        "max": round(float(max(values)), 1),
    }


def _print_event(event: str, payload: dict[str, Any]) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
