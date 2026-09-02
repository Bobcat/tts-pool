from __future__ import annotations

import argparse
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen
import wave

import grpc
from google.protobuf.json_format import MessageToDict

from app.grpc_api.v1 import tts_pb2
from app.grpc_api.v1 import tts_pb2_grpc


DEFAULT_REFERENCE_TEXT = (
    "This is a short reference recording for a repeatable text to speech benchmark."
)
DEFAULT_TARGET_TEXT = (
    "The scheduler should share available speech synthesis capacity while keeping every slot busy."
)


@dataclass(frozen=True)
class _GrpcSynthesisResult:
    pcm: bytes
    sample_rate_hz: int
    channel_count: int
    duration_ms: int
    metrics: dict[str, Any]

    def wav_bytes(self) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(self.channel_count)
            writer.setsampwidth(2)
            writer.setframerate(self.sample_rate_hz)
            writer.writeframes(self.pcm)
        return output.getvalue()


class _GrpcClient:
    def __init__(self, channel: grpc.Channel) -> None:
        self._stub = tts_pb2_grpc.TTSServiceStub(channel)

    def synthesize(
        self,
        request: tts_pb2.SynthesisRequest,
        *,
        timeout_s: float,
    ) -> _GrpcSynthesisResult:
        started: tts_pb2.Started | None = None
        completed: tts_pb2.Completed | None = None
        chunks: list[bytes] = []
        next_sequence = 0
        next_sample = 0
        for event in self._stub.Synthesize(request, timeout=timeout_s):
            kind = event.WhichOneof("payload")
            if completed is not None:
                raise RuntimeError("gRPC event received after Completed")
            if kind == "started":
                if started is not None or chunks:
                    raise RuntimeError("gRPC Started event is out of order")
                if event.started.encoding != tts_pb2.AUDIO_ENCODING_PCM_S16LE:
                    raise RuntimeError("gRPC stream did not return PCM S16LE")
                started = event.started
            elif kind == "audio_chunk":
                if started is None:
                    raise RuntimeError("gRPC audio arrived before Started")
                chunk = event.audio_chunk
                if chunk.sequence_number != next_sequence or chunk.first_sample != next_sample:
                    raise RuntimeError("gRPC audio chunk is out of order")
                frame_bytes = int(started.channel_count) * 2
                if frame_bytes <= 0 or len(chunk.pcm) % frame_bytes:
                    raise RuntimeError("gRPC audio chunk is not frame-aligned")
                chunks.append(bytes(chunk.pcm))
                next_sequence += 1
                next_sample += len(chunk.pcm) // frame_bytes
            elif kind == "completed":
                completed = event.completed
            else:
                raise RuntimeError("gRPC stream returned an empty event")
        if started is None or completed is None:
            raise RuntimeError("gRPC stream ended without terminal metadata")
        if completed.chunk_count != next_sequence or completed.total_sample_count != next_sample:
            raise RuntimeError("gRPC terminal counts do not match received audio")
        return _GrpcSynthesisResult(
            pcm=b"".join(chunks),
            sample_rate_hz=int(started.sample_rate_hz),
            channel_count=int(started.channel_count),
            duration_ms=int(completed.duration_ms),
            metrics=MessageToDict(completed.metrics, preserving_proto_field_name=True),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark tts-pool NanoVLLM concurrency and scheduler fairness through gRPC.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--grpc-target", default="127.0.0.1:8021")
    parser.add_argument("--model", default="nanovllm_voxcpm")
    parser.add_argument("--reference-model", default="kokoro")
    parser.add_argument("--reference-preset", default="af_heart")
    parser.add_argument("--concurrencies", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fairness-capacity", type=int, default=0)
    parser.add_argument("--weighted-check", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max-generate-length", type=int, default=64)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    concurrencies = _parse_positive_ints(args.concurrencies)
    result = run_benchmark(args, concurrencies=concurrencies)
    rendered = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    failed_checks = [
        name
        for name in ("fair_progress", "weighted_check")
        if result[name] is not None and not result[name]["passed"]
    ]
    if failed_checks:
        raise SystemExit(f"benchmark checks failed: {', '.join(failed_checks)}")


def run_benchmark(args: argparse.Namespace, *, concurrencies: list[int]) -> dict[str, Any]:
    base_url = str(args.base_url).rstrip("/")
    admin_before = _model_admin_entry(base_url, args.model, timeout_s=args.timeout_s)
    if admin_before.get("runtime_state") != "loaded":
        raise RuntimeError(f"model {args.model!r} is not loaded: {admin_before}")

    gpu_sampler = _GpuSampler()
    gpu_sampler.start()
    started_at = time.time()
    try:
        with grpc.insecure_channel(str(args.grpc_target)) as channel:
            tts_client = _GrpcClient(channel)
            reference = _create_reference(
                tts_client=tts_client,
                model=args.reference_model,
                preset=args.reference_preset,
                timeout_s=args.timeout_s,
            )
            _warm_request_shapes(tts_client=tts_client, args=args, reference=reference)

            batches: list[dict[str, Any]] = []
            for case_name, case_reference in (
                ("no_reference", None),
                ("paired_reference", reference),
            ):
                for concurrency in concurrencies:
                    for repeat in range(1, args.repeats + 1):
                        batches.append(
                            _run_batch(
                                tts_client=tts_client,
                                args=args,
                                case_name=case_name,
                                reference=case_reference,
                                concurrency=concurrency,
                                repeat=repeat,
                            )
                        )

            fairness = None
            if args.fairness_capacity > 0:
                fairness = _run_fair_progress_check(
                    base_url=base_url,
                    tts_client=tts_client,
                    args=args,
                    reference=reference,
                    capacity=args.fairness_capacity,
                )

            weighted = None
            if args.weighted_check:
                weighted = _run_weighted_check(
                    base_url=base_url,
                    tts_client=tts_client,
                    args=args,
                    reference=reference,
                )
    finally:
        gpu_sampler.stop()

    admin_after = _model_admin_entry(base_url, args.model, timeout_s=args.timeout_s)
    return {
        "schema_version": 1,
        "started_at_unix_s": started_at,
        "finished_at_unix_s": time.time(),
        "base_url": base_url,
        "grpc_target": str(args.grpc_target),
        "model": args.model,
        "concurrencies": concurrencies,
        "repeats": args.repeats,
        "reference": {
            "model": args.reference_model,
            "preset": args.reference_preset,
            "text": DEFAULT_REFERENCE_TEXT,
            "duration_ms": reference["duration_ms"],
            "wav_bytes": reference["wav_bytes"],
        },
        "admin_before": admin_before,
        "admin_after": admin_after,
        "gpu_memory_mib": gpu_sampler.summary(),
        "batches": batches,
        "fair_progress": fairness,
        "weighted_check": weighted,
    }


def _create_reference(
    *,
    tts_client: _GrpcClient,
    model: str,
    preset: str,
    timeout_s: float,
) -> dict[str, Any]:
    response = tts_client.synthesize(
        tts_pb2.SynthesisRequest(
            model=model,
            input=DEFAULT_REFERENCE_TEXT,
            language="English",
            fairness_key="bench-reference",
            voice=tts_pb2.VoiceSpec(preset=preset),
            generation=tts_pb2.GenerationParams(
                kokoro=tts_pb2.KokoroGenerationParams(speed=1.0),
            ),
            output_encoding=tts_pb2.AUDIO_ENCODING_PCM_S16LE,
        ),
        timeout_s=timeout_s,
    )
    wav_bytes = response.wav_bytes()
    return {
        "mime_type": "audio/wav",
        "data": wav_bytes,
        "duration_ms": response.duration_ms,
        "wav_bytes": len(wav_bytes),
        "prompt_text": DEFAULT_REFERENCE_TEXT,
    }


def _warm_request_shapes(
    *,
    tts_client: _GrpcClient,
    args: argparse.Namespace,
    reference: dict[str, Any],
) -> None:
    for case_reference in (None, reference):
        _synthesize(
            tts_client=tts_client,
            args=args,
            label="warmup",
            fairness_key="bench-warmup",
            reference=case_reference,
            target_text=DEFAULT_TARGET_TEXT,
        )


def _run_batch(
    *,
    tts_client: _GrpcClient,
    args: argparse.Namespace,
    case_name: str,
    reference: dict[str, Any] | None,
    concurrency: int,
    repeat: int,
) -> dict[str, Any]:
    batch_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _synthesize,
                tts_client=tts_client,
                args=args,
                label=f"{case_name}-c{concurrency}-r{repeat}-j{index}",
                fairness_key="bench-throughput",
                reference=reference,
                target_text=DEFAULT_TARGET_TEXT,
            )
            for index in range(concurrency)
        ]
        results = [future.result() for future in futures]
    batch_wall_ms = (time.perf_counter() - batch_started) * 1000.0
    audio_seconds = sum(float(item["metrics"].get("output_audio_seconds") or 0.0) for item in results)
    metric_names = (
        "engine_queue_wait_ms",
        "backend_synthesis_wall_ms",
        "nanovllm_reference_encode_wall_ms",
        "nanovllm_first_chunk_wall_ms",
        "nanovllm_generate_wall_ms",
        "pool_total_wall_ms",
        "realtime_factor",
        "output_audio_seconds",
    )
    return {
        "case": case_name,
        "concurrency": concurrency,
        "repeat": repeat,
        "batch_wall_ms": round(batch_wall_ms, 1),
        "completed_audio_seconds": round(audio_seconds, 3),
        "audio_seconds_per_wall_second": round(
            audio_seconds / (batch_wall_ms / 1000.0),
            3,
        )
        if batch_wall_ms > 0
        else 0.0,
        "metrics": {
            metric_name: _stats(
                [
                    float(item["metrics"][metric_name])
                    for item in results
                    if item["metrics"].get(metric_name) is not None
                ]
            )
            for metric_name in metric_names
        },
        "responses": results,
    }


def _run_fair_progress_check(
    *,
    base_url: str,
    tts_client: _GrpcClient,
    args: argparse.Namespace,
    reference: dict[str, Any],
    capacity: int,
) -> dict[str, Any]:
    scenario_started = time.perf_counter()
    key_a = "bench-fair-a"
    key_b = "bench-fair-b"
    a_jobs = capacity * 2
    snapshots: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=a_jobs + 1) as pool:
        a_futures: list[Future[dict[str, Any]]] = [
            pool.submit(
                _synthesize,
                tts_client=tts_client,
                args=args,
                label=f"fair-a-{index}",
                fairness_key=key_a,
                reference=reference,
                target_text=DEFAULT_TARGET_TEXT * 2,
                max_generate_length=max(96, args.max_generate_length),
            )
            for index in range(a_jobs)
        ]
        a_full = _wait_for_admin_state(
            base_url=base_url,
            model=args.model,
            timeout_s=args.timeout_s,
            predicate=lambda entry: _key_value(entry, key_a, "active") >= capacity,
            snapshots=snapshots,
        )
        b_future = pool.submit(
            _synthesize,
            tts_client=tts_client,
            args=args,
            label="fair-b",
            fairness_key=key_b,
            reference=reference,
            target_text=DEFAULT_TARGET_TEXT * 2,
            max_generate_length=max(96, args.max_generate_length),
        )
        b_active = _wait_for_admin_state(
            base_url=base_url,
            model=args.model,
            timeout_s=args.timeout_s,
            predicate=lambda entry: _key_value(entry, key_b, "active") >= 1,
            snapshots=snapshots,
            allow_all_done=[*a_futures, b_future],
        )
        results = [future.result() for future in a_futures]
        b_result = b_future.result()

    a_starts = sorted(item["estimated_backend_started_at"] for item in results)
    first_queued_a_start = a_starts[capacity] if len(a_starts) > capacity else None
    b_started_before_queued_a = (
        first_queued_a_start is not None
        and b_result["estimated_backend_started_at"] <= first_queued_a_start + 0.05
    )
    return {
        "capacity": capacity,
        "a_jobs": a_jobs,
        "a_filled_capacity_observed": a_full is not None,
        "b_active_observed": b_active is not None,
        "b_started_before_first_queued_a": b_started_before_queued_a,
        "passed": bool(a_full is not None and (b_active is not None or b_started_before_queued_a)),
        "scenario_wall_ms": round((time.perf_counter() - scenario_started) * 1000.0, 1),
        "b_queue_wait_ms": b_result["metrics"].get("engine_queue_wait_ms"),
        "b_start_offset_ms": round(
            (b_result["estimated_backend_started_at"] - scenario_started) * 1000.0,
            1,
        ),
        "first_queued_a_start_offset_ms": round(
            (first_queued_a_start - scenario_started) * 1000.0,
            1,
        )
        if first_queued_a_start is not None
        else None,
        "snapshots": snapshots,
    }


def _run_weighted_check(
    *,
    base_url: str,
    tts_client: _GrpcClient,
    args: argparse.Namespace,
    reference: dict[str, Any],
) -> dict[str, Any]:
    key_a = "bench-weight-a"
    key_b = "bench-weight-b"
    jobs: list[tuple[str, int]] = [
        (key, index)
        for index in range(4)
        for key in (key_a, key_b)
    ]
    snapshots: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [
            pool.submit(
                _synthesize,
                tts_client=tts_client,
                args=args,
                label=f"weight-{key}-{index}",
                fairness_key=key,
                reference=reference,
                target_text=DEFAULT_TARGET_TEXT * 2,
                max_generate_length=max(96, args.max_generate_length),
            )
            for key, index in jobs
        ]
        _wait_for_admin_state(
            base_url=base_url,
            model=args.model,
            timeout_s=args.timeout_s,
            predicate=lambda entry: len(entry.get("fairness", {}).get("keys", [])) >= 2,
            snapshots=snapshots,
            allow_all_done=futures,
        )
        results = [future.result() for future in futures]

    start_order = [
        item["fairness_key"]
        for item in sorted(results, key=lambda item: item["estimated_backend_started_at"])
    ]
    prefix = start_order[:6]
    observed_weights: dict[str, float] = {}
    for snapshot in snapshots:
        for key_entry in snapshot.get("fairness", {}).get("keys", []):
            fairness_key = key_entry.get("fairness_key")
            if fairness_key in {key_a, key_b}:
                observed_weights[str(fairness_key)] = float(key_entry["weight"])
    return {
        "jobs_per_key": 4,
        "expected_weights": {key_a: 2.0, key_b: 1.0},
        "observed_weights": observed_weights,
        "start_order": start_order,
        "first_six": prefix,
        "first_six_counts": {key_a: prefix.count(key_a), key_b: prefix.count(key_b)},
        "passed": (
            observed_weights.get(key_a) == 2.0
            and observed_weights.get(key_b) == 1.0
            and prefix.count(key_a) >= 4
        ),
        "snapshots": snapshots,
    }


def _synthesize(
    *,
    tts_client: _GrpcClient,
    args: argparse.Namespace,
    label: str,
    fairness_key: str,
    reference: dict[str, Any] | None,
    target_text: str,
    max_generate_length: int | None = None,
) -> dict[str, Any]:
    voice = tts_pb2.VoiceSpec(
        instructions="Speak in English with a clear, natural voice.",
    )
    if reference is not None:
        voice.reference_audio.CopyFrom(
            tts_pb2.ReferenceAudio(
                mime_type=reference["mime_type"],
                data=reference["data"],
                max_duration_s=8.0,
                prompt_text=reference["prompt_text"],
                also_use_as_reference=True,
            )
        )
    submitted_at = time.perf_counter()
    response = tts_client.synthesize(
        tts_pb2.SynthesisRequest(
            model=args.model,
            input=target_text,
            language="English",
            fairness_key=fairness_key,
            voice=voice,
            generation=tts_pb2.GenerationParams(
                nanovllm_voxcpm=tts_pb2.NanoVLLMVoxCPMGenerationParams(
                    temperature=args.temperature,
                    max_generate_length=max_generate_length or args.max_generate_length,
                ),
            ),
            output_encoding=tts_pb2.AUDIO_ENCODING_PCM_S16LE,
        ),
        timeout_s=args.timeout_s,
    )
    finished_at = time.perf_counter()
    metrics = response.metrics
    queue_wait_s = float(metrics.get("engine_queue_wait_ms") or 0.0) / 1000.0
    return {
        "label": label,
        "fairness_key": fairness_key,
        "client_wall_ms": round((finished_at - submitted_at) * 1000.0, 1),
        "estimated_backend_started_at": submitted_at + queue_wait_s,
        "audio_bytes": len(response.pcm),
        "duration_ms": response.duration_ms,
        "metrics": metrics,
    }


def _wait_for_admin_state(
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    predicate: Any,
    snapshots: list[dict[str, Any]],
    allow_all_done: list[Future[Any]] | None = None,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        entry = _model_admin_entry(base_url, model, timeout_s=min(10.0, timeout_s))
        state = {
            "runtime_inflight": entry.get("runtime_inflight"),
            "queue_depth": entry.get("queue_depth"),
            "fairness": entry.get("fairness"),
        }
        previous_state = None
        if snapshots:
            previous_state = {
                key: snapshots[-1].get(key)
                for key in ("runtime_inflight", "queue_depth", "fairness")
            }
        if state != previous_state:
            snapshots.append(
                {
                    "observed_at_monotonic_ms": round(time.perf_counter() * 1000.0, 1),
                    **state,
                }
            )
        if predicate(entry):
            return entry
        if allow_all_done is not None and all(future.done() for future in allow_all_done):
            return None
        time.sleep(0.01)
    return None


def _key_value(entry: dict[str, Any], fairness_key: str, field: str) -> int:
    for key_entry in entry.get("fairness", {}).get("keys", []):
        if key_entry.get("fairness_key") == fairness_key:
            return int(key_entry.get(field) or 0)
    return 0


def _model_admin_entry(base_url: str, model: str, *, timeout_s: float) -> dict[str, Any]:
    payload = _get_json(f"{base_url}/v1/admin/models", timeout_s=timeout_s)
    for entry in payload.get("models", []):
        if entry.get("name") == model:
            return dict(entry)
    raise RuntimeError(f"model {model!r} is missing from /v1/admin/models")


def _get_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    return _request_json(Request(url, method="GET"), timeout_s=timeout_s)


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _request_json(request, timeout_s=timeout_s)


def _request_json(request: Request, *, timeout_s: float) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {request.full_url}: {body}") from exc
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object from {request.full_url}")
    return payload


def _parse_positive_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(item < 1 for item in parsed):
        raise SystemExit("--concurrencies must contain positive integers")
    return parsed


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "max": round(max(values), 3),
    }


@dataclass
class _GpuSampler:
    interval_s: float = 0.1

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._samples: list[int] = []
        self._thread = threading.Thread(target=self._run, name="tts-pool-bench-gpu", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, int | None]:
        if not self._samples:
            return {"first": None, "minimum": None, "maximum": None, "last": None}
        return {
            "first": self._samples[0],
            "minimum": min(self._samples),
            "maximum": max(self._samples),
            "last": self._samples[-1],
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            used = _gpu_used_memory_mib()
            if used is not None:
                self._samples.append(used)
            self._stop.wait(self.interval_s)


def _gpu_used_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.splitlines()[0].strip())
    except Exception:
        return None


if __name__ == "__main__":
    main()
