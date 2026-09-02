from __future__ import annotations

import threading
import time
from typing import Any
import uuid

from app.config import AppSettings
from app.config import ModelSettings
from app.schemas import AdminLoadRequest
from app.schemas import ResponseRequest

from .common import capabilities_for_backend
from .common import estimate_model_artifact_size_mib
from .common import exception_message
from .common import ModelRuntimeState
from .common import ModelStateError
from .common import model_definition_payload
from .common import query_gpu_memory
from .common import query_primary_gpu_used_mib
from .common import UnknownModelError
from .scheduler import ExecutorSnapshot
from .scheduler import LoadedModelExecutor
from .streaming import SynthesisHandle


class TTSRouterEngine:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._configured_models = dict(settings.engine.models)
        self._fairness_settings = settings.engine.fairness
        if not self._configured_models:
            raise ValueError("no configured models")
        self._runtimes: dict[str, Any] = {}
        self._executors: dict[str, LoadedModelExecutor] = {}
        self._model_states: dict[str, ModelRuntimeState] = {}
        self._state_lock = threading.RLock()
        self._state_changed = threading.Condition(self._state_lock)

        for model_name, model_settings in self._configured_models.items():
            resolved_backend = self._resolve_model_backend(settings.engine.backend, model_settings)
            self._model_states[model_name] = ModelRuntimeState(
                resolved_backend=resolved_backend,
                configured_enabled=model_settings.enabled,
                configured_target_inflight=model_settings.target_inflight,
                effective_target_inflight=model_settings.target_inflight,
            )

        for model_name, model_settings in self._configured_models.items():
            if not model_settings.enabled:
                continue
            try:
                self.load_model(model_name, settings)
            except Exception:
                # The admin API exposes the failed state; startup should still serve inspection endpoints.
                pass

    def close(self) -> None:
        for model_name in list(self._configured_models):
            try:
                self.unload_model(model_name)
            except Exception:
                pass

    def list_models_payload(self) -> dict[str, object]:
        with self._state_lock:
            models = sorted(
                model_name
                for model_name, state in self._model_states.items()
                if state.lifecycle == "loaded"
            )
        return {"models": models}

    def stream(self, request: ResponseRequest) -> SynthesisHandle:
        model_name = str(request.model or "").strip()
        with self._state_lock:
            if model_name not in self._configured_models:
                raise UnknownModelError(model_name)
            state = self._model_states[model_name]
            if state.lifecycle != "loaded":
                raise ModelStateError(model_name, self._lifecycle_error_code(state.lifecycle))
            executor = self._executors.get(model_name)
            if executor is None:
                raise ModelStateError(model_name, "model_not_loaded")
            state.inflight_requests += 1

        grpc_settings = self._settings.service.grpc
        handle = SynthesisHandle(
            response_id=f"ttsresp_{uuid.uuid4().hex}",
            model_name=model_name,
            max_buffer_chunks=grpc_settings.stream_buffer_chunks,
            max_buffer_bytes=grpc_settings.stream_buffer_bytes,
            stalled_consumer_timeout_s=grpc_settings.stalled_consumer_timeout_s,
        )
        handle.add_terminal_callback(lambda _: self._stream_finished(model_name))
        try:
            executor.enqueue_stream(request, handle)
        except Exception:
            self._stream_finished(model_name)
            raise
        return handle

    def admin_models_payload(self, settings: AppSettings | None = None) -> dict[str, object]:
        del settings
        with self._state_lock:
            return {
                "models": [
                    self._admin_model_entry_locked(model_name, model_settings)
                    for model_name, model_settings in self._configured_models.items()
                ]
            }

    def admin_gpu_memory_payload(self, settings: AppSettings | None = None) -> dict[str, object]:
        del settings
        gpus, error = query_gpu_memory()
        with self._state_lock:
            models = [
                self._admin_gpu_model_entry_locked(model_name, model_settings)
                for model_name, model_settings in self._configured_models.items()
            ]
        return {"gpus": gpus, "models": models, "error": error}

    def load_model(
        self,
        model_name: str,
        settings: AppSettings | None = None,
        load_request: AdminLoadRequest | None = None,
    ) -> dict[str, object]:
        del settings
        with self._state_lock:
            model_settings = self._configured_models.get(model_name)
            if model_settings is None:
                raise UnknownModelError(model_name)
            state = self._model_states[model_name]
            if state.lifecycle == "unloading":
                raise ModelStateError(model_name, "model_unloading")
            if state.lifecycle in {"loaded", "loading"}:
                return self._admin_model_entry_locked(model_name, model_settings)
            state.lifecycle = "loading"
            state.last_error = None
            resolved_backend = state.resolved_backend
            target_inflight = (
                model_settings.target_inflight
                if load_request is None or load_request.target_inflight is None
                else max(1, int(load_request.target_inflight))
            )
            state.configured_target_inflight = target_inflight

        gpu_used_before_mib = query_primary_gpu_used_mib()
        try:
            runtime = _build_runtime(
                model_name=model_name,
                model_settings=model_settings,
                resolved_backend=resolved_backend,
            )
            runtime.load()
            maximum_chunk_bytes = getattr(runtime, "max_stream_chunk_bytes", None)
            if (
                maximum_chunk_bytes is not None
                and int(maximum_chunk_bytes) > self._settings.service.grpc.stream_buffer_bytes
            ):
                raise ValueError(
                    "service.grpc.stream_buffer_bytes is smaller than the runtime's "
                    "maximum PCM chunk"
                )
            runtime_capability = int(getattr(runtime, "runtime_capability", 1) or 1)
            executor = LoadedModelExecutor(
                model_name=model_name,
                complete_fn=runtime.synthesize,
                stream_fn=getattr(runtime, "synthesize_stream", None),
                configured_target_inflight=target_inflight,
                fairness_settings=self._fairness_settings,
                runtime_capability=runtime_capability,
            )
            executor.start()
        except Exception as exc:
            message = exception_message(exc)
            with self._state_lock:
                state = self._model_states[model_name]
                state.lifecycle = "failed"
                state.last_error = message
                self._state_changed.notify_all()
            raise RuntimeError(message) from exc

        gpu_used_after_mib = query_primary_gpu_used_mib()
        observed_vram_mib: int | None = None
        if (
            gpu_used_before_mib is not None
            and gpu_used_after_mib is not None
            and gpu_used_after_mib >= gpu_used_before_mib
        ):
            delta = gpu_used_after_mib - gpu_used_before_mib
            if delta > 0:
                observed_vram_mib = delta

        with self._state_lock:
            self._runtimes[model_name] = runtime
            self._executors[model_name] = executor
            state = self._model_states[model_name]
            state.lifecycle = "loaded"
            state.last_error = None
            state.effective_target_inflight = min(target_inflight, int(getattr(runtime, "runtime_capability", 1) or 1))
            if observed_vram_mib is not None:
                state.observed_vram_mib = observed_vram_mib
            self._state_changed.notify_all()
            return self._admin_model_entry_locked(model_name, model_settings)

    def _stream_finished(self, model_name: str) -> None:
        with self._state_lock:
            state = self._model_states.get(model_name)
            if state is not None and state.inflight_requests > 0:
                state.inflight_requests -= 1
                self._state_changed.notify_all()

    def unload_model(self, model_name: str, settings: AppSettings | None = None) -> dict[str, object]:
        del settings
        with self._state_lock:
            model_settings = self._configured_models.get(model_name)
            if model_settings is None:
                raise UnknownModelError(model_name)
            state = self._model_states[model_name]
            if state.lifecycle == "loading":
                raise ModelStateError(model_name, "model_loading")
            if state.lifecycle == "unloaded":
                return self._admin_model_entry_locked(model_name, model_settings)
            if state.lifecycle == "unloading":
                return self._admin_model_entry_locked(model_name, model_settings)
            state.lifecycle = "unloading"
            executor = self._executors.get(model_name)

        if executor is not None:
            executor.begin_shutdown()
        deadline = time.monotonic() + self._settings.service.grpc.shutdown_grace_s
        with self._state_lock:
            while state.inflight_requests > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    state.lifecycle = "failed"
                    state.last_error = "model unload timed out while synthesis was still active"
                    self._state_changed.notify_all()
                    raise RuntimeError(state.last_error)
                self._state_changed.wait(timeout=remaining)
            executor = self._executors.pop(model_name, None)
            runtime = self._runtimes.pop(model_name, None)

        if executor is not None:
            executor.join(timeout=1.0)
        close = getattr(runtime, "close", None)
        if callable(close):
            close()

        with self._state_lock:
            state = self._model_states[model_name]
            state.lifecycle = "unloaded"
            state.last_error = None
            state.effective_target_inflight = state.configured_target_inflight
            self._state_changed.notify_all()
            return self._admin_model_entry_locked(model_name, model_settings)

    def _admin_model_entry_locked(self, model_name: str, model_settings: ModelSettings) -> dict[str, object]:
        state = self._model_states[model_name]
        snapshot = self._executor_snapshot_locked(model_name)
        vram_estimate_mib, vram_estimate_source = self._vram_estimate_locked(model_name, model_settings)
        return {
            "name": model_name,
            "resolved_backend": state.resolved_backend,
            "configured_enabled": state.configured_enabled,
            "runtime_state": state.lifecycle,
            "is_loaded": state.lifecycle == "loaded",
            "inflight_requests": state.inflight_requests,
            "queue_depth": snapshot.queue_depth,
            "runtime_inflight": snapshot.runtime_inflight,
            "configured_target_inflight": state.configured_target_inflight,
            "effective_target_inflight": snapshot.effective_target_inflight,
            "accepting_new_requests": snapshot.accepting_new_requests,
            "fairness": {
                "rejected_per_key_limit": snapshot.fairness_rejected_per_key_limit,
                "rejected_executor_limit": snapshot.fairness_rejected_executor_limit,
                "keys": [
                    {
                        "fairness_key": key_snapshot.fairness_key,
                        "pending": key_snapshot.pending,
                        "active": key_snapshot.active,
                        "weight": key_snapshot.weight,
                        "score": key_snapshot.score,
                        "rejected_per_key_limit": key_snapshot.rejected_per_key_limit,
                        "rejected_executor_limit": key_snapshot.rejected_executor_limit,
                    }
                    for key_snapshot in snapshot.fairness_keys
                ],
            },
            "last_error": state.last_error,
            "vram_estimate_mib": vram_estimate_mib,
            "vram_estimate_source": vram_estimate_source,
            "capabilities": capabilities_for_backend(state.resolved_backend, model_settings),
            "definition": model_definition_payload(model_settings, resolved_backend=state.resolved_backend),
        }

    def _admin_gpu_model_entry_locked(self, model_name: str, model_settings: ModelSettings) -> dict[str, object]:
        state = self._model_states[model_name]
        vram_estimate_mib, vram_estimate_source = self._vram_estimate_locked(model_name, model_settings)
        return {
            "name": model_name,
            "runtime_state": state.lifecycle,
            "is_loaded": state.lifecycle == "loaded",
            "vram_estimate_mib": vram_estimate_mib,
            "vram_estimate_source": vram_estimate_source,
        }

    def _vram_estimate_locked(
        self,
        model_name: str,
        model_settings: ModelSettings,
    ) -> tuple[int | None, str]:
        state = self._model_states[model_name]
        if state.observed_vram_mib is not None:
            return state.observed_vram_mib, "observed_load_delta"
        if state.artifact_size_mib is None:
            state.artifact_size_mib = estimate_model_artifact_size_mib(model_settings.model_path)
        if state.artifact_size_mib is not None:
            return state.artifact_size_mib, "model_artifact_size"
        return None, "unavailable"

    def _executor_snapshot_locked(self, model_name: str) -> ExecutorSnapshot:
        executor = self._executors.get(model_name)
        if executor is not None:
            return executor.snapshot()
        state = self._model_states[model_name]
        return ExecutorSnapshot(
            queue_depth=0,
            runtime_inflight=0,
            configured_target_inflight=state.configured_target_inflight,
            effective_target_inflight=state.effective_target_inflight,
            accepting_new_requests=state.lifecycle == "loaded",
        )

    @staticmethod
    def _resolve_model_backend(global_backend: str, model_settings: ModelSettings) -> str:
        return str(model_settings.backend or global_backend or "stub").strip().lower()

    @staticmethod
    def _lifecycle_error_code(lifecycle: str) -> str:
        return {
            "unloaded": "model_not_loaded",
            "loading": "model_loading",
            "unloading": "model_unloading",
            "failed": "model_failed",
        }.get(lifecycle, "model_not_loaded")


def _build_runtime(*, model_name: str, model_settings: ModelSettings, resolved_backend: str) -> Any:
    if resolved_backend == "stub":
        from .stub import StubTTSRuntime

        return StubTTSRuntime(model_name=model_name, model_settings=model_settings)
    if resolved_backend == "kokoro":
        from .kokoro import KokoroTTSRuntime

        return KokoroTTSRuntime(model_name=model_name, model_settings=model_settings)
    if resolved_backend == "voxcpm2":
        from .voxcpm2 import VoxCPM2TTSRuntime

        return VoxCPM2TTSRuntime(model_name=model_name, model_settings=model_settings)
    if resolved_backend == "nanovllm_voxcpm":
        from .nanovllm_voxcpm import NanoVLLMVoxCPMTTSRuntime

        return NanoVLLMVoxCPMTTSRuntime(model_name=model_name, model_settings=model_settings)
    raise ValueError(f"unsupported backend: {resolved_backend!r}")
