from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException

from app.config import load_settings
from app.engine import build_engine
from app.engine import ModelStateError
from app.engine import RequestAdmissionError
from app.engine import UnknownModelError
from app.schemas import AdminGpuMemoryEnvelope
from app.schemas import AdminLoadRequest
from app.schemas import AdminModelEntry
from app.schemas import AdminModelsEnvelope
from app.schemas import AudioOutput
from app.schemas import ResponseEnvelope
from app.schemas import ResponseRequest


LOGGER = logging.getLogger("tts_pool.metrics")


def _metrics_payload(metrics: dict[str, float | int | None]) -> dict[str, float | int | None]:
    return dict(metrics)


def _log_inference(response_id: str, request: ResponseRequest, metrics: dict[str, float | int | None]) -> None:
    payload = {
        "event": "tts_pool.inference",
        "request_id": response_id,
        "model": request.model,
        "fairness_key": request.fairness_key,
        "stream": request.stream,
        "metrics": _metrics_payload(metrics),
    }
    LOGGER.info("%s", json.dumps(payload, ensure_ascii=True, sort_keys=True))


def create_app(settings_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(settings_path)
    engine = build_engine(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                close()

    app = FastAPI(title="TTS Pool API", lifespan=lifespan)

    @app.get("/v1/models")
    def list_models() -> dict[str, object]:
        return engine.list_models_payload()

    @app.get(
        "/v1/admin/models",
        response_model=AdminModelsEnvelope,
        tags=["admin"],
        summary="List configured TTS models and runtime state",
    )
    def list_admin_models() -> dict[str, object]:
        return engine.admin_models_payload(settings)

    @app.get(
        "/v1/admin/gpu-memory",
        response_model=AdminGpuMemoryEnvelope,
        tags=["admin"],
        summary="Get GPU memory usage and model artifact estimates",
    )
    def get_gpu_memory() -> dict[str, object]:
        return engine.admin_gpu_memory_payload(settings)

    @app.post(
        "/v1/admin/models/{model_name}/load",
        response_model=AdminModelEntry,
        tags=["admin"],
        summary="Load one configured TTS model",
    )
    def load_model(model_name: str, load_request: AdminLoadRequest | None = None) -> dict[str, object]:
        try:
            return engine.load_model(model_name, settings, load_request)
        except UnknownModelError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_model", "model": model_name},
            ) from exc
        except ModelStateError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": exc.code, "model": model_name},
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "model_load_failed", "model": model_name, "message": str(exc)},
            ) from exc

    @app.post(
        "/v1/admin/models/{model_name}/unload",
        response_model=AdminModelEntry,
        tags=["admin"],
        summary="Unload one loaded TTS model",
    )
    def unload_model(model_name: str) -> dict[str, object]:
        try:
            return engine.unload_model(model_name, settings)
        except UnknownModelError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_model", "model": model_name},
            ) from exc
        except ModelStateError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": exc.code, "model": model_name},
            ) from exc

    @app.post("/v1/responses", response_model=ResponseEnvelope)
    def create_response(request: ResponseRequest) -> ResponseEnvelope:
        if request.stream:
            raise HTTPException(status_code=400, detail={"code": "stream_unsupported"})
        started_at = time.perf_counter()
        response_id = f"ttsresp_{uuid.uuid4().hex}"
        try:
            result = engine.complete(request)
        except UnknownModelError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_model", "model": request.model},
            ) from exc
        except ModelStateError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": exc.code, "model": request.model},
            ) from exc
        except RequestAdmissionError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": exc.message},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_request", "message": str(exc)},
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "synthesis_failed", "model": request.model, "message": str(exc)},
            ) from exc

        metrics = dict(result.metrics)
        metrics["pool_total_wall_ms"] = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        _log_inference(response_id, request, metrics)
        return ResponseEnvelope(
            id=response_id,
            model=request.model,
            audio=AudioOutput(
                mime_type=result.mime_type,
                data_base64=base64.b64encode(result.audio).decode("ascii"),
                sample_rate_hz=result.sample_rate_hz,
                duration_ms=result.duration_ms,
            ),
            metrics=metrics,
            metadata=result.metadata,
        )

    return app


app = create_app()
