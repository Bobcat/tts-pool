from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException

from app.config import load_settings
from app.engine import build_engine
from app.engine import ModelStateError
from app.engine import UnknownModelError
from app.grpc_api.server import GrpcServer
from app.schemas import AdminGpuMemoryEnvelope
from app.schemas import AdminLoadRequest
from app.schemas import AdminModelEntry
from app.schemas import AdminModelsEnvelope


def create_app(settings_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(settings_path)
    engine = build_engine(settings)
    grpc_server = GrpcServer(settings=settings.service.grpc, engine=engine) if settings.service.grpc.enabled else None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            if grpc_server is not None:
                await grpc_server.start()
            yield
        finally:
            if grpc_server is not None:
                await grpc_server.stop()
            close = getattr(engine, "close", None)
            if callable(close):
                close()

    app = FastAPI(title="TTS Pool API", lifespan=lifespan)
    app.state.grpc_server = grpc_server

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
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "model_unload_failed", "model": model_name, "message": str(exc)},
            ) from exc

    return app


app = create_app()
