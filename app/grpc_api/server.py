"""gRPC server lifecycle owned by the TTS-pool process."""

from __future__ import annotations

import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from app.config import GrpcSettings

from .service import TTSService
from .v1 import tts_pb2_grpc


SERVICE_NAME = "tts.v1.TTSService"


class GrpcServer:
    def __init__(self, *, settings: GrpcSettings, engine: object) -> None:
        self._settings = settings
        self._engine = engine
        self._server: grpc.aio.Server | None = None
        self._health: health.aio.HealthServicer | None = None
        self.bound_port: int | None = None

    async def start(self) -> None:
        if self._server is not None:
            return
        server = grpc.aio.server(
            options=(
                ("grpc.max_receive_message_length", self._settings.max_receive_message_bytes),
                ("grpc.max_send_message_length", self._settings.max_send_message_bytes),
            )
        )
        tts_pb2_grpc.add_TTSServiceServicer_to_server(TTSService(self._engine), server)
        health_service = health.aio.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_service, server)
        bound_port = server.add_insecure_port(f"{self._settings.host}:{self._settings.port}")
        if bound_port == 0:
            raise RuntimeError(
                f"could not bind gRPC server to {self._settings.host}:{self._settings.port}"
            )
        await server.start()
        await health_service.set("", health_pb2.HealthCheckResponse.SERVING)
        await health_service.set(SERVICE_NAME, health_pb2.HealthCheckResponse.SERVING)
        self._server = server
        self._health = health_service
        self.bound_port = bound_port

    async def stop(self) -> None:
        server = self._server
        if server is None:
            return
        health_service = self._health
        if health_service is not None:
            await health_service.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
            await health_service.set(SERVICE_NAME, health_pb2.HealthCheckResponse.NOT_SERVING)
        await server.stop(self._settings.shutdown_grace_s)
        self._server = None
        self._health = None
        self.bound_port = None
