from __future__ import annotations

from app.config import AppSettings

from .common import ModelStateError
from .common import RequestAdmissionError
from .common import UnknownModelError
from .router import TTSRouterEngine


def build_engine(settings: AppSettings) -> TTSRouterEngine:
    return TTSRouterEngine(settings)


__all__ = [
    "ModelStateError",
    "RequestAdmissionError",
    "TTSRouterEngine",
    "UnknownModelError",
    "build_engine",
]
