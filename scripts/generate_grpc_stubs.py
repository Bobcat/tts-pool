#!/usr/bin/env python3
"""Regenerate the checked-in Python bindings from the canonical TTS proto."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import grpc_tools
from grpc_tools import protoc


ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = ROOT / "proto"
PROTO_PATH = PROTO_ROOT / "tts" / "v1" / "tts.proto"
WELL_KNOWN_PROTO_ROOT = Path(grpc_tools.__file__).resolve().parent / "_proto"
OUTPUT_DIR = ROOT / "app" / "grpc_api" / "v1"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        generated_root = Path(tmpdir)
        status = protoc.main(
            [
                "grpc_tools.protoc",
                f"-I{PROTO_ROOT}",
                f"-I{WELL_KNOWN_PROTO_ROOT}",
                f"--python_out={generated_root}",
                f"--grpc_python_out={generated_root}",
                str(PROTO_PATH.relative_to(PROTO_ROOT)),
            ]
        )
        if status != 0:
            raise SystemExit(status)
        generated_dir = generated_root / "tts" / "v1"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated_dir / "tts_pb2.py", OUTPUT_DIR / "tts_pb2.py")
        grpc_source = (generated_dir / "tts_pb2_grpc.py").read_text(encoding="utf-8")
        grpc_source = grpc_source.replace(
            "from tts.v1 import tts_pb2",
            "from . import tts_pb2",
        )
        (OUTPUT_DIR / "tts_pb2_grpc.py").write_text(grpc_source, encoding="utf-8")


if __name__ == "__main__":
    main()
