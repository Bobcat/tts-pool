from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
import json
import math
from pathlib import Path
import statistics
import struct
import time
from typing import Any


GRPC_METHOD = "/tts.transport.Benchmark/Synthesize"
DEFAULT_H2_PORT = 18020
DEFAULT_GRPC_PORT = 18021
_AUDIO = b"A"
_TERMINAL = b"T"
_REQUEST_HEADER = struct.Struct("!I")
_SEQUENCE = struct.Struct("!I")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw HTTP/2 and grpc.aio server streaming.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("--bind", default="0.0.0.0")
    server_parser.add_argument("--h2-port", type=int, default=DEFAULT_H2_PORT)
    server_parser.add_argument("--grpc-port", type=int, default=DEFAULT_GRPC_PORT)

    client_parser = subparsers.add_parser("client")
    client_parser.add_argument("--host", default="127.0.0.1")
    client_parser.add_argument("--h2-port", type=int, default=DEFAULT_H2_PORT)
    client_parser.add_argument("--grpc-port", type=int, default=DEFAULT_GRPC_PORT)
    client_parser.add_argument("--concurrencies", default="1,2,4")
    client_parser.add_argument("--repeats", type=int, default=20)
    client_parser.add_argument("--warmups", type=int, default=3)
    client_parser.add_argument("--first-delay-ms", type=float, default=65.0)
    client_parser.add_argument("--chunks", type=int, default=32)
    client_parser.add_argument("--chunk-bytes", type=int, default=15_360)
    client_parser.add_argument("--interval-ms", type=float, default=28.0)
    client_parser.add_argument("--request-bytes", type=int, default=0)
    client_parser.add_argument("--slow-consumer-ms", type=float, default=0.0)
    client_parser.add_argument("--cancel-after-chunks", type=int, default=0)
    client_parser.add_argument("--batch-pause-ms", type=float, default=0.0)
    client_parser.add_argument(
        "--connection-mode",
        choices=("warm", "cold"),
        default="warm",
    )
    client_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "server":
        try:
            asyncio.run(_serve(args))
        except KeyboardInterrupt:
            pass
        return
    result = asyncio.run(_benchmark(args))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _require_dependencies() -> tuple[Any, Any]:
    try:
        import grpc
        import h2
    except ImportError as exc:
        raise SystemExit("Install benchmark dependencies with: pip install grpcio h2") from exc
    return grpc, h2


def _parse_concurrencies(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise SystemExit("--concurrencies must contain positive integers")
    return values


def _request_payload(spec: dict[str, int | float], request_bytes: int) -> bytes:
    encoded = json.dumps(spec, separators=(",", ":"), sort_keys=True).encode("utf-8")
    prefix = _REQUEST_HEADER.pack(len(encoded)) + encoded
    return prefix + bytes(max(0, request_bytes - len(prefix)))


def _parse_request(payload: bytes) -> dict[str, int | float]:
    if len(payload) < _REQUEST_HEADER.size:
        raise ValueError("request header is missing")
    (size,) = _REQUEST_HEADER.unpack_from(payload)
    start = _REQUEST_HEADER.size
    end = start + size
    if end > len(payload):
        raise ValueError("request metadata is incomplete")
    parsed = json.loads(payload[start:end])
    return {
        "first_delay_ms": float(parsed["first_delay_ms"]),
        "chunks": int(parsed["chunks"]),
        "chunk_bytes": int(parsed["chunk_bytes"]),
        "interval_ms": float(parsed["interval_ms"]),
    }


async def _synthetic_frames(payload: bytes) -> AsyncIterator[bytes]:
    spec = _parse_request(payload)
    started = time.perf_counter()
    await asyncio.sleep(spec["first_delay_ms"] / 1000.0)
    first_ready = time.perf_counter()
    chunk = bytes(spec["chunk_bytes"])
    for sequence in range(spec["chunks"]):
        yield _AUDIO + _SEQUENCE.pack(sequence) + chunk
        if sequence + 1 < spec["chunks"]:
            await asyncio.sleep(spec["interval_ms"] / 1000.0)
    completed = time.perf_counter()
    terminal = {
        "server_first_ready_ms": (first_ready - started) * 1000.0,
        "server_total_ms": (completed - started) * 1000.0,
    }
    yield _TERMINAL + json.dumps(terminal, separators=(",", ":")).encode("utf-8")


async def _serve(args: argparse.Namespace) -> None:
    grpc, _ = _require_dependencies()
    h2_server = await asyncio.start_server(
        _handle_h2_connection,
        args.bind,
        args.h2_port,
    )
    grpc_server = grpc.aio.server(
        options=(
            ("grpc.max_send_message_length", 8 * 1024 * 1024),
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),
            ("grpc.max_concurrent_streams", 2_048),
        )
    )
    handler = grpc.unary_stream_rpc_method_handler(
        _grpc_synthesize,
        request_deserializer=lambda value: value,
        response_serializer=lambda value: value,
    )
    grpc_server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler("tts.transport.Benchmark", {"Synthesize": handler}),)
    )
    grpc_server.add_insecure_port(f"{args.bind}:{args.grpc_port}")
    await grpc_server.start()
    sockets = ", ".join(str(sock.getsockname()) for sock in h2_server.sockets or ())
    print(f"raw HTTP/2 listening on {sockets}", flush=True)
    print(f"grpc.aio listening on {args.bind}:{args.grpc_port}", flush=True)
    try:
        await asyncio.gather(h2_server.serve_forever(), grpc_server.wait_for_termination())
    finally:
        h2_server.close()
        await h2_server.wait_closed()
        await grpc_server.stop(grace=1.0)


async def _grpc_synthesize(request: bytes, context: Any) -> AsyncIterator[bytes]:
    del context
    async for frame in _synthetic_frames(request):
        yield frame


class _H2ServerConnection:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        from h2.config import H2Configuration
        from h2.connection import H2Connection

        self.reader = reader
        self.writer = writer
        self.connection = H2Connection(H2Configuration(client_side=False))
        self.lock = asyncio.Lock()
        self.window_updated = asyncio.Event()
        self.bodies: dict[int, bytearray] = defaultdict(bytearray)
        self.tasks: dict[int, asyncio.Task[None]] = {}

    async def run(self) -> None:
        from h2.events import ConnectionTerminated
        from h2.events import DataReceived
        from h2.events import StreamEnded
        from h2.events import StreamReset
        from h2.events import WindowUpdated
        from h2.settings import SettingCodes

        self.connection.initiate_connection()
        self.connection.update_settings(
            {
                SettingCodes.MAX_CONCURRENT_STREAMS: 2_048,
                SettingCodes.INITIAL_WINDOW_SIZE: 1_048_576,
            }
        )
        self.connection.increment_flow_control_window(64 * 1024 * 1024)
        self.writer.write(self.connection.data_to_send())
        await self.writer.drain()
        try:
            while data := await self.reader.read(65_535):
                async with self.lock:
                    events = self.connection.receive_data(data)
                    for event in events:
                        if isinstance(event, DataReceived):
                            self.bodies[event.stream_id].extend(event.data)
                            self.connection.acknowledge_received_data(
                                event.flow_controlled_length,
                                event.stream_id,
                            )
                        elif isinstance(event, WindowUpdated):
                            self.window_updated.set()
                        elif isinstance(event, StreamReset):
                            task = self.tasks.pop(event.stream_id, None)
                            if task:
                                task.cancel()
                            self.bodies.pop(event.stream_id, None)
                        elif isinstance(event, ConnectionTerminated):
                            return
                    outbound = self.connection.data_to_send()
                if outbound:
                    self.writer.write(outbound)
                    await self.writer.drain()
                for event in events:
                    if isinstance(event, StreamEnded):
                        payload = bytes(self.bodies.pop(event.stream_id, b""))
                        task = asyncio.create_task(self._respond(event.stream_id, payload))
                        self.tasks[event.stream_id] = task
        finally:
            for task in self.tasks.values():
                task.cancel()
            if self.tasks:
                await asyncio.gather(*self.tasks.values(), return_exceptions=True)
            self.writer.close()
            await self.writer.wait_closed()

    async def _respond(self, stream_id: int, payload: bytes) -> None:
        try:
            async with self.lock:
                self.connection.send_headers(
                    stream_id,
                    ((":status", "200"), ("content-type", "application/octet-stream")),
                )
                outbound = self.connection.data_to_send()
            if outbound:
                self.writer.write(outbound)
                await self.writer.drain()
            async for frame in _synthetic_frames(payload):
                await self._send_data(stream_id, frame, end_stream=False)
            await self._send_data(stream_id, b"", end_stream=True)
        finally:
            self.tasks.pop(stream_id, None)

    async def _send_data(self, stream_id: int, data: bytes, *, end_stream: bool) -> None:
        offset = 0
        while offset < len(data):
            sent = 0
            async with self.lock:
                window = self.connection.local_flow_control_window(stream_id)
                size = min(window, self.connection.max_outbound_frame_size, len(data) - offset)
                if size > 0:
                    self.connection.send_data(stream_id, data[offset : offset + size])
                    sent = size
                    outbound = self.connection.data_to_send()
                else:
                    self.window_updated.clear()
                    outbound = b""
            if sent:
                self.writer.write(outbound)
                await self.writer.drain()
                offset += sent
            else:
                await self.window_updated.wait()
        if end_stream:
            async with self.lock:
                self.connection.end_stream(stream_id)
                outbound = self.connection.data_to_send()
            self.writer.write(outbound)
            await self.writer.drain()


async def _handle_h2_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    connection = _H2ServerConnection(reader, writer)
    try:
        await connection.run()
    except (ConnectionError, asyncio.IncompleteReadError):
        return


class _H2Client:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.connection: Any = None
        self.lock = asyncio.Lock()
        self.window_updated = asyncio.Event()
        self.queues: dict[
            int,
            asyncio.Queue[tuple[bytes, int] | BaseException | None],
        ] = {}
        self.reader_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        from h2.config import H2Configuration
        from h2.connection import H2Connection
        from h2.settings import SettingCodes

        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        self.connection = H2Connection(H2Configuration(client_side=True))
        self.connection.initiate_connection()
        self.connection.update_settings(
            {
                SettingCodes.MAX_CONCURRENT_STREAMS: 2_048,
                SettingCodes.INITIAL_WINDOW_SIZE: 1_048_576,
            }
        )
        self.connection.increment_flow_control_window(64 * 1024 * 1024)
        self.writer.write(self.connection.data_to_send())
        await self.writer.drain()
        self.reader_task = asyncio.create_task(self._read())

    async def close(self) -> None:
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        if self.reader_task:
            self.reader_task.cancel()
            await asyncio.gather(self.reader_task, return_exceptions=True)

    async def stream(self, payload: bytes) -> AsyncIterator[bytes]:
        if self.connection is None or self.writer is None:
            raise RuntimeError("HTTP/2 client is not connected")
        async with self.lock:
            stream_id = self.connection.get_next_available_stream_id()
            queue: asyncio.Queue[tuple[bytes, int] | BaseException | None] = asyncio.Queue()
            self.queues[stream_id] = queue
            self.connection.send_headers(
                stream_id,
                (
                    (":method", "POST"),
                    (":scheme", "http"),
                    (":authority", f"{self.host}:{self.port}"),
                    (":path", "/synthesize"),
                    ("content-type", "application/octet-stream"),
                ),
                end_stream=not payload,
            )
            outbound = self.connection.data_to_send()
        self.writer.write(outbound)
        await self.writer.drain()
        if payload:
            await self._send_data(stream_id, payload, end_stream=True)
        ended = False
        try:
            while True:
                item = await queue.get()
                if item is None:
                    ended = True
                    return
                if isinstance(item, BaseException):
                    raise item
                frame, flow_controlled_length = item
                yield frame
                await self._acknowledge(stream_id, flow_controlled_length)
        finally:
            self.queues.pop(stream_id, None)
            if not ended:
                from h2.errors import ErrorCodes

                async with self.lock:
                    self.connection.reset_stream(stream_id, error_code=ErrorCodes.CANCEL)
                    outbound = self.connection.data_to_send()
                self.writer.write(outbound)
                await self.writer.drain()

    async def _send_data(self, stream_id: int, data: bytes, *, end_stream: bool) -> None:
        if self.writer is None:
            raise RuntimeError("HTTP/2 client is not connected")
        offset = 0
        while offset < len(data):
            sent = 0
            async with self.lock:
                window = self.connection.local_flow_control_window(stream_id)
                size = min(window, self.connection.max_outbound_frame_size, len(data) - offset)
                if size > 0:
                    final = end_stream and offset + size == len(data)
                    self.connection.send_data(
                        stream_id,
                        data[offset : offset + size],
                        end_stream=final,
                    )
                    sent = size
                    outbound = self.connection.data_to_send()
                else:
                    self.window_updated.clear()
                    outbound = b""
            if sent:
                self.writer.write(outbound)
                await self.writer.drain()
                offset += sent
            else:
                await self.window_updated.wait()

    async def _acknowledge(self, stream_id: int, flow_controlled_length: int) -> None:
        if self.writer is None:
            return
        async with self.lock:
            self.connection.acknowledge_received_data(flow_controlled_length, stream_id)
            outbound = self.connection.data_to_send()
        if outbound:
            self.writer.write(outbound)
            await self.writer.drain()

    async def _read(self) -> None:
        from h2.events import ConnectionTerminated
        from h2.events import DataReceived
        from h2.events import StreamEnded
        from h2.events import StreamReset
        from h2.events import WindowUpdated

        if self.reader is None or self.writer is None:
            return
        try:
            while data := await self.reader.read(65_535):
                async with self.lock:
                    events = self.connection.receive_data(data)
                    for event in events:
                        if isinstance(event, DataReceived):
                            queue = self.queues.get(event.stream_id)
                            if queue:
                                queue.put_nowait(
                                    (bytes(event.data), event.flow_controlled_length)
                                )
                        elif isinstance(event, WindowUpdated):
                            self.window_updated.set()
                        elif isinstance(event, StreamEnded):
                            queue = self.queues.get(event.stream_id)
                            if queue:
                                queue.put_nowait(None)
                        elif isinstance(event, StreamReset):
                            queue = self.queues.get(event.stream_id)
                            if queue:
                                queue.put_nowait(RuntimeError(f"stream reset: {event.error_code}"))
                        elif isinstance(event, ConnectionTerminated):
                            raise ConnectionError(f"HTTP/2 connection terminated: {event.error_code}")
                    outbound = self.connection.data_to_send()
                if outbound:
                    self.writer.write(outbound)
                    await self.writer.drain()
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                for queue in self.queues.values():
                    queue.put_nowait(exc)
            raise


class _GrpcClient:
    def __init__(self, host: str, port: int) -> None:
        grpc, _ = _require_dependencies()
        self.channel = grpc.aio.insecure_channel(
            f"{host}:{port}",
            options=(
                ("grpc.max_send_message_length", 8 * 1024 * 1024),
                ("grpc.max_receive_message_length", 8 * 1024 * 1024),
            ),
        )
        self.call = self.channel.unary_stream(
            GRPC_METHOD,
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )

    async def connect(self) -> None:
        await self.channel.channel_ready()

    async def close(self) -> None:
        await self.channel.close()

    async def stream(self, payload: bytes) -> AsyncIterator[bytes]:
        call = self.call(payload)
        try:
            async for frame in call:
                yield frame
        finally:
            if not call.done():
                call.cancel()


async def _measure_one(
    client: _H2Client | _GrpcClient,
    payload: bytes,
    *,
    slow_consumer_ms: float,
    cancel_after_chunks: int,
) -> dict[str, float | int | bool | None]:
    started = time.perf_counter()
    first_received: float | None = None
    audio_frames = 0
    audio_bytes = 0
    terminal: dict[str, float] = {}
    cancelled = False
    stream = client.stream(payload)
    try:
        async for frame in stream:
            now = time.perf_counter()
            if frame.startswith(_AUDIO):
                if first_received is None:
                    first_received = now
                audio_frames += 1
                audio_bytes += max(0, len(frame) - 1 - _SEQUENCE.size)
                if slow_consumer_ms > 0:
                    await asyncio.sleep(slow_consumer_ms / 1000.0)
                if cancel_after_chunks and audio_frames >= cancel_after_chunks:
                    cancelled = True
                    break
            elif frame.startswith(_TERMINAL):
                terminal = json.loads(frame[1:])
    finally:
        await stream.aclose()
    completed = time.perf_counter()
    if first_received is None:
        raise RuntimeError("stream returned no audio")
    ttfb_ms = (first_received - started) * 1000.0
    first_ready_ms = terminal.get("server_first_ready_ms")
    return {
        "ttfb_ms": ttfb_ms,
        "total_ms": (completed - started) * 1000.0,
        "transport_after_ready_ms": (
            ttfb_ms - float(first_ready_ms) if first_ready_ms is not None else None
        ),
        "server_first_ready_ms": float(first_ready_ms) if first_ready_ms is not None else None,
        "server_total_ms": (
            float(terminal["server_total_ms"]) if "server_total_ms" in terminal else None
        ),
        "audio_frames": audio_frames,
        "audio_bytes": audio_bytes,
        "slow_consumer_ms": slow_consumer_ms,
        "cancelled": cancelled,
    }


async def _run_batch(
    client: _H2Client | _GrpcClient,
    payload: bytes,
    *,
    concurrency: int,
    slow_consumer_ms: float,
    cancel_after_chunks: int,
) -> list[dict[str, float | int | bool | None]]:
    return await asyncio.gather(
        *(
            _measure_one(
                client,
                payload,
                slow_consumer_ms=slow_consumer_ms if index == 0 else 0.0,
                cancel_after_chunks=cancel_after_chunks,
            )
            for index in range(concurrency)
        )
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[rank]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "p99": round(_percentile(values, 0.99), 3),
        "max": round(max(values), 3),
    }


async def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _require_dependencies()
    if args.repeats < 1 or args.warmups < 0:
        raise SystemExit("--repeats must be positive and --warmups cannot be negative")
    spec: dict[str, int | float] = {
        "first_delay_ms": args.first_delay_ms,
        "chunks": args.chunks,
        "chunk_bytes": args.chunk_bytes,
        "interval_ms": args.interval_ms,
    }
    payload = _request_payload(spec, args.request_bytes)
    protocols = ("h2", "grpc")

    def new_client(protocol: str) -> _H2Client | _GrpcClient:
        if protocol == "h2":
            return _H2Client(args.host, args.h2_port)
        return _GrpcClient(args.host, args.grpc_port)

    clients = (
        {protocol: new_client(protocol) for protocol in protocols}
        if args.connection_mode == "warm"
        else {}
    )
    if args.connection_mode == "warm":
        for client in clients.values():
            await client.connect()

    async def run_protocol(
        protocol: str,
        concurrency: int,
    ) -> list[dict[str, float | int | bool | None]]:
        if args.connection_mode == "cold":
            client = new_client(protocol)
            await client.connect()
        else:
            client = clients[protocol]
        try:
            return await _run_batch(
                client,
                payload,
                concurrency=concurrency,
                slow_consumer_ms=args.slow_consumer_ms,
                cancel_after_chunks=args.cancel_after_chunks,
            )
        finally:
            if args.connection_mode == "cold":
                await client.close()

    try:
        for _ in range(args.warmups):
            for protocol in protocols:
                await run_protocol(protocol, 1)

        records: list[dict[str, Any]] = []
        for concurrency in _parse_concurrencies(args.concurrencies):
            for repeat in range(args.repeats):
                order = ("h2", "grpc") if repeat % 2 == 0 else ("grpc", "h2")
                for protocol in order:
                    results = await run_protocol(protocol, concurrency)
                    records.extend(
                        {
                            "protocol": protocol,
                            "concurrency": concurrency,
                            "repeat": repeat + 1,
                            **result,
                        }
                        for result in results
                    )
                    if args.batch_pause_ms > 0:
                        await asyncio.sleep(args.batch_pause_ms / 1000.0)
    finally:
        if args.connection_mode == "warm":
            for client in clients.values():
                await client.close()

    summaries: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for protocol in protocols:
        summaries[protocol] = {}
        for concurrency in _parse_concurrencies(args.concurrencies):
            selected = [
                record
                for record in records
                if record["protocol"] == protocol and record["concurrency"] == concurrency
            ]
            metric_summaries: dict[str, dict[str, float]] = {}
            for metric in ("ttfb_ms", "transport_after_ready_ms", "total_ms"):
                values = [
                    float(record[metric])
                    for record in selected
                    if record[metric] is not None
                ]
                if values:
                    metric_summaries[metric] = _summary(values)
            summaries[protocol][str(concurrency)] = metric_summaries
            if args.slow_consumer_ms > 0:
                normal = [record for record in selected if not record["slow_consumer_ms"]]
                slow = [record for record in selected if record["slow_consumer_ms"]]
                if normal:
                    summaries[protocol][str(concurrency)]["normal_total_ms"] = _summary(
                        [float(record["total_ms"]) for record in normal]
                    )
                summaries[protocol][str(concurrency)]["slow_total_ms"] = _summary(
                    [float(record["total_ms"]) for record in slow]
                )
    return {
        "schema_version": 1,
        "host": args.host,
        "h2_port": args.h2_port,
        "grpc_port": args.grpc_port,
        "concurrencies": _parse_concurrencies(args.concurrencies),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "connection_mode": args.connection_mode,
        "request_bytes": len(payload),
        "spec": spec,
        "slow_consumer_ms": args.slow_consumer_ms,
        "cancel_after_chunks": args.cancel_after_chunks,
        "batch_pause_ms": args.batch_pause_ms,
        "summaries": summaries,
        "records": records,
    }


if __name__ == "__main__":
    main()
