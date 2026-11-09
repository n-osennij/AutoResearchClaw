"""MCP transport layer: stdio (newline-delimited JSON-RPC) and SSE stub.

The MCP stdio framing is **one JSON-RPC message per line**, not the LSP
``Content-Length`` header block this module used to emit. An LSP frame makes
every spec-compliant client (pi, Claude, Codex) hang on the first read, so the
old implementation was never usable.

``StdioTransport`` takes optional reader/writer objects, which keeps it
testable without touching real stdin/stdout. Reads run in a worker thread:
``sys.stdin.buffer`` is a blocking pipe on Windows and cannot be driven by
asyncio stream pipes there.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class MCPTransport(Protocol):
    """Protocol for MCP message transport."""

    async def send(self, message: dict[str, Any]) -> None: ...
    async def receive(self) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class StdioTransport:
    """MCP transport over stdin/stdout with newline-delimited JSON framing.

    Reader must expose ``readline() -> bytes | str``; writer must expose
    ``write(str)`` and ``flush()``. Both default to the process streams.
    """

    def __init__(self, reader: Any = None, writer: Any = None) -> None:
        self._reader = reader
        self._writer = writer
        self._started = False

    async def start(self) -> None:
        """Bind to stdin/stdout unless explicit streams were injected."""
        if self._reader is None:
            self._reader = sys.stdin.buffer
        if self._writer is None:
            self._writer = sys.stdout
        self._started = True

    async def send(self, message: dict[str, Any]) -> None:
        """Write one JSON-RPC message as a single line."""
        if not self._started:
            raise RuntimeError("Transport not started")
        line = json.dumps(message, ensure_ascii=False, default=str)
        self._writer.write(line + "\n")
        self._writer.flush()

    async def receive(self) -> dict[str, Any]:
        """Read the next JSON-RPC message.

        Blank lines are skipped. Raises ``EOFError`` when the peer closes the
        stream and ``ValueError`` when a line is not valid JSON.
        """
        if not self._started:
            raise RuntimeError("Transport not started")
        while True:
            raw = await asyncio.to_thread(self._reader.readline)
            if not raw:
                raise EOFError("stdin closed")
            text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            text = text.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON-RPC message: {exc}") from exc
            if not isinstance(message, dict):
                raise ValueError("JSON-RPC message must be an object")
            return message

    async def close(self) -> None:
        """Detach the streams (stdin/stdout are not ours to close)."""
        self._started = False


class SSETransport:
    """MCP transport over Server-Sent Events (for web integration).

    This is a stub — a full implementation would use aiohttp or similar.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 3000) -> None:
        self.host = host
        self.port = port
        self._running = False

    async def start(self) -> None:
        """Start the SSE server."""
        self._running = True
        logger.info("SSE transport started on %s:%d", self.host, self.port)

    async def send(self, message: dict[str, Any]) -> None:
        """Send an SSE event (stub)."""
        logger.debug("SSE send: %s", json.dumps(message, default=str)[:200])

    async def receive(self) -> dict[str, Any]:
        """Receive from SSE (stub)."""
        raise NotImplementedError("SSE receive not yet implemented")

    async def close(self) -> None:
        """Stop the SSE server."""
        self._running = False
