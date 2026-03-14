"""
Unified traffic logger for debugging and auditing.

Provides a single generic interface for logging all I/O traffic —
HTTP requests/responses, SSE chunks, MCP JSON-RPC messages, subprocess
stderr, and any other structured or unstructured output.

Every log entry follows the format::

    [timestamp] [source] [direction] [category] payload

Where:
    - timestamp: ISO 8601 with milliseconds
    - source: context identifier (e.g. ``"http"``, ``"playwright"``, session ID)
    - direction: ``>>>`` outgoing, ``<<<`` incoming, ``---`` informational
    - category: ``REQUEST``, ``RESPONSE``, ``SSE``, ``NOTIFICATION``,
      ``STDERR``, ``STDOUT``, ``STDIN``, or any custom label
    - payload: JSON-formatted dict or raw string
"""

import json
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Protocol


class TrafficLogger(Protocol):
    """Protocol for traffic logging callbacks.

    A single method covers all traffic types.  Callers identify themselves
    via ``source`` and use ``direction`` / ``category`` to describe what
    is being logged.
    """

    def log(
        self,
        source: str,
        direction: str,
        category: str,
        payload: dict[str, Any] | str | None = None,
    ) -> None:
        """
        Log a traffic entry.

        Args:
            source: Context identifier (e.g. ``"http"``, ``"playwright"``)
            direction: ``">>>"`` outgoing, ``"<<<"`` incoming, ``"---"`` info
            category: Entry type (e.g. ``"REQUEST"``, ``"STDERR"``)
            payload: Data to log — dict is JSON-serialised, str is kept as-is
        """
        ...


class FileTrafficLogger:
    """
    Logs all traffic to a single file.

    Covers HTTP, MCP, subprocess I/O, and anything else that needs
    auditing.  Thread-safe for sync writes (one entry per ``log()``
    call, opened-and-closed each time).

    Example::

        logger = FileTrafficLogger(Path("logs/session.txt"))

        # HTTP request
        logger.log("http", ">>>", "REQUEST", {"method": "POST", "url": "..."})

        # MCP tool call
        logger.log("playwright", ">>>", "REQUEST", {"method": "tools/call", ...})

        # Subprocess stderr
        logger.log("playwright", "---", "STDERR", "Listening on port 3000")
    """

    SENSITIVE_KEYS = frozenset({"authorization", "x-api-key", "api-key"})

    def __init__(self, log_file: Path) -> None:
        """
        Initialize the file logger.

        Args:
            log_file: Destination file.  Parent directories are created
                      automatically.
        """
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch(exist_ok=True)

    def log(
        self,
        source: str,
        direction: str,
        category: str,
        payload: dict[str, Any] | str | None = None,
    ) -> None:
        """Append a single log entry to the file."""
        timestamp = datetime.now(UTC).isoformat(timespec="milliseconds")

        if isinstance(payload, dict):
            sanitised = self._sanitize(payload)
            body = json.dumps(sanitised, ensure_ascii=False)
        elif payload is not None:
            body = payload
        else:
            body = ""

        entry = f"[{timestamp}] [{source}] {direction} {category} {body}\n"
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(entry)

    # ------------------------------------------------------------------
    # Convenience helpers (thin wrappers so call-sites stay readable)
    # ------------------------------------------------------------------

    def log_http_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any] | None,
        session_id: str | None = None,
    ) -> None:
        """Log an outgoing HTTP request."""
        self.log(
            source=session_id or "http",
            direction=">>>",
            category="REQUEST",
            payload={
                "method": method,
                "url": url,
                "headers": {k: self._mask(k, v) for k, v in headers.items()},
                "body": body,
            },
        )

    def log_http_response(
        self,
        url: str,
        status: int,
        body: str | dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        """Log an incoming HTTP response."""
        parsed_body: str | dict[str, Any]
        if isinstance(body, str):
            try:
                parsed_body = json.loads(body)
            except json.JSONDecodeError:
                parsed_body = body
        else:
            parsed_body = body

        self.log(
            source=session_id or "http",
            direction="<<<",
            category="RESPONSE",
            payload={"url": url, "status": status, "body": parsed_body},
        )

    def log_http_sse_chunk(
        self,
        url: str,
        data: str,
        session_id: str | None = None,
    ) -> None:
        """Log an incoming SSE chunk."""
        parsed: str | dict[str, Any]
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            parsed = data

        self.log(
            source=session_id or "http",
            direction="<<<",
            category="SSE",
            payload={"url": url, "data": parsed},
        )

    def log_mcp_request(
        self,
        server_name: str,
        method: str,
        params: dict[str, Any] | None,
    ) -> None:
        """Log an outgoing MCP JSON-RPC request."""
        self.log(
            source=server_name,
            direction=">>>",
            category="REQUEST",
            payload={"method": method, "params": params},
        )

    def log_mcp_response(
        self,
        server_name: str,
        method: str,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> None:
        """Log an incoming MCP JSON-RPC response."""
        self.log(
            source=server_name,
            direction="<<<",
            category="RESPONSE",
            payload={"method": method, "result": result, "error": error},
        )

    def log_mcp_notification(
        self,
        server_name: str,
        method: str,
        params: dict[str, Any] | None,
    ) -> None:
        """Log an MCP notification (sent or received)."""
        self.log(
            source=server_name,
            direction="---",
            category="NOTIFICATION",
            payload={"method": method, "params": params},
        )

    def log_mcp_stderr(self, server_name: str, line: str) -> None:
        """Log a stderr line from an MCP subprocess."""
        self.log(source=server_name, direction="---", category="STDERR", payload=line)

    def log_mcp_stdout_raw(self, server_name: str, line: str) -> None:
        """Log a raw stdout line from an MCP subprocess (before parsing)."""
        self.log(source=server_name, direction="<<<", category="STDOUT", payload=line)

    def log_mcp_stdin_raw(self, server_name: str, line: str) -> None:
        """Log a raw stdin line sent to an MCP subprocess."""
        self.log(source=server_name, direction=">>>", category="STDIN", payload=line)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask(self, key: str, value: str) -> str:
        """Mask sensitive header values."""
        if key.lower() in self.SENSITIVE_KEYS:
            if len(value) > 14:
                return value[:10] + "..." + value[-4:]
            return "***"
        return value

    def _sanitize(self, obj: dict[str, Any]) -> dict[str, Any]:
        """Recursively sanitize a dict, masking sensitive header values."""
        result: dict[str, Any] = {}
        for key, value in obj.items():
            if key == "headers" and isinstance(value, dict):
                result[key] = {k: self._mask(k, v) for k, v in value.items() if isinstance(v, str)}
            elif isinstance(value, dict):
                result[key] = self._sanitize(value)
            else:
                result[key] = value
        return result
