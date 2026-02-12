"""
HTTP traffic logger for debugging and auditing.

Logs all HTTP requests and SSE responses to a file with timestamps,
session IDs, direction indicators, and full payloads.
"""

import json
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Protocol


class HTTPLogger(Protocol):
    """Protocol for HTTP logging callbacks."""

    def log_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any] | None,
        session_id: str | None = None,
    ) -> None:
        """Log an outgoing HTTP request."""
        ...

    def log_response(
        self,
        url: str,
        status: int,
        body: str | dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        """Log an incoming HTTP response."""
        ...

    def log_sse_chunk(
        self,
        url: str,
        data: str,
        session_id: str | None = None,
    ) -> None:
        """Log an incoming SSE chunk."""
        ...


class FileHTTPLogger:
    """
    Logs HTTP traffic to a file.

    Format:
        [timestamp] [session_id] [direction] [type] payload

    Where:
        - timestamp: ISO 8601 format
        - session_id: The session ID or "no-session"
        - direction: >>> for outgoing, <<< for incoming
        - type: REQUEST, RESPONSE, or SSE
        - payload: JSON-formatted data
    """

    def __init__(self, log_file: Path):
        """
        Initialize the file logger.

        Args:
            log_file: Path to the log file. Parent directories will be created
                      if they don't exist.
        """
        self.log_file = log_file
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """Ensure the log file and parent directories exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Touch the file to ensure it exists
        self.log_file.touch(exist_ok=True)

    def _format_timestamp(self) -> str:
        """Get current timestamp in ISO 8601 format."""
        return datetime.now(UTC).isoformat(timespec="milliseconds")

    def _write_log(self, entry: str) -> None:
        """Append a log entry to the file."""
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")

    def _sanitize_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Sanitize headers by masking sensitive values."""
        sanitized = {}
        sensitive_keys = {"authorization", "x-api-key", "api-key"}
        for key, value in headers.items():
            if key.lower() in sensitive_keys:
                # Show first 10 chars, mask the rest
                if len(value) > 14:
                    sanitized[key] = value[:10] + "..." + value[-4:]
                else:
                    sanitized[key] = "***"
            else:
                sanitized[key] = value
        return sanitized

    def log_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any] | None,
        session_id: str | None = None,
    ) -> None:
        """Log an outgoing HTTP request."""
        timestamp = self._format_timestamp()
        sid = session_id or "no-session"

        payload = {
            "method": method,
            "url": url,
            "headers": self._sanitize_headers(headers),
            "body": body,
        }

        entry = f"[{timestamp}] [{sid}] >>> REQUEST {json.dumps(payload, ensure_ascii=False)}"
        self._write_log(entry)

    def log_response(
        self,
        url: str,
        status: int,
        body: str | dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        """Log an incoming HTTP response."""
        timestamp = self._format_timestamp()
        sid = session_id or "no-session"

        # Try to parse body as JSON if it's a string
        parsed_body: str | dict[str, Any]
        if isinstance(body, str):
            try:
                parsed_body = json.loads(body)
            except json.JSONDecodeError:
                parsed_body = body
        else:
            parsed_body = body

        payload = {
            "url": url,
            "status": status,
            "body": parsed_body,
        }

        entry = f"[{timestamp}] [{sid}] <<< RESPONSE {json.dumps(payload, ensure_ascii=False)}"
        self._write_log(entry)

    def log_sse_chunk(
        self,
        url: str,
        data: str,
        session_id: str | None = None,
    ) -> None:
        """Log an incoming SSE chunk."""
        timestamp = self._format_timestamp()
        sid = session_id or "no-session"

        # Try to parse data as JSON
        parsed_data: str | dict[str, Any]
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            parsed_data = data

        payload = {
            "url": url,
            "data": parsed_data,
        }

        entry = f"[{timestamp}] [{sid}] <<< SSE {json.dumps(payload, ensure_ascii=False)}"
        self._write_log(entry)
