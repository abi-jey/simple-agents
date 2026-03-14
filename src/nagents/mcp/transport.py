"""
MCP stdio transport implementation.

Manages subprocess lifecycle and JSON-RPC message framing over stdin/stdout.
"""

import asyncio
import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from .constants import PROCESS_TERMINATION_TIMEOUT
from .errors import MCPTimeoutError
from .errors import MCPTransportError

if TYPE_CHECKING:
    from ..logger import FileTrafficLogger

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    DEFAULT_INHERITED_ENV_VARS = [
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "USERNAME",
        "USERPROFILE",
    ]
else:
    DEFAULT_INHERITED_ENV_VARS = [
        "DISPLAY",
        "HOME",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "USER",
        "WAYLAND_DISPLAY",
        "XAUTHORITY",
        "XDG_RUNTIME_DIR",
        "XDG_SESSION_TYPE",
    ]


def _get_default_environment() -> dict[str, str]:
    """Returns a default environment including safe-to-inherit variables."""
    env: dict[str, str] = {}
    for key in DEFAULT_INHERITED_ENV_VARS:
        value = os.environ.get(key)
        if value is None:
            continue
        if value.startswith("()"):
            continue
        env[key] = value
    return env


class StdioServerParameters:
    """
    Parameters for connecting to an MCP server via stdio.

    Attributes:
        command: The executable to run (e.g., "npx", "python", "node")
        args: Command line arguments to pass to the executable
        env: Environment variables (merged with defaults)
        cwd: Working directory for the subprocess
        encoding: Text encoding for stdin/stdout (default: utf-8)
        timeout: Request timeout in seconds (default: 30.0)
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        encoding: str = "utf-8",
        timeout: float = 30.0,
    ):
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = str(cwd) if cwd else None
        self.encoding = encoding
        self.timeout = timeout

    def get_full_env(self) -> dict[str, str]:
        """Get merged environment (defaults + overrides)."""
        base = _get_default_environment()
        if self.env:
            base.update(self.env)
        return base


class StdioTransport:
    """
    Transport for MCP servers using stdio (subprocess stdin/stdout).

    Manages the subprocess lifecycle and JSON-RPC message framing.
    """

    def __init__(
        self,
        server_name: str = "mcp",
        traffic_logger: "FileTrafficLogger | None" = None,
    ) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._params: StdioServerParameters | None = None
        self._request_id = 0
        self._pending_requests: dict[int | str, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._started = False
        self._server_name = server_name
        self._traffic_logger = traffic_logger

    def set_logger(self, traffic_logger: "FileTrafficLogger | None") -> None:
        """Set the traffic file logger."""
        self._traffic_logger = traffic_logger

    def set_server_name(self, server_name: str) -> None:
        """Set the server name for logging."""
        self._server_name = server_name

    @property
    def is_connected(self) -> bool:
        """Check if transport is connected to a running process."""
        return self._started and self._process is not None and self._process.returncode is None

    async def connect(self, params: StdioServerParameters) -> None:
        """
        Start the MCP server subprocess.

        Args:
            params: Server connection parameters

        Raises:
            MCPTransportError: If the process fails to start
        """
        if self._started:
            raise MCPTransportError("Transport already connected")

        self._params = params

        try:
            self._process = await asyncio.create_subprocess_exec(
                params.command,
                *params.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=params.get_full_env(),
                cwd=params.cwd,
                start_new_session=True,
            )

            # Increase stream reader limits for large MCP responses (e.g., browser snapshots)
            # Default limit is 64KB which can be exceeded by large page content
            # StreamReader._limit is a private runtime attribute
            if self._process.stdout:
                self._process.stdout._limit = 10 * 1024 * 1024  # type: ignore[attr-defined]
            if self._process.stderr:
                self._process.stderr._limit = 1024 * 1024  # type: ignore[attr-defined]

            self._started = True

            self._reader_task = asyncio.create_task(self._read_stdout())

            self._stderr_task = asyncio.create_task(self._read_stderr())

            logger.info(f"Started MCP server: {params.command} {' '.join(params.args)}")

        except OSError as e:
            raise MCPTransportError(f"Failed to start MCP server: {e}") from e

    async def _read_stdout(self) -> None:
        """Background task to read and parse stdout lines."""
        if not self._process or not self._process.stdout:
            return

        try:
            while True:
                line_bytes = await self._process.stdout.readline()
                if not line_bytes:
                    break

                encoding = self._params.encoding if self._params else "utf-8"
                line = line_bytes.decode(encoding).strip()
                if not line:
                    continue

                # Log raw stdout line before parsing
                if self._traffic_logger:
                    self._traffic_logger.log_mcp_stdout_raw(self._server_name, line)

                try:
                    message = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from server: {e}")
                    continue

                if "id" in message:
                    request_id = message["id"]
                    if request_id in self._pending_requests:
                        future = self._pending_requests.pop(request_id)
                        if not future.done():
                            future.set_result(message)
                    else:
                        logger.warning(f"Received response for unknown request id: {request_id}")
                else:
                    # Notification — log to file and Python logger
                    if self._traffic_logger:
                        self._traffic_logger.log_mcp_notification(
                            self._server_name,
                            message.get("method", "unknown"),
                            message.get("params"),
                        )
                    logger.debug(f"Received notification: {message.get('method', 'unknown')}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Error reading stdout: {e}")
        finally:
            for _request_id, future in self._pending_requests.items():
                if not future.done():
                    future.set_exception(MCPTransportError("Connection closed"))
            self._pending_requests.clear()

    async def _read_stderr(self) -> None:
        """Background task to log stderr output."""
        if not self._process or not self._process.stderr:
            return

        try:
            while True:
                line_bytes = await self._process.stderr.readline()
                if not line_bytes:
                    break

                encoding = self._params.encoding if self._params else "utf-8"
                line = line_bytes.decode(encoding).strip()
                if line:
                    # Log to file logger and Python logger
                    if self._traffic_logger:
                        self._traffic_logger.log_mcp_stderr(self._server_name, line)
                    logger.debug(f"MCP server stderr: {line}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Error reading stderr: {e}")

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Send a JSON-RPC request and wait for response.

        Args:
            method: The method name
            params: Optional parameters

        Returns:
            The JSON-RPC response

        Raises:
            MCPTransportError: If not connected or connection lost
            MCPTimeoutError: If request times out
        """
        if not self.is_connected:
            raise MCPTransportError("Not connected to MCP server")

        self._request_id += 1
        request_id = self._request_id

        request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            request["params"] = params

        message = json.dumps(request) + "\n"

        assert self._process is not None
        assert self._process.stdin is not None

        future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        self._pending_requests[request_id] = future

        try:
            encoding = self._params.encoding if self._params else "utf-8"
            self._process.stdin.write(message.encode(encoding))
            await self._process.stdin.drain()

            # Log raw stdin and structured request
            if self._traffic_logger:
                self._traffic_logger.log_mcp_stdin_raw(self._server_name, message.strip())
                self._traffic_logger.log_mcp_request(self._server_name, method, params)

            logger.debug(f"Sent MCP request: {method} (id={request_id})")

            timeout = self._params.timeout if self._params else 30.0
            response = await asyncio.wait_for(future, timeout=timeout)

            if self._traffic_logger:
                self._traffic_logger.log_mcp_response(
                    self._server_name,
                    method,
                    response.get("result"),
                    response.get("error"),
                )

            return response

        except TimeoutError:
            self._pending_requests.pop(request_id, None)
            logger.warning(f"MCP request timed out: {method} (id={request_id})")
            raise MCPTimeoutError(f"Request timed out: {method}") from None
        except Exception:
            self._pending_requests.pop(request_id, None)
            raise

    async def send_notification(
        self,
        method: str,
        params: dict[str, Any] | None,
    ) -> None:
        """
        Send a JSON-RPC notification (no response expected).

        Args:
            method: The notification method name
            params: Optional parameters
        """
        if not self.is_connected:
            raise MCPTransportError("Not connected to MCP server")

        notification: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            notification["params"] = params

        message = json.dumps(notification) + "\n"

        assert self._process is not None
        assert self._process.stdin is not None

        encoding = self._params.encoding if self._params else "utf-8"
        self._process.stdin.write(message.encode(encoding))
        await self._process.stdin.drain()

        # Log raw stdin and structured notification
        if self._traffic_logger:
            self._traffic_logger.log_mcp_stdin_raw(self._server_name, message.strip())
            self._traffic_logger.log_mcp_notification(self._server_name, method, params)

        logger.debug(f"Sent MCP notification: {method}")

    async def close(self) -> None:
        """
        Close the connection and terminate the subprocess.

        Follows MCP shutdown sequence:
        1. Close stdin
        2. Wait for graceful exit
        3. Terminate if needed
        4. Kill if still running
        """
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None

        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None

        if not self._process:
            return

        if self._process.stdin:
            try:
                self._process.stdin.close()
                await self._process.stdin.wait_closed()
            except Exception:
                pass

        try:
            await asyncio.wait_for(self._process.wait(), timeout=PROCESS_TERMINATION_TIMEOUT)
            logger.debug("MCP server exited gracefully")
        except TimeoutError:
            logger.warning("MCP server did not exit gracefully, terminating...")
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=PROCESS_TERMINATION_TIMEOUT)
            except TimeoutError:
                logger.warning("MCP server did not terminate, killing...")
                self._process.kill()
                await self._process.wait()
        except ProcessLookupError:
            pass

        self._process = None
        self._started = False
        self._pending_requests.clear()
        logger.info("MCP server connection closed")
