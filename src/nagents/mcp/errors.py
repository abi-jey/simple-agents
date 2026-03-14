"""
MCP-specific exceptions.
"""

from typing import Any

from .constants import CONNECTION_CLOSED
from .constants import INTERNAL_ERROR
from .constants import INVALID_PARAMS
from .constants import INVALID_REQUEST
from .constants import METHOD_NOT_FOUND
from .constants import PARSE_ERROR
from .constants import REQUEST_TIMEOUT


class MCPError(Exception):
    """Base exception for MCP errors."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"[{code}] {message}")

    @classmethod
    def from_jsonrpc(cls, error: dict[str, Any]) -> "MCPError":
        """Create MCPError from JSON-RPC error object."""
        code = error.get("code", INTERNAL_ERROR)
        message = error.get("message", "Unknown error")
        data = error.get("data")
        return cls(code, message, data)


class MCPTransportError(MCPError):
    """Transport-level error (connection, process, etc.)."""

    def __init__(self, message: str, data: Any = None):
        super().__init__(CONNECTION_CLOSED, message, data)


class MCPProtocolError(MCPError):
    """Protocol-level error (invalid message, etc.)."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(code, message, data)


class MCPTimeoutError(MCPError):
    """Request timeout error."""

    def __init__(self, message: str = "Request timeout", data: Any = None):
        super().__init__(REQUEST_TIMEOUT, message, data)


class MCPMethodNotFoundError(MCPProtocolError):
    """Method not found error."""

    def __init__(self, method: str, data: Any = None):
        super().__init__(METHOD_NOT_FOUND, f"Method not found: {method}", data)


class MCPInvalidParamsError(MCPProtocolError):
    """Invalid parameters error."""

    def __init__(self, message: str = "Invalid parameters", data: Any = None):
        super().__init__(INVALID_PARAMS, message, data)


class MCPParseError(MCPProtocolError):
    """Parse error."""

    def __init__(self, message: str = "Parse error", data: Any = None):
        super().__init__(PARSE_ERROR, message, data)


class MCPInvalidRequestError(MCPProtocolError):
    """Invalid request error."""

    def __init__(self, message: str = "Invalid request", data: Any = None):
        super().__init__(INVALID_REQUEST, message, data)
