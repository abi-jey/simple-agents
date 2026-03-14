"""
MCP client session for communicating with MCP servers.

Provides a high-level API for:
- Connecting to MCP servers via stdio transport
- Discovering tools, resources, and prompts
- Calling tools and reading resources
"""

import logging
from typing import Any

from .constants import LATEST_PROTOCOL_VERSION
from .constants import SUPPORTED_PROTOCOL_VERSIONS
from .errors import MCPError
from .errors import MCPInvalidParamsError
from .errors import MCPMethodNotFoundError
from .errors import MCPParseError
from .errors import MCPTransportError
from .transport import StdioServerParameters
from .transport import StdioTransport

logger = logging.getLogger(__name__)


class MCPSession:
    """
    MCP client session for communicating with MCP servers.

    Provides a high-level async API for:
    - Connecting to MCP servers via stdio transport
    - Discovering tools, resources, and prompts
    - Calling tools and reading resources

    Example:
        async with MCPSession() as session:
            await session.connect(StdioServerParameters(
                command="npx",
                args=["-y", "@anthropic/mcp-server-test"],
            ))
            tools = await session.list_tools()
            result = await session.call_tool("my_tool", {"arg": "value"})
    """

    def __init__(
        self,
        transport: StdioTransport | None = None,
        client_info: dict[str, str] | None = None,
    ):
        """
        Initialize MCP session.

        Args:
            transport: Transport layer (defaults to StdioTransport)
            client_info: Client name/version (defaults to "nagents/0.1.0")
        """
        self._transport = transport or StdioTransport()
        self._client_info = client_info or {"name": "nagents", "version": "0.1.0"}
        self._server_capabilities: dict[str, Any] | None = None
        self._initialized = False

    @property
    def server_capabilities(self) -> dict[str, Any] | None:
        """Get server capabilities (available after initialize)."""
        return self._server_capabilities

    @property
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        return self._transport.is_connected

    @property
    def is_initialized(self) -> bool:
        """Check if session has been initialized."""
        return self._initialized

    async def __aenter__(self) -> "MCPSession":
        """Enter async context (does not auto-connect)."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context and close connection."""
        await self.close()

    async def connect(self, params: StdioServerParameters) -> dict[str, Any]:
        """
        Connect to MCP server and initialize session.

        This combines transport connection and protocol initialization
        into a single convenient method.

        Args:
            params: Server connection parameters

        Returns:
            InitializeResult with server capabilities

        Raises:
            MCPTransportError: If connection fails
            MCPError: If initialization fails
        """
        await self._transport.connect(params)
        result = await self.initialize()
        return result

    async def initialize(self) -> dict[str, Any]:
        """
        Send initialize request to server.

        Must be called after transport is connected.

        Returns:
            InitializeResult with server capabilities

        Raises:
            MCPError: If initialization fails
        """
        if not self._transport.is_connected:
            raise MCPTransportError("Transport not connected")

        params = {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": self._client_info,
        }

        result = await self._send_request("initialize", params)

        protocol_version = result.get("protocolVersion")
        if protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise MCPError(
                -1,
                f"Unsupported protocol version: {protocol_version}",
            )

        if protocol_version != LATEST_PROTOCOL_VERSION:
            logger.info(f"Server using older protocol version: {protocol_version} (latest: {LATEST_PROTOCOL_VERSION})")

        self._server_capabilities = result.get("capabilities")

        await self._transport.send_notification("notifications/initialized", None)

        self._initialized = True
        server_info = result.get("serverInfo", {})
        server_name = server_info.get("name", "unknown") if server_info else "unknown"
        logger.info(f"MCP session initialized: {server_name}")

        return result

    async def close(self) -> None:
        """Close the session and transport."""
        self._initialized = False
        self._server_capabilities = None
        await self._transport.close()

    async def list_tools(self, cursor: str | None = None) -> dict[str, Any]:
        """
        List available tools from the server.

        Args:
            cursor: Pagination cursor for large lists

        Returns:
            ListToolsResult with tools list

        Raises:
            MCPError: If request fails
        """
        params: dict[str, Any] | None = None
        if cursor:
            params = {"cursor": cursor}

        return await self._send_request("tools/list", params)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Call a tool on the server.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            CallToolResult with tool output

        Raises:
            MCPError: If tool call fails
        """
        params: dict[str, Any] = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments

        return await self._send_request("tools/call", params)

    async def list_resources(self, cursor: str | None = None) -> dict[str, Any]:
        """
        List available resources from the server.

        Args:
            cursor: Pagination cursor for large lists

        Returns:
            ListResourcesResult with resources list

        Raises:
            MCPError: If request fails
        """
        params: dict[str, Any] | None = None
        if cursor:
            params = {"cursor": cursor}

        return await self._send_request("resources/list", params)

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """
        Read a resource from the server.

        Args:
            uri: Resource URI

        Returns:
            ReadResourceResult with resource contents

        Raises:
            MCPError: If read fails
        """
        params = {"uri": uri}
        return await self._send_request("resources/read", params)

    async def subscribe_resource(self, uri: str) -> None:
        """
        Subscribe to resource updates.

        Args:
            uri: Resource URI

        Raises:
            MCPError: If subscription fails
        """
        params = {"uri": uri}
        await self._send_request("resources/subscribe", params)

    async def unsubscribe_resource(self, uri: str) -> None:
        """
        Unsubscribe from resource updates.

        Args:
            uri: Resource URI

        Raises:
            MCPError: If unsubscription fails
        """
        params = {"uri": uri}
        await self._send_request("resources/unsubscribe", params)

    async def list_prompts(self, cursor: str | None = None) -> dict[str, Any]:
        """
        List available prompts from the server.

        Args:
            cursor: Pagination cursor for large lists

        Returns:
            ListPromptsResult with prompts list

        Raises:
            MCPError: If request fails
        """
        params: dict[str, Any] | None = None
        if cursor:
            params = {"cursor": cursor}

        return await self._send_request("prompts/list", params)

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Get a prompt from the server.

        Args:
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            GetPromptResult with prompt messages

        Raises:
            MCPError: If request fails
        """
        params: dict[str, Any] = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments

        return await self._send_request("prompts/get", params)

    async def ping(self) -> None:
        """
        Send ping to check server is alive.

        Raises:
            MCPError: If ping fails
        """
        await self._send_request("ping", None)

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Send a JSON-RPC request and wait for response.

        Args:
            method: Method name
            params: Optional parameters

        Returns:
            Response result

        Raises:
            MCPError: If request fails
        """
        response = await self._transport.send_request(method, params)

        if "error" in response:
            error = response["error"]
            error_dict = error if isinstance(error, dict) else {}
            code = error_dict.get("code", -1)
            message = error_dict.get("message", "Unknown error")
            data = error_dict.get("data")

            if code == -32601:
                raise MCPMethodNotFoundError(method, data)
            elif code == -32602:
                raise MCPInvalidParamsError(message, data)
            elif code == -32700:
                raise MCPParseError(message, data)
            else:
                raise MCPError(code, message, data)

        result = response.get("result")
        return result if isinstance(result, dict) else {}
