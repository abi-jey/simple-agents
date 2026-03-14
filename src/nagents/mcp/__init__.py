"""
MCP (Model Context Protocol) client for connecting to MCP servers.

This module provides a lightweight, zero-dependency implementation of the
MCP client protocol for connecting to MCP servers via stdio transport.

Example:
    from nagents.mcp import MCPSession, StdioServerParameters

    async with MCPSession() as session:
        await session.connect(StdioServerParameters(
            command="npx",
            args=["-y", "@playwright/mcp@latest"],
        ))

        # List available tools
        tools = await session.list_tools()
        for tool in tools["tools"]:
            print(f"Tool: {tool['name']}")

        # Call a tool
        result = await session.call_tool("browser_navigate", {"url": "https://example.com"})
        print(result)
"""

from .constants import CONNECTION_CLOSED
from .constants import INTERNAL_ERROR
from .constants import INVALID_PARAMS
from .constants import INVALID_REQUEST
from .constants import LATEST_PROTOCOL_VERSION
from .constants import METHOD_NOT_FOUND
from .constants import PARSE_ERROR
from .constants import PROCESS_TERMINATION_TIMEOUT
from .constants import REQUEST_TIMEOUT
from .constants import SUPPORTED_PROTOCOL_VERSIONS
from .errors import MCPError
from .errors import MCPInvalidParamsError
from .errors import MCPInvalidRequestError
from .errors import MCPMethodNotFoundError
from .errors import MCPParseError
from .errors import MCPProtocolError
from .errors import MCPTimeoutError
from .errors import MCPTransportError
from .session import MCPSession
from .transport import StdioServerParameters
from .transport import StdioTransport
from .types import CallToolResult
from .types import ClientCapabilities
from .types import ContentBlock
from .types import EmbeddedResource
from .types import EmptyResult
from .types import GetPromptResult
from .types import ImageContent
from .types import InitializeResult
from .types import ListPromptsResult
from .types import ListResourcesResult
from .types import ListToolsResult
from .types import LoggingMessageNotification
from .types import LoggingMessageNotificationParams
from .types import ProgressNotification
from .types import ProgressNotificationParams
from .types import Prompt
from .types import PromptArgument
from .types import PromptMessage
from .types import ReadResourceResult
from .types import Resource
from .types import ResourceContents
from .types import ServerCapabilities
from .types import TextContent
from .types import Tool
from .types import ToolInputSchema

__all__ = [
    "CONNECTION_CLOSED",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "LATEST_PROTOCOL_VERSION",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "PROCESS_TERMINATION_TIMEOUT",
    "REQUEST_TIMEOUT",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "CallToolResult",
    "ClientCapabilities",
    "ContentBlock",
    "EmbeddedResource",
    "EmptyResult",
    "GetPromptResult",
    "ImageContent",
    "InitializeResult",
    "ListPromptsResult",
    "ListResourcesResult",
    "ListToolsResult",
    "LoggingMessageNotification",
    "LoggingMessageNotificationParams",
    "MCPError",
    "MCPInvalidParamsError",
    "MCPInvalidRequestError",
    "MCPMethodNotFoundError",
    "MCPParseError",
    "MCPProtocolError",
    "MCPSession",
    "MCPTimeoutError",
    "MCPTransportError",
    "ProgressNotification",
    "ProgressNotificationParams",
    "Prompt",
    "PromptArgument",
    "PromptMessage",
    "ReadResourceResult",
    "Resource",
    "ResourceContents",
    "ServerCapabilities",
    "StdioServerParameters",
    "StdioTransport",
    "TextContent",
    "Tool",
    "ToolInputSchema",
]
