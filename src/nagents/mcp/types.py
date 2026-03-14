"""
MCP protocol types using TypedDict for zero-dependency implementation.

Based on MCP specification: https://modelcontextprotocol.io/specification/2025-03-26
"""

from typing import Any
from typing import Literal
from typing import TypedDict

# === JSON-RPC Base Types ===


class JSONRPCRequest(TypedDict):
    """JSON-RPC 2.0 request."""

    jsonrpc: Literal["2.0"]
    id: int | str
    method: str
    params: dict[str, Any] | None


class JSONRPCResponse(TypedDict):
    """JSON-RPC 2.0 response."""

    jsonrpc: Literal["2.0"]
    id: int | str
    result: dict[str, Any] | None
    error: "JSONRPCError | None"


class JSONRPCError(TypedDict):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any | None


class JSONRPCNotification(TypedDict):
    """JSON-RPC 2.0 notification (no id, no response expected)."""

    jsonrpc: Literal["2.0"]
    method: str
    params: dict[str, Any] | None


# === Common Types ===


class Implementation(TypedDict):
    """Client or server implementation info."""

    name: str
    version: str


class RequestParamsMeta(TypedDict, total=False):
    """Metadata for request parameters."""

    progressToken: int | str


class RequestParams(TypedDict, total=False):
    """Base request parameters."""

    _meta: RequestParamsMeta


# === Capabilities ===


class ToolsCapability(TypedDict, total=False):
    """Server tools capability."""

    listChanged: bool


class ResourcesCapability(TypedDict, total=False):
    """Server resources capability."""

    subscribe: bool
    listChanged: bool


class PromptsCapability(TypedDict, total=False):
    """Server prompts capability."""

    listChanged: bool


class LoggingCapability(TypedDict, total=False):
    """Server logging capability."""

    pass


class ClientCapabilities(TypedDict, total=False):
    """Client capabilities sent during initialization."""

    experimental: dict[str, Any] | None
    roots: "RootsCapability | None"


class RootsCapability(TypedDict, total=False):
    """Client roots capability."""

    listChanged: bool


class ServerCapabilities(TypedDict, total=False):
    """Server capabilities returned during initialization."""

    experimental: dict[str, Any] | None
    tools: ToolsCapability | None
    resources: ResourcesCapability | None
    prompts: PromptsCapability | None
    logging: LoggingCapability | None


# === Initialization ===


class InitializeRequestParams(TypedDict):
    """Parameters for initialize request."""

    protocolVersion: str
    capabilities: ClientCapabilities
    clientInfo: Implementation


class InitializeResult(TypedDict):
    """Result of initialize request."""

    protocolVersion: str
    capabilities: ServerCapabilities
    serverInfo: Implementation
    instructions: str | None


# === Tools ===


class ToolInputSchema(TypedDict, total=False):
    """JSON Schema for tool input."""

    type: Literal["object"]
    properties: dict[str, Any]
    required: list[str] | None


class Tool(TypedDict, total=False):
    """Tool definition from server."""

    name: str
    description: str
    inputSchema: ToolInputSchema


class ListToolsResult(TypedDict, total=False):
    """Result of tools/list request."""

    tools: list[Tool]
    nextCursor: str | None


class CallToolRequestParams(TypedDict):
    """Parameters for tools/call request."""

    name: str
    arguments: dict[str, Any] | None


class TextContent(TypedDict):
    """Text content block."""

    type: Literal["text"]
    text: str


class ImageContent(TypedDict):
    """Image content block."""

    type: Literal["image"]
    data: str
    mimeType: str


class EmbeddedResource(TypedDict):
    """Embedded resource content block."""

    type: Literal["resource"]
    resource: "ResourceContents"


ContentBlock = TextContent | ImageContent | EmbeddedResource


class CallToolResult(TypedDict, total=False):
    """Result of tools/call request."""

    content: list[ContentBlock]
    isError: bool


# === Resources ===


class ResourceContents(TypedDict, total=False):
    """Resource contents."""

    uri: str
    mimeType: str | None
    text: str | None
    blob: str | None


class Resource(TypedDict, total=False):
    """Resource definition."""

    uri: str
    name: str
    description: str | None
    mimeType: str | None


class ListResourcesResult(TypedDict, total=False):
    """Result of resources/list request."""

    resources: list[Resource]
    nextCursor: str | None


class ReadResourceRequestParams(TypedDict):
    """Parameters for resources/read request."""

    uri: str


class ReadResourceResult(TypedDict):
    """Result of resources/read request."""

    contents: list[ResourceContents]


class SubscribeRequestParams(TypedDict):
    """Parameters for resources/subscribe request."""

    uri: str


class UnsubscribeRequestParams(TypedDict):
    """Parameters for resources/unsubscribe request."""

    uri: str


# === Prompts ===


class PromptArgument(TypedDict, total=False):
    """Argument for a prompt template."""

    name: str
    description: str | None
    required: bool | None


class Prompt(TypedDict, total=False):
    """Prompt definition."""

    name: str
    description: str | None
    arguments: list[PromptArgument] | None


class ListPromptsResult(TypedDict, total=False):
    """Result of prompts/list request."""

    prompts: list[Prompt]
    nextCursor: str | None


class PromptMessage(TypedDict):
    """Message in a prompt."""

    role: Literal["user", "assistant"]
    content: ContentBlock


class GetPromptRequestParams(TypedDict, total=False):
    """Parameters for prompts/get request."""

    name: str
    arguments: dict[str, str] | None


class GetPromptResult(TypedDict, total=False):
    """Result of prompts/get request."""

    description: str | None
    messages: list[PromptMessage]


# === Notifications ===


class ProgressNotificationParams(TypedDict):
    """Parameters for progress notification."""

    progressToken: int | str
    progress: float
    total: float | None


class ProgressNotification(TypedDict):
    """Progress notification."""

    jsonrpc: Literal["2.0"]
    method: Literal["notifications/progress"]
    params: ProgressNotificationParams


class LoggingMessageNotificationParams(TypedDict):
    """Parameters for logging message notification."""

    level: Literal["debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"]
    logger: str | None
    data: Any


class LoggingMessageNotification(TypedDict):
    """Logging message notification."""

    jsonrpc: Literal["2.0"]
    method: Literal["notifications/message"]
    params: LoggingMessageNotificationParams


# === Empty Result ===


class EmptyResult(TypedDict):
    """Empty result for requests that return nothing."""

    pass
