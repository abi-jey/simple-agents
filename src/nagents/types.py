"""
Core types for the v2 LLM integration module.
"""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal
from typing import TypedDict

# =============================================================================
# JSON and Schema Type Definitions
# =============================================================================

# Type alias for JSON-compatible values (used for API payloads and arguments)
JsonValue = str | int | float | bool | None | list[Any] | dict[str, Any]

# Type alias for tool arguments (always a dict mapping param names to values)
ToolArguments = dict[str, JsonValue]


class JsonSchemaProperty(TypedDict, total=False):
    """JSON Schema property definition."""

    type: str
    description: str
    enum: list[str]
    items: "JsonSchemaProperty"
    properties: dict[str, "JsonSchemaProperty"]
    required: list[str]


class JsonSchema(TypedDict, total=False):
    """JSON Schema for tool parameters."""

    type: str
    properties: dict[str, JsonSchemaProperty]
    required: list[str]
    additionalProperties: bool


class GeminiThinkingConfig(TypedDict, total=False):
    """Gemini-specific thinking/reasoning configuration."""

    thinkingBudget: int  # Maximum tokens for thinking
    includeThoughts: bool  # Whether to include thoughts in response


# =============================================================================
# Multimodal Content Types
# =============================================================================


@dataclass
class TextContent:
    """Text content part."""

    text: str
    type: Literal["text"] = "text"


@dataclass
class ImageContent:
    """
    Image content part.

    Requires base64-encoded image data with media type.
    """

    base64_data: str
    media_type: str  # e.g., "image/jpeg", "image/png", "image/gif", "image/webp"
    type: Literal["image"] = "image"
    detail: str | None = None  # OpenAI-specific: "auto", "low", "high"


@dataclass
class AudioContent:
    """
    Audio content part (OpenAI-specific for input).

    Used for audio input in OpenAI's GPT-4o and similar models.
    """

    base64_data: str
    format: str = "wav"  # "wav" or "mp3"
    type: Literal["audio"] = "audio"


@dataclass
class DocumentContent:
    """
    Document/PDF content part (Anthropic-specific).

    Used for PDF and document input in Claude models.
    """

    base64_data: str
    media_type: str = "application/pdf"
    title: str | None = None
    type: Literal["document"] = "document"


# Union type for all content parts
ContentPart = TextContent | ImageContent | AudioContent | DocumentContent


@dataclass
class ToolCall:
    """A function/tool call requested by the model."""

    id: str
    name: str
    arguments: ToolArguments = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """Definition of a tool/function the model can call."""

    name: str
    description: str
    parameters: JsonSchema  # JSON Schema for function parameters
    func: Callable[..., Any] | None = None  # The actual callable (not sent to API)


@dataclass
class Message:
    """
    A message in a conversation.

    Content can be:
    - str: Simple text content (for backward compatibility)
    - list[ContentPart]: Multimodal content (text, images, audio, documents)
    - None: For assistant messages with only tool calls
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # For tool result messages
    name: str | None = None  # Tool name for tool results


@dataclass
class GenerationConfig:
    """Configuration for generation."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    # Gemini-specific
    thinking_config: GeminiThinkingConfig | None = None
