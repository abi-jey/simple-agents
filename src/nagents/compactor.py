"""Agent-based context compaction system.

This module provides intelligent context management using separate agents
for summarization, with configurable triggers and events.

Key components:
- Tokens/Messages: Trigger configurations
- CompactionStartedEvent/CompactionDoneEvent: Events emitted during compaction
- Compactor: Configuration for compaction agent
- DEFAULT_COMPACTOR: Pre-configured default compactor
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .events.types import CompactionDoneEvent
from .events.types import CompactionStartedEvent

if TYPE_CHECKING:
    from .provider import Provider

# Default system prompt for compaction
DEFAULT_COMPACT_PROMPT = """You are a context compaction assistant.
Summarize the conversation history in organized sections.
Focus on:
- What was discussed
- Key decisions made
- What needs to be done next
- Important context for continuing the conversation

Be concise but thorough. Preserve all critical information."""


@dataclass
class Tokens:
    """Token-based compaction trigger.

    Compaction triggers when estimated tokens >= threshold.

    Can be configured in two ways:
    1. Total context window: Tokens(total=200000) - compacts at 80% utilization
    2. Detailed: Tokens(input=100000, output=8000) - compact when tokens >= input - output

    Attributes:
        total: Total context window size (triggers at 80% utilization)
        input: Maximum input context tokens before compaction (used if total not set)
        output: Reserved tokens for output/response (default 10% of context)
    """

    total: int | None = None
    input: int | None = None
    output: int | None = None

    def __post_init__(self) -> None:
        if self.total is not None:
            if self.input is not None or self.output is not None:
                raise ValueError("Cannot specify both 'total' and 'input'/'output'")
            self.input = int(self.total * 0.8)
            self.output = int(self.total * 0.1)
        elif self.input is None:
            raise ValueError("Must specify either 'total' or 'input'")
        if self.output is None:
            self.output = 8000


@dataclass
class Messages:
    """Message-count-based compaction trigger.

    Compaction triggers when message count >= length.

    Attributes:
        length: Maximum message count before compaction
    """

    length: int


__all__ = [
    "DEFAULT_COMPACTOR",
    "CompactionDoneEvent",
    "CompactionStartedEvent",
    "Compactor",
    "Messages",
    "Tokens",
]


class Compactor:
    """Configuration for agent-based context compaction.

    A Compactor defines how context compaction should be performed,
    including the provider/model to use, system prompt, and trigger
    configuration.

    Attributes:
        provider: Provider to use for compaction (None = use main agent's provider)
        system_prompt: System prompt for the compaction agent
        compact_on: Trigger configuration (Tokens or Messages)
        model: Model override for compaction (None = use provider's default)

    Example:
        # Use smaller model for compaction
        compactor = Compactor(
            model="gpt-4o-mini",
            compact_on=Tokens(input=50000, output=5000),
        )
        agent = Agent(provider=provider, session_manager=session, compactor=compactor)
    """

    DEFAULT_SYSTEM_PROMPT = DEFAULT_COMPACT_PROMPT

    def __init__(
        self,
        provider: Provider | None = None,
        system_prompt: str | None = None,
        compact_on: Tokens | Messages | None = None,
        model: str | None = None,
    ):
        """Initialize the Compactor.

        Args:
            provider: Provider to use for compaction. If None, uses the main
                     agent's provider.
            system_prompt: System prompt for the compaction agent. If None,
                          uses DEFAULT_SYSTEM_PROMPT.
            compact_on: Trigger configuration. If None, uses resolution order:
                        1. Main agent's compact_on
                        2. Provider's default
                        3. Fallback based on model context limit
            model: Model override. If provided, overrides the provider's model
                  for compaction requests.
        """
        self.provider = provider
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.compact_on = compact_on
        self.model = model

    def get_model_name(self) -> str:
        """Get the model name to use for compaction."""
        if self.model:
            return self.model
        if self.provider and hasattr(self.provider, "model") and self.provider.model:
            return self.provider.model
        return "unknown"


# Pre-defined default compactor
# When main agent doesn't specify compactor, this is used
# At runtime, it's configured with the main agent's provider and session
DEFAULT_COMPACTOR = Compactor()


def generate_compaction_session_id(session_id: str) -> str:
    """Generate a unique session ID for compaction.

    Args:
        session_id: Original session ID

    Returns:
        Unique session ID for the compaction session
    """
    return f"{session_id}-compaction-{uuid.uuid4().hex[:8]}"
