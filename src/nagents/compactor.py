"""Agent-based context compaction system.

This module provides intelligent context management using separate agents
for summarization, with configurable triggers and events.

Key components:
- Tokens/Messages: Trigger configurations
- CompactionStartedEvent/CompactionDoneEvent: Events emitted during compaction
- Compactor: Compaction agent that inherits from Agent
- DEFAULT_COMPACTOR: Pre-configured default compactor
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from .events.types import CompactionDoneEvent
from .events.types import CompactionStartedEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .agent import Agent
    from .events import Event
    from .provider import Provider
    from .session import SessionManager

# Default system prompt for compaction
DEFAULT_COMPACT_PROMPT = """

Below is a conversation history  between an AI Autonomous Agent and the Human User.
Summarize the conversation history in organized sections.
- The timeline of events and What was dicussed
- What were different todos and goals and decisions were made
- What are things that are done and what is pending and still needs to be done.
- Context about environment, tools, and resources, for example: "for performing Task A You decided to tool x, y, z and
this was the becuase the user said abc, and environment had pqr.", or "host name for machine is abc, and we can use user abc."

This summary should be representative of whole converstation and actions that were taken.
Nothing should be left out

Be concise and try to perseve as much information as possible. The summary should be addressing the AI agent as You.

Include last few messages from user and agent in the summary.
===
"""


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


# Lazy import to avoid circular dependency
_CompactorClass = None


def _get_compactor_class() -> Any:
    """Lazily create Compactor class to avoid circular import."""
    global _CompactorClass
    if _CompactorClass is None:
        from .agent import Agent
        from .provider import PLACEHOLDER_PROVIDER
        from .session.manager import PLACEHOLDER_SESSION_MANAGER

        class _Compactor(Agent):
            """Compaction agent that IS AN Agent."""

            DEFAULT_SYSTEM_PROMPT = DEFAULT_COMPACT_PROMPT

            def __init__(
                self,
                provider: Provider | None = None,
                session_manager: SessionManager | None = None,
                system_prompt: str | None = None,
                compact_on: Tokens | Messages | None = None,
                model: str | None = None,
            ):
                # Use placeholders if not provided
                if provider is None:
                    provider = PLACEHOLDER_PROVIDER
                if session_manager is None:
                    session_manager = PLACEHOLDER_SESSION_MANAGER
                actual_prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT
                # At this point, provider and session_manager are guaranteed non-None
                assert provider is not None and session_manager is not None

                super().__init__(
                    provider=provider,
                    session_manager=session_manager,
                    tools=[],
                    system_prompt=actual_prompt,
                    compactor=None,  # Disable compaction for compaction agent
                    compact_on=None,
                )
                self.compact_on = compact_on
                self._model_override = model

            def get_model_name(self) -> str:
                """Get the model name to use for compaction."""
                if self._model_override:
                    return self._model_override
                if self.provider and hasattr(self.provider, "model") and self.provider.model:
                    return self.provider.model
                return "unknown"

        _CompactorClass = _Compactor

    return _CompactorClass


class Compactor:
    """Compaction agent that IS AN Agent.

    A Compactor IS-A specialized Agent for context compaction.
    Uses placeholder provider and session by default, which get replaced
    before run() by the main agent.

    This class provides a convenient interface for creating and managing
    the underlying Agent instance that performs compaction.

    Attributes:
        provider: Provider to use for compaction (None = use main agent's provider)
        system_prompt: System prompt for the compaction agent
        compact_on: Trigger configuration (Tokens or Messages)
        model: Model override for compaction (None = use provider's default)
    """

    DEFAULT_SYSTEM_PROMPT = DEFAULT_COMPACT_PROMPT

    def __init__(
        self,
        provider: Provider | None = None,
        system_prompt: str | None = None,
        compact_on: Tokens | Messages | None = None,
        model: str | None = None,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT
        self._compact_on = compact_on
        self._model = model
        self._agent: Agent | None = None

    def _ensure_agent(self) -> Agent:
        """Ensure the Agent instance exists and has provider set.

        Returns:
            The Agent instance

        Raises:
            ValueError: If provider is not set
        """
        if self._agent is None:
            CompactorCls = _get_compactor_class()
            self._agent = CompactorCls(
                provider=self._provider,
                system_prompt=self._system_prompt,
                compact_on=self._compact_on,
                model=self._model,
            )
        return self._agent

    @property
    def provider(self) -> Provider | None:
        """Get the provider."""
        return self._provider

    @provider.setter
    def provider(self, value: Provider) -> None:
        """Set the provider."""
        self._provider = value
        if self._agent is not None:
            self._agent.provider = value

    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Set the system prompt."""
        self._system_prompt = value
        if self._agent is not None:
            self._agent.system_prompt = value

    @property
    def compact_on(self) -> Tokens | Messages | None:
        """Get the compaction trigger."""
        return self._compact_on

    @property
    def model(self) -> str | None:
        """Get the model override."""
        return self._model

    def get_model_name(self) -> str:
        """Get the model name to use for compaction."""
        if self._model:
            return self._model
        if self._provider and hasattr(self._provider, "model") and self._provider.model:
            return self._provider.model
        return "unknown"

    @property
    def session(self) -> SessionManager | None:
        """Get session from underlying agent."""
        if self._agent is None:
            return None
        return self._agent.session

    @session.setter
    def session(self, value: SessionManager) -> None:
        """Set session on underlying agent."""
        agent = self._ensure_agent()
        agent.session = value

    async def run(self, user_message: str, session_id: str) -> AsyncIterator[Event]:
        """Run compaction as an Agent.

        Args:
            user_message: The formatted conversation text
            session_id: Session ID for compaction

        Yields:
            Events from the compaction agent
        """
        if self._provider is None:
            raise ValueError("Compactor.provider must be set before run()")

        agent = self._ensure_agent()
        async for event in agent.run(user_message=user_message, session_id=session_id):
            yield event


# Pre-defined default compactor
DEFAULT_COMPACTOR = Compactor()


def generate_compaction_session_id(session_id: str) -> str:
    """Generate a unique session ID for compaction.

    Args:
        session_id: Original session ID

    Returns:
        Unique session ID for the compaction session
    """
    return f"{session_id}-compaction-{uuid.uuid4().hex[:8]}"
