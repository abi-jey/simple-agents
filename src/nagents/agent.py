"""
Main orchestrator for LLM interactions with auto tool execution.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .events import DoneEvent
from .events import ErrorEvent
from .events import Event
from .events import TextChunkEvent
from .events import TextDoneEvent
from .events import TokenUsage
from .events import ToolCallEvent
from .events import Usage
from .exceptions import ToolHallucinationError
from .http import FileHTTPLogger
from .provider import Provider
from .session import SessionManager
from .tools import ToolExecutor
from .tools import ToolRegistry
from .types import GenerationConfig
from .types import Message
from .types import ToolCall
from .types import ToolDefinition

logger = logging.getLogger(__name__)


class Agent:
    """
    Main orchestrator for LLM interactions with automatic tool execution.

    Coordinates the provider, session management, and tool execution to
    provide a complete conversational interface.

    The agent auto-initializes on first `run()` call, verifying the model
    and setting up the session database. You can also call `initialize()`
    manually if you want to catch initialization errors early.

    Example:
        provider = Provider(...)
        session = SessionManager(Path("sessions.db"))

        agent = Agent(
            provider=provider,
            session_manager=session,
            tools=[my_tool_func],
            system_prompt="You are a helpful assistant.",
        )

        # Just use it - auto-initializes on first run()
        async for event in agent.run("Hello!"):
            if isinstance(event, TextChunkEvent):
                print(event.chunk, end="")
            elif isinstance(event, ToolCallEvent):
                print(f"Calling tool: {event.name}")
            elif isinstance(event, DoneEvent):
                print(f"\\nSession: {event.session_id}")

        # Don't forget to close when done
        await agent.close()

        # Or pre-initialize to catch errors early:
        await agent.initialize()
        async for event in agent.run("Hello!"):
            ...
    """

    def __init__(
        self,
        provider: Provider,
        session_manager: SessionManager,
        tools: list[Callable[..., Any]] | None = None,
        system_prompt: str | None = None,
        max_tool_rounds: int = 10,
        streaming: bool = False,
        log_file: Path | str | None = None,
        fail_on_invalid_tool: bool = False,
    ):
        """
        Initialize the agent.

        Args:
            provider: The LLM provider to use
            session_manager: Session manager for conversation history
            tools: Optional list of tool functions to register
            system_prompt: Optional system prompt to prepend to conversations
            max_tool_rounds: Maximum number of tool execution rounds
            streaming: Whether to stream responses (default: False)
            log_file: Optional path to log file for HTTP/SSE traffic debugging.
                      Creates parent directories if they don't exist.
            fail_on_invalid_tool: If True, emit an error and stop when the LLM
                                  tries to call a tool that doesn't exist (tool
                                  hallucination). If False (default), the error
                                  is passed back to the LLM so it can recover.
        """
        self.provider = provider
        self.session = session_manager
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.streaming = streaming
        self.fail_on_invalid_tool = fail_on_invalid_tool

        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)

        self._initialized = False
        self._http_logger: FileHTTPLogger | None = None

        # Set up HTTP logging if log_file is provided
        if log_file:
            log_path = Path(log_file) if isinstance(log_file, str) else log_file
            self._http_logger = FileHTTPLogger(log_path)
            self.provider.set_http_logger(self._http_logger)
            logger.info(f"HTTP logging enabled: {log_path}")

        if tools:
            for tool in tools:
                self.tool_registry.register(tool)

    @property
    def is_initialized(self) -> bool:
        """Check if the agent has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """
        Initialize the agent (called automatically on first run()).

        This method:
        1. Initializes the session manager database
        2. Verifies the model exists with the provider

        You can call this manually to catch initialization errors early,
        otherwise it's called automatically on the first run().

        Raises:
            ValueError: If model verification fails
        """
        if self._initialized:
            logger.debug("Agent already initialized")
            return

        # Initialize session manager
        await self.session.initialize()
        logger.debug("Session manager initialized")

        # Always verify model
        if not await self.provider.verify_model():
            raise ValueError(
                f"Model '{self.provider.model}' not found or not accessible. Check your API key and model name."
            )
        logger.debug(f"Model verified: {self.provider.model}")

        self._initialized = True
        logger.info("Agent initialized successfully")

    async def _ensure_initialized(self) -> None:
        """Ensure agent is initialized, auto-initialize if not."""
        if not self._initialized:
            await self.initialize()

    def register_tool(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> ToolDefinition:
        """
        Register a tool function.

        Args:
            func: The function to register
            name: Optional override for tool name
            description: Optional override for description
        """
        return self.tool_registry.register(func, name, description)

    async def run(
        self,
        user_message: str,
        session_id: str | None = None,
        user_id: str = "default",
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[Event]:
        """
        Run a complete interaction with automatic tool execution.

        Auto-initializes the agent on first call (verifies model, sets up database).

        Yields all events (text chunks, tool calls, tool results, etc.)
        as they occur. Tool calls are automatically executed and their
        results fed back to the model.

        Args:
            user_message: The user's message
            session_id: Optional session identifier. If provided, verifies it exists
                        or creates a new one. If None, creates a new session.
            user_id: User identifier (default: "default")
            config: Optional generation configuration

        Yields:
            Events as they occur

        Raises:
            ValueError: If model verification fails during auto-initialization
        """
        await self._ensure_initialized()

        # Handle session ID
        if session_id is None:
            # Create new session with generated ID
            session_id = f"session-{uuid.uuid4().hex[:12]}"
            await self.session.get_or_create_session(session_id, user_id)
            logger.info(f"Created new session: {session_id}")
        else:
            # Verify session exists or create it
            if await self.session.session_exists(session_id):
                logger.debug(f"Using existing session: {session_id}")
            else:
                logger.info(f"Session '{session_id}' not found, creating new session")
                await self.session.get_or_create_session(session_id, user_id)

        # Add user message to history
        await self.session.add_message(session_id, Message(role="user", content=user_message))

        # Set session ID for HTTP logging
        self.provider.set_session_id(session_id)

        tools = self.tool_registry.get_all() if self.tool_registry.has_tools() else None

        # Track last known usage to use for events without usage data
        last_usage = Usage()

        for round_num in range(self.max_tool_rounds):
            logger.debug(f"Generation round {round_num + 1}/{self.max_tool_rounds}")

            # Get current history
            history = await self.session.get_history(session_id)

            # Prepend system prompt if set
            messages: list[Message] = []
            if self.system_prompt:
                messages.append(Message(role="system", content=self.system_prompt))
            messages.extend(history)

            # Generate response
            pending_tool_calls: list[ToolCall] = []
            full_text = ""
            has_error = False

            async for event in self.provider.generate(
                messages=messages,
                tools=tools,
                config=config,
                stream=self.streaming,
            ):
                # Track usage from events that have actual token counts
                if event.usage.has_usage():
                    last_usage = Usage(
                        prompt_tokens=event.usage.prompt_tokens,
                        completion_tokens=event.usage.completion_tokens,
                        total_tokens=event.usage.total_tokens,
                    )
                else:
                    # Use last known usage for events without usage data
                    event.usage = Usage(
                        prompt_tokens=last_usage.prompt_tokens,
                        completion_tokens=last_usage.completion_tokens,
                        total_tokens=last_usage.total_tokens,
                    )

                # For now, session equals current usage (TODO: accumulate across compactions)
                event.usage.session = TokenUsage(
                    prompt_tokens=last_usage.prompt_tokens,
                    completion_tokens=last_usage.completion_tokens,
                    total_tokens=last_usage.total_tokens,
                )

                yield event  # Always emit events to caller

                if isinstance(event, TextChunkEvent):
                    full_text += event.chunk
                elif isinstance(event, TextDoneEvent):
                    full_text = event.text
                elif isinstance(event, ToolCallEvent):
                    pending_tool_calls.append(
                        ToolCall(
                            id=event.id,
                            name=event.name,
                            arguments=event.arguments,
                        )
                    )
                elif isinstance(event, ErrorEvent):
                    has_error = True
                    if not event.recoverable:
                        yield DoneEvent(
                            final_text="",
                            session_id=session_id,
                            usage=Usage(
                                prompt_tokens=last_usage.prompt_tokens,
                                completion_tokens=last_usage.completion_tokens,
                                total_tokens=last_usage.total_tokens,
                                session=TokenUsage(
                                    prompt_tokens=last_usage.prompt_tokens,
                                    completion_tokens=last_usage.completion_tokens,
                                    total_tokens=last_usage.total_tokens,
                                ),
                            ),
                        )
                        return

            # If we hit an error, don't continue
            if has_error:
                continue

            # If we got tool calls, execute them
            if pending_tool_calls:
                # Add assistant message with tool calls to history
                await self.session.add_message(
                    session_id,
                    Message(role="assistant", content=None, tool_calls=pending_tool_calls),
                )

                # Execute each tool
                for tool_call in pending_tool_calls:
                    logger.debug(f"Executing tool: {tool_call.name}")
                    result_event = await self.tool_executor.execute(tool_call)
                    # Attach last known usage info to tool result events
                    result_event.usage = Usage(
                        prompt_tokens=last_usage.prompt_tokens,
                        completion_tokens=last_usage.completion_tokens,
                        total_tokens=last_usage.total_tokens,
                        session=TokenUsage(
                            prompt_tokens=last_usage.prompt_tokens,
                            completion_tokens=last_usage.completion_tokens,
                            total_tokens=last_usage.total_tokens,
                        ),
                    )
                    yield result_event

                    # Check for invalid tool (hallucination) and raise if configured
                    if result_event.error and self.fail_on_invalid_tool and "does not exist" in result_event.error:
                        logger.error(f"Tool hallucination detected: {tool_call.name}")
                        raise ToolHallucinationError(
                            tool_name=tool_call.name,
                            available_tools=self.tool_registry.names(),
                            message=result_event.error,
                        )

                    # Add tool result to history
                    result_content = (
                        str(result_event.result) if result_event.error is None else f"Error: {result_event.error}"
                    )
                    await self.session.add_message(
                        session_id,
                        Message(
                            role="tool",
                            content=result_content,
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        ),
                    )

                # Continue to next round (model will see tool results)
                continue

            # No tool calls - we're done
            if full_text:
                await self.session.add_message(session_id, Message(role="assistant", content=full_text))

            yield DoneEvent(
                final_text=full_text,
                session_id=session_id,
                usage=Usage(
                    prompt_tokens=last_usage.prompt_tokens,
                    completion_tokens=last_usage.completion_tokens,
                    total_tokens=last_usage.total_tokens,
                    session=TokenUsage(
                        prompt_tokens=last_usage.prompt_tokens,
                        completion_tokens=last_usage.completion_tokens,
                        total_tokens=last_usage.total_tokens,
                    ),
                ),
            )
            return

        # Hit max rounds
        logger.warning(f"Max tool rounds ({self.max_tool_rounds}) exceeded")
        yield ErrorEvent(
            message=f"Max tool rounds ({self.max_tool_rounds}) exceeded",
            recoverable=False,
            usage=Usage(
                prompt_tokens=last_usage.prompt_tokens,
                completion_tokens=last_usage.completion_tokens,
                total_tokens=last_usage.total_tokens,
                session=TokenUsage(
                    prompt_tokens=last_usage.prompt_tokens,
                    completion_tokens=last_usage.completion_tokens,
                    total_tokens=last_usage.total_tokens,
                ),
            ),
        )
        yield DoneEvent(
            final_text="",
            session_id=session_id,
            usage=Usage(
                prompt_tokens=last_usage.prompt_tokens,
                completion_tokens=last_usage.completion_tokens,
                total_tokens=last_usage.total_tokens,
                session=TokenUsage(
                    prompt_tokens=last_usage.prompt_tokens,
                    completion_tokens=last_usage.completion_tokens,
                    total_tokens=last_usage.total_tokens,
                ),
            ),
        )

    async def run_simple(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[Event]:
        """
        Run a simple generation without session management.

        Auto-initializes the agent on first call.

        Useful for one-off generations or when managing history externally.
        Does NOT automatically execute tools.

        Args:
            messages: List of messages to send
            config: Optional generation configuration

        Yields:
            Events as they occur
        """
        await self._ensure_initialized()

        tools = self.tool_registry.get_all() if self.tool_registry.has_tools() else None

        # For simple runs, session_usage equals usage (single generation)
        session_usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        # Track last known usage to use for events without usage data
        last_usage = Usage()

        async for event in self.provider.generate(
            messages=messages,
            tools=tools,
            config=config,
            stream=self.streaming,
        ):
            # Track usage from events that have actual token counts
            if event.usage.has_usage():
                # For simple runs, session_usage = current usage
                session_usage = TokenUsage(
                    prompt_tokens=event.usage.prompt_tokens,
                    completion_tokens=event.usage.completion_tokens,
                    total_tokens=event.usage.total_tokens,
                )
                last_usage = Usage(
                    prompt_tokens=event.usage.prompt_tokens,
                    completion_tokens=event.usage.completion_tokens,
                    total_tokens=event.usage.total_tokens,
                )
            else:
                # Use last known usage for events without usage data
                event.usage = Usage(
                    prompt_tokens=last_usage.prompt_tokens,
                    completion_tokens=last_usage.completion_tokens,
                    total_tokens=last_usage.total_tokens,
                )

            # Always attach session usage to the event
            event.usage.session = session_usage

            yield event

    async def clear_session(self, session_id: str) -> None:
        """
        Clear a session's conversation history.

        Args:
            session_id: Session identifier to clear
        """
        await self._ensure_initialized()
        await self.session.clear_session(session_id)

    async def close(self) -> None:
        """Close the agent and release resources."""
        await self.provider.close()
        self._initialized = False
