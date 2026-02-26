"""
Main orchestrator for LLM interactions with auto tool execution.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from .batch import BatchClient
from .batch import BatchConfig
from .batch import BatchRequest
from .batch import BatchStatus
from .events import DoneEvent
from .events import ErrorEvent
from .events import Event
from .events import FinishReason
from .events import TextChunkEvent
from .events import TextDoneEvent
from .events import TokenUsage
from .events import ToolCallEvent
from .events import Usage
from .exceptions import ToolHallucinationError
from .http import FileHTTPLogger
from .http import HTTPError
from .media import transcode_audio_to_wav
from .provider import Provider
from .provider import ProviderType
from .session import SessionManager
from .tools import ToolExecutor
from .tools import ToolRegistry
from .types import AudioContent
from .types import ContentPart
from .types import GenerationConfig
from .types import Message
from .types import TextContent
from .types import ToolCall
from .types import ToolDefinition

if TYPE_CHECKING:
    from .stt import STTService

logger = logging.getLogger(__name__)


class UnsupportedAudioBehavior(Enum):
    """Controls how the agent handles audio sent to models that don't support inline audio input.

    - RAISE: Raise an error immediately (default). Safest option — caller must handle audio explicitly.
    - DROP: Silently drop the audio and replace with a text warning in the message.
    - TRANSCRIBE: Auto-transcribe the audio to text using the configured STT service.
                   Requires an ``stt_service`` to be provided to the Agent.
    """

    RAISE = "raise"
    DROP = "drop"
    TRANSCRIBE = "transcribe"


class UnsupportedAudioError(Exception):
    """Raised when audio content is sent to a model that does not support inline audio input."""

    def __init__(self, model: str, audio_format: str):
        self.model = model
        self.audio_format = audio_format
        super().__init__(
            f"Model '{model}' does not support inline audio input (format: '{audio_format}'). "
            "Set unsupported_audio='drop' to silently omit audio, "
            "or unsupported_audio='transcribe' with an stt_service to auto-transcribe."
        )


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

    Batch Mode:
        When batch=True, the agent uses batch processing API (50% cost discount).
        The interface remains the same - just add batch=True:

        agent = Agent(
            provider=provider,
            session_manager=session,
            batch=True,  # Enable batch mode
        )

        async for event in agent.run("Hello!"):
            # Events are yielded after batch completes
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
        batch: bool = False,
        batch_poll_interval: float = 30.0,
        stt_service: "STTService | None" = None,
        unsupported_audio: UnsupportedAudioBehavior = UnsupportedAudioBehavior.RAISE,
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
            batch: If True, use batch processing API (50% cost discount).
                   Results are delayed but significantly cheaper.
            batch_poll_interval: Seconds between batch status checks (default: 30)
            stt_service: Optional Speech-to-Text service for audio transcription.
                         If provided, audio content in messages will be transcribed
                         to text before being sent to the LLM.
            unsupported_audio: How to handle audio sent to models that don't support
                               inline audio input. Options:
                               - RAISE (default): Raise UnsupportedAudioError
                               - DROP: Replace audio with a text warning
                               - TRANSCRIBE: Auto-transcribe using stt_service
                                 (requires stt_service to be set)
        """
        self.provider = provider
        self.session = session_manager
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.streaming = streaming
        self.fail_on_invalid_tool = fail_on_invalid_tool
        self.batch = batch
        self.batch_poll_interval = batch_poll_interval
        self.stt_service = stt_service
        self.unsupported_audio = unsupported_audio

        # Validate that TRANSCRIBE mode has an STT service
        if self.unsupported_audio == UnsupportedAudioBehavior.TRANSCRIBE and not self.stt_service:
            raise ValueError(
                "unsupported_audio='transcribe' requires an stt_service to be provided. "
                "Use OpenAISTTService or GeminiSTTService."
            )

        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)

        self._initialized = False
        self._http_logger: FileHTTPLogger | None = None
        self._batch_client: BatchClient | None = None

        # Set up HTTP logging if log_file is provided
        if log_file:
            log_path = Path(log_file) if isinstance(log_file, str) else log_file
            self._http_logger = FileHTTPLogger(log_path)
            self.provider.set_http_logger(self._http_logger)
            logger.info(f"HTTP logging enabled: {log_path}")

        if tools:
            for tool in tools:
                self.tool_registry.register(tool)

        # Initialize batch client if batch mode enabled
        if self.batch:
            self._init_batch_client()

    def _init_batch_client(self) -> None:
        """Initialize the batch client for batch mode."""
        valid_types = (
            ProviderType.OPENAI_COMPATIBLE,
            ProviderType.AZURE_OPENAI_COMPATIBLE,
            ProviderType.AZURE_OPENAI_COMPATIBLE_V1,
            ProviderType.ANTHROPIC,
        )
        if self.provider.provider_type not in valid_types:
            raise ValueError(
                f"Batch mode only supported for OPENAI_COMPATIBLE, AZURE_OPENAI, and ANTHROPIC providers. "
                f"Got: {self.provider.provider_type}"
            )
        self._batch_client = BatchClient(
            provider_type=self.provider.provider_type,
            api_key=self.provider.api_key,
            model=self.provider.model,
            base_url=self.provider.base_url,
            logger=self._http_logger,
        )
        logger.info("Batch mode enabled - requests will be processed with 50% cost discount")

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

    async def _process_multimodal_content(
        self,
        content: str | list[ContentPart] | None,
    ) -> str | list[ContentPart] | None:
        """
        Process multimodal content, handling audio based on provider capabilities
        and the ``unsupported_audio`` configuration.

        Processing order for each AudioContent part:

        1. If provider supports the audio format natively → pass through.
        2. If provider supports audio but not *this* format → transcode to WAV.
        3. If provider does NOT support audio at all → apply ``unsupported_audio`` policy:
           - RAISE: raise UnsupportedAudioError
           - DROP:  replace with a TextContent warning
           - TRANSCRIBE: use stt_service to transcribe to text

        When ``stt_service`` is set independently (without ``unsupported_audio=TRANSCRIBE``),
        audio is transcribed only when the provider doesn't support the format and
        transcoding also isn't an option.

        Args:
            content: Message content (str, list of ContentParts, or None)

        Returns:
            Processed content with audio handled appropriately

        Raises:
            UnsupportedAudioError: If unsupported_audio is RAISE and audio is sent
                                    to a model that doesn't support it
        """
        if content is None or isinstance(content, str):
            return content

        capabilities = self.provider.supported_media_formats
        processed_parts: list[ContentPart] = []

        for part in content:
            if not isinstance(part, AudioContent):
                processed_parts.append(part)
                continue

            # Case 1: Provider supports this exact audio format → pass through
            if capabilities.supports_audio(part.format):
                processed_parts.append(part)
                continue

            # Case 2: Provider supports audio, but not this format → transcode to WAV
            if capabilities.audio_formats:
                logger.info(
                    f"Audio format '{part.format}' not supported by provider "
                    f"(supported: {sorted(capabilities.audio_formats)}). "
                    "Transcoding to WAV via ffmpeg..."
                )
                try:
                    wav_data = await transcode_audio_to_wav(part.base64_data, part.format)
                    processed_parts.append(AudioContent(base64_data=wav_data, format="wav"))
                    logger.info(f"Transcoded audio from '{part.format}' to 'wav' successfully")
                except RuntimeError as e:
                    logger.error(f"Audio transcoding failed: {e}")
                    processed_parts.append(TextContent(text=f"[Audio transcoding failed: {e}]"))
                continue

            # Case 3: Provider does NOT support audio at all → apply policy
            if self.unsupported_audio == UnsupportedAudioBehavior.RAISE:
                raise UnsupportedAudioError(self.provider.model, part.format)

            elif self.unsupported_audio == UnsupportedAudioBehavior.DROP:
                logger.warning(
                    f"Model '{self.provider.model}' does not support inline audio input. "
                    "Audio content dropped (unsupported_audio='drop')."
                )
                processed_parts.append(
                    TextContent(
                        text="[Voice message received, but this model does not support audio input. "
                        "The audio content was omitted.]"
                    )
                )

            elif self.unsupported_audio == UnsupportedAudioBehavior.TRANSCRIBE:
                # stt_service is guaranteed present (validated in __init__)
                assert self.stt_service is not None
                logger.info("Auto-transcribing audio (unsupported_audio='transcribe')...")
                try:
                    result = await self.stt_service.transcribe_base64(
                        part.base64_data,
                        part.format,
                    )
                    logger.info(f"Transcription: {result.text[:100]}...")
                    if result.text:
                        processed_parts.append(TextContent(text=result.text))
                    else:
                        processed_parts.append(TextContent(text="[Audio transcription returned empty]"))
                except Exception as e:
                    logger.error(f"Failed to transcribe audio: {e}")
                    processed_parts.append(TextContent(text=f"[Audio transcription failed: {e}]"))

        return processed_parts

    def _filter_unsupported_audio(self, messages: list[Message]) -> list[Message]:
        """
        Filter unsupported AudioContent from message history.

        When a model doesn't support inline audio (empty audio_formats),
        replaces AudioContent parts in *history* messages with TextContent warnings.
        This prevents 400 errors from old audio in session history being sent
        to non-audio models.

        Only applies when ``unsupported_audio`` is DROP or TRANSCRIBE.
        When RAISE, history audio should not exist (since the initial send would
        have raised), but we still filter defensively.
        """
        capabilities = self.provider.supported_media_formats
        if capabilities.audio_formats:
            # Model supports audio — no filtering needed
            return messages

        filtered: list[Message] = []
        for msg in messages:
            if not isinstance(msg.content, list):
                filtered.append(msg)
                continue

            has_audio = any(isinstance(p, AudioContent) for p in msg.content)
            if not has_audio:
                filtered.append(msg)
                continue

            # Replace AudioContent with text warning
            new_parts: list[ContentPart] = []
            for part in msg.content:
                if isinstance(part, AudioContent):
                    new_parts.append(
                        TextContent(
                            text="[Voice message received, but this model does not support audio input. "
                            "The audio content was omitted.]"
                        )
                    )
                else:
                    new_parts.append(part)

            filtered.append(
                Message(
                    role=msg.role,
                    content=new_parts,
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            )
        return filtered

    async def run(
        self,
        user_message: str | list[ContentPart] | Message,
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
            user_message: The user's message. Can be:
                          - str: Simple text message
                          - list[ContentPart]: Multimodal content (text, images, audio)
                          - Message: Full message object
                          If audio content is present and stt_service is configured,
                          audio will be transcribed to text.
            session_id: Optional session identifier. If provided, verifies it exists
                        or creates a new one. If None, creates a new session.
            user_id: User identifier (default: "default")
            config: Optional generation configuration

        Yields:
            Events as they occur

        Raises:
            ValueError: If model verification fails during auto-initialization
            UnsupportedAudioError: If audio content is sent to a model that doesn't
                                    support it and unsupported_audio is RAISE (default)
        """
        await self._ensure_initialized()

        # Normalize user_message to a Message object
        message = Message(role="user", content=user_message) if isinstance(user_message, str | list) else user_message

        # Process audio content through STT if present
        processed_content = await self._process_multimodal_content(message.content)

        # Create processed message
        processed_message = Message(
            role=message.role,
            content=processed_content,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            name=message.name,
        )

        # Use batch mode if enabled
        if self.batch:
            async for event in self._run_batch(processed_message, session_id, user_id, config):
                yield event
            return

        # Handle session ID
        if session_id is None:
            session_id = f"session-{uuid.uuid4().hex[:12]}"
            await self.session.get_or_create_session(session_id, user_id)
            logger.info(f"Created new session: {session_id}")
        else:
            if await self.session.session_exists(session_id):
                logger.debug(f"Using existing session: {session_id}")
            else:
                logger.info(f"Session '{session_id}' not found, creating new session")
                await self.session.get_or_create_session(session_id, user_id)

        # Add user message to history
        await self.session.add_message(session_id, processed_message)

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

            # Filter unsupported audio from all messages (including history)
            messages = self._filter_unsupported_audio(messages)

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
                            metadata=event.metadata,
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

    async def _run_batch(
        self,
        user_message: Message,
        session_id: str | None,
        user_id: str | None,
        config: GenerationConfig | None,
    ) -> AsyncIterator[Event]:
        """Run with batch processing for cost-effective large-scale operations.

        This method submits the request as a batch job, waits for completion,
        and yields events from the results. Supports multi-turn tool execution
        by submitting subsequent batches with tool results.
        """
        if not self._batch_client:
            yield ErrorEvent(message="Batch client not initialized")
            return

        # Handle session ID
        if session_id is None:
            session_id = f"batch-session-{uuid.uuid4().hex[:12]}"
            await self.session.get_or_create_session(session_id, user_id or "default")
            logger.info(f"Created new batch session: {session_id}")
        else:
            if not await self.session.session_exists(session_id):
                await self.session.get_or_create_session(session_id, user_id or "default")

        # Add user message to history
        await self.session.add_message(session_id, user_message)

        # Set session ID for HTTP logging in batch client
        if self._batch_client:
            self._batch_client.set_session_id(session_id)

        # Track accumulated usage and final text
        total_usage = Usage()
        full_text = ""
        finish_reason = FinishReason.UNKNOWN

        # Loop until no more tool calls
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"Batch iteration {iteration}")

            # Get current history for the batch request
            history = await self.session.get_history(session_id)

            # Build messages for batch request
            messages: list[Message] = []
            if self.system_prompt:
                messages.append(Message(role="system", content=self.system_prompt))
            messages.extend(history)

            # Create batch request
            batch_request = BatchRequest(
                custom_id=f"{session_id}-{uuid.uuid4().hex[:8]}",
                messages=messages,
                config=config,
            )

            # Submit batch
            tools = self.tool_registry.get_all() if self.tool_registry.has_tools() else None
            logger.info(f"Submitting batch request: {batch_request.custom_id}")

            try:
                job = await self._batch_client.create_batch(
                    requests=[batch_request],
                    config=BatchConfig(),
                    tools=tools,
                    generation_config=config,
                )
                logger.info(f"Batch job created: {job.id}, status: {job.status}")
            except HTTPError as e:
                error_msg = f"Failed to create batch: {e}"
                if e.body:
                    error_msg += f"\nResponse: {e.body}"
                logger.error(error_msg)
                yield ErrorEvent(message=error_msg)
                return
            except Exception as e:
                error_msg = f"Failed to create batch: {e}"
                logger.error(error_msg)
                yield ErrorEvent(message=error_msg)
                return

            # Wait for completion
            logger.info("Waiting for batch completion...")
            job = await self._batch_client.wait_for_completion(
                job.id,
                poll_interval=self.batch_poll_interval,
            )
            logger.info(f"Batch completed: {job.status}")

            # Check status
            if job.status != BatchStatus.COMPLETED:
                yield ErrorEvent(
                    message=f"Batch job failed with status: {job.status}",
                    code=str(job.status.value),
                )
                return

            # Get results
            iteration_text = ""
            iteration_usage = Usage()
            iteration_finish_reason = FinishReason.UNKNOWN
            tool_calls_to_execute: list[ToolCall] = []

            async for result in self._batch_client.get_results(job):
                if result.result_type == "succeeded":
                    iteration_finish_reason = result.finish_reason

                    # Parse detailed usage
                    if result.usage:
                        iteration_usage = Usage(
                            prompt_tokens=result.usage.get("prompt_tokens", 0),
                            completion_tokens=result.usage.get("completion_tokens", 0),
                            total_tokens=result.usage.get("total_tokens", 0),
                            cached_tokens=result.usage.get("cached_tokens", 0),
                            audio_tokens=result.usage.get("audio_tokens", 0),
                            reasoning_tokens=result.usage.get("reasoning_tokens", 0),
                        )
                        # Accumulate usage
                        total_usage.prompt_tokens += iteration_usage.prompt_tokens
                        total_usage.completion_tokens += iteration_usage.completion_tokens
                        total_usage.total_tokens += iteration_usage.total_tokens
                        total_usage.cached_tokens += iteration_usage.cached_tokens
                        total_usage.audio_tokens += iteration_usage.audio_tokens
                        total_usage.reasoning_tokens += iteration_usage.reasoning_tokens

                    # Handle tool calls
                    if result.tool_calls:
                        for tc in result.tool_calls:
                            import json as json_mod

                            from .types import ToolArguments

                            args_str = tc.get("function", {}).get("arguments", "{}")
                            try:
                                args: ToolArguments = (
                                    json_mod.loads(args_str) if isinstance(args_str, str) else args_str
                                )
                            except json_mod.JSONDecodeError:
                                args = {"raw": args_str}

                            tool_call = ToolCall(
                                id=tc.get("id", ""),
                                name=tc.get("function", {}).get("name", ""),
                                arguments=args,
                            )
                            tool_calls_to_execute.append(tool_call)
                            yield ToolCallEvent(
                                id=tool_call.id,
                                name=tool_call.name,
                                arguments=tool_call.arguments,
                                usage=iteration_usage,
                                finish_reason=FinishReason.TOOL_CALLS,
                                metadata=tool_call.metadata,
                            )

                    # Handle text content
                    if result.content:
                        iteration_text = result.content
                        yield TextDoneEvent(
                            text=result.content,
                            usage=iteration_usage,
                            finish_reason=iteration_finish_reason,
                        )

                elif result.result_type == "errored":
                    yield ErrorEvent(
                        message=result.error_message or "Unknown batch error",
                        code=result.error_type,
                    )
                    return

            # Check if we have tool calls to execute
            if tool_calls_to_execute:
                # Execute tools
                for tool_call in tool_calls_to_execute:
                    result_event = await self.tool_executor.execute(tool_call)
                    yield result_event

                    # Add tool result to history
                    await self.session.add_message(
                        session_id,
                        Message(
                            role="tool",
                            content=str(result_event.result)
                            if result_event.result is not None
                            else (result_event.error or "Error"),
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        ),
                    )

                # Continue the loop - will submit new batch with tool results
                logger.info(
                    f"Batch iteration {iteration} completed with {len(tool_calls_to_execute)} tool calls, continuing..."
                )
                continue

            # No tool calls - we're done
            full_text = iteration_text
            finish_reason = iteration_finish_reason
            break

        # Add final assistant message to history
        if full_text:
            await self.session.add_message(session_id, Message(role="assistant", content=full_text))

        yield DoneEvent(
            final_text=full_text,
            session_id=session_id,
            usage=total_usage,
            finish_reason=finish_reason,
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
        if self._batch_client:
            await self._batch_client.close()
        self._initialized = False
