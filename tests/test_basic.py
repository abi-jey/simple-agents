"""Basic tests for nagents package."""

from pathlib import Path
from tempfile import TemporaryDirectory

from nagents import Agent
from nagents import DoneEvent
from nagents import ErrorEvent
from nagents import FileHTTPLogger
from nagents import GenerationConfig
from nagents import Message
from nagents import Provider
from nagents import ProviderType
from nagents import SessionManager
from nagents import TextChunkEvent
from nagents import TextDoneEvent
from nagents import TokenUsage
from nagents import ToolCallEvent
from nagents import ToolDefinition
from nagents import ToolResultEvent
from nagents import Usage


class TestTypes:
    """Test core type definitions."""

    def test_message_creation(self) -> None:
        """Test Message dataclass creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls == []
        assert msg.tool_call_id is None

    def test_message_with_tool_calls(self) -> None:
        """Test Message with tool calls."""
        from nagents import ToolCall

        tool_call = ToolCall(id="123", name="test_tool", arguments={"arg": "value"})
        msg = Message(role="assistant", content="", tool_calls=[tool_call])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "test_tool"

    def test_tool_definition(self) -> None:
        """Test ToolDefinition creation."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
            },
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_generation_config(self) -> None:
        """Test GenerationConfig creation."""
        config = GenerationConfig(
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
        )
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 0.9


class TestEvents:
    """Test event types."""

    def test_text_chunk_event(self) -> None:
        """Test TextChunkEvent creation."""
        event = TextChunkEvent(chunk="Hello")
        assert event.chunk == "Hello"

    def test_text_done_event(self) -> None:
        """Test TextDoneEvent creation."""
        event = TextDoneEvent(text="Complete message")
        assert event.text == "Complete message"

    def test_tool_call_event(self) -> None:
        """Test ToolCallEvent creation."""
        event = ToolCallEvent(
            id="123",
            name="test_tool",
            arguments={"arg": "value"},
        )
        assert event.id == "123"
        assert event.name == "test_tool"
        assert event.arguments == {"arg": "value"}

    def test_tool_result_event(self) -> None:
        """Test ToolResultEvent creation."""
        event = ToolResultEvent(
            id="123",
            name="test_tool",
            result="success",
            duration_ms=10.5,
        )
        assert event.id == "123"
        assert event.result == "success"
        assert event.duration_ms == 10.5
        assert event.error is None

    def test_usage(self) -> None:
        """Test Usage creation with session."""
        session = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            session=session,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.has_usage() is True
        assert usage.session is not None
        assert usage.session.total_tokens == 300

    def test_usage_default(self) -> None:
        """Test Usage default values."""
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.has_usage() is False
        assert usage.session is None

    def test_event_with_usage(self) -> None:
        """Test Event with usage containing session."""
        session = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            session=session,
        )
        event = TextChunkEvent(chunk="Hello", usage=usage)
        assert event.usage.prompt_tokens == 100
        assert event.usage.session is not None
        assert event.usage.session.total_tokens == 300

    def test_event_default_usage(self) -> None:
        """Test Event has default usage that is never None."""
        event = TextChunkEvent(chunk="Hello")
        # Usage is always present, never None
        assert event.usage.prompt_tokens == 0
        assert event.usage.has_usage() is False

    def test_error_event(self) -> None:
        """Test ErrorEvent creation."""
        event = ErrorEvent(
            message="Something went wrong",
            code="ERR001",
            recoverable=True,
        )
        assert event.message == "Something went wrong"
        assert event.code == "ERR001"
        assert event.recoverable is True

    def test_done_event(self) -> None:
        """Test DoneEvent creation."""
        event = DoneEvent(
            final_text="Complete response",
            session_id="session-123",
        )
        assert event.final_text == "Complete response"
        assert event.session_id == "session-123"


class TestProvider:
    """Test Provider configuration."""

    def test_provider_creation_openai(self) -> None:
        """Test OpenAI provider creation."""
        provider = Provider(
            provider_type=ProviderType.OPENAI_COMPATIBLE,
            api_key="test-key",
            model="gpt-4o-mini",
        )
        assert provider.provider_type == ProviderType.OPENAI_COMPATIBLE
        assert provider.model == "gpt-4o-mini"

    def test_provider_creation_gemini(self) -> None:
        """Test Gemini provider creation."""
        provider = Provider(
            provider_type=ProviderType.GEMINI_NATIVE,
            api_key="test-key",
            model="gemini-2.0-flash",
        )
        assert provider.provider_type == ProviderType.GEMINI_NATIVE
        assert provider.model == "gemini-2.0-flash"

    def test_provider_creation_anthropic(self) -> None:
        """Test Anthropic provider creation."""
        provider = Provider(
            provider_type=ProviderType.ANTHROPIC,
            api_key="test-key",
            model="claude-3-5-sonnet-20241022",
        )
        assert provider.provider_type == ProviderType.ANTHROPIC
        assert provider.model == "claude-3-5-sonnet-20241022"


class TestAgent:
    """Test Agent class."""

    def test_agent_creation(self) -> None:
        """Test Agent instantiation."""
        with TemporaryDirectory() as tmpdir:
            provider = Provider(
                provider_type=ProviderType.OPENAI_COMPATIBLE,
                api_key="test-key",
                model="gpt-4o-mini",
            )
            session_manager = SessionManager(Path(tmpdir) / "sessions.db")
            agent = Agent(provider=provider, session_manager=session_manager)
            assert agent.provider == provider

    def test_agent_with_tools(self) -> None:
        """Test Agent with tool functions."""

        def example_tool(arg: str) -> str:
            """An example tool."""
            return f"Result: {arg}"

        with TemporaryDirectory() as tmpdir:
            provider = Provider(
                provider_type=ProviderType.OPENAI_COMPATIBLE,
                api_key="test-key",
                model="gpt-4o-mini",
            )
            session_manager = SessionManager(Path(tmpdir) / "sessions.db")
            agent = Agent(
                provider=provider,
                session_manager=session_manager,
                tools=[example_tool],
            )
            # Check tool was registered
            assert agent.tool_registry is not None

    def test_agent_with_system_prompt(self) -> None:
        """Test Agent with system prompt."""
        with TemporaryDirectory() as tmpdir:
            provider = Provider(
                provider_type=ProviderType.OPENAI_COMPATIBLE,
                api_key="test-key",
                model="gpt-4o-mini",
            )
            session_manager = SessionManager(Path(tmpdir) / "sessions.db")
            agent = Agent(
                provider=provider,
                session_manager=session_manager,
                system_prompt="You are a helpful assistant.",
            )
            assert agent.system_prompt == "You are a helpful assistant."

    def test_agent_with_log_file(self) -> None:
        """Test Agent with HTTP logging enabled."""
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "http.log"
            provider = Provider(
                provider_type=ProviderType.OPENAI_COMPATIBLE,
                api_key="test-key",
                model="gpt-4o-mini",
            )
            session_manager = SessionManager(Path(tmpdir) / "sessions.db")
            agent = Agent(
                provider=provider,
                session_manager=session_manager,
                log_file=log_path,
            )
            # Check that log file and parent dirs were created
            assert log_path.exists()
            assert agent._http_logger is not None


class TestHTTPLogger:
    """Test HTTP logging functionality."""

    def test_file_logger_creates_dirs(self) -> None:
        """Test that FileHTTPLogger creates parent directories."""
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nested" / "dirs" / "http.log"
            FileHTTPLogger(log_path)  # Creates dirs on init
            assert log_path.exists()
            assert log_path.parent.exists()

    def test_file_logger_logs_request(self) -> None:
        """Test logging an HTTP request."""
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "http.log"
            logger = FileHTTPLogger(log_path)

            logger.log_request(
                method="POST",
                url="https://api.example.com/v1/chat",
                headers={"Authorization": "Bearer sk-1234567890abcdef", "Content-Type": "application/json"},
                body={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
                session_id="test-session-123",
            )

            content = log_path.read_text()
            assert "[test-session-123]" in content
            assert ">>> REQUEST" in content
            assert "POST" in content
            assert "https://api.example.com/v1/chat" in content
            # Check that Authorization header is sanitized (full key not in output)
            assert "sk-1234567890abcdef" not in content
            # Sanitized format: first 10 chars + "..." + last 4 chars
            assert "Bearer sk-..." in content

    def test_file_logger_logs_response(self) -> None:
        """Test logging an HTTP response."""
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "http.log"
            logger = FileHTTPLogger(log_path)

            logger.log_response(
                url="https://api.example.com/v1/chat",
                status=200,
                body={"choices": [{"message": {"content": "Hello!"}}]},
                session_id="test-session-123",
            )

            content = log_path.read_text()
            assert "[test-session-123]" in content
            assert "<<< RESPONSE" in content
            assert "200" in content
            assert "Hello!" in content

    def test_file_logger_logs_sse_chunk(self) -> None:
        """Test logging an SSE chunk."""
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "http.log"
            logger = FileHTTPLogger(log_path)

            logger.log_sse_chunk(
                url="https://api.example.com/v1/chat",
                data='{"choices": [{"delta": {"content": "Hi"}}]}',
                session_id="test-session-123",
            )

            content = log_path.read_text()
            assert "[test-session-123]" in content
            assert "<<< SSE" in content
            assert "Hi" in content

    def test_file_logger_no_session_id(self) -> None:
        """Test logging without a session ID uses 'no-session'."""
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "http.log"
            logger = FileHTTPLogger(log_path)

            logger.log_request(
                method="GET",
                url="https://api.example.com/models",
                headers={},
                body=None,
            )

            content = log_path.read_text()
            assert "[no-session]" in content
