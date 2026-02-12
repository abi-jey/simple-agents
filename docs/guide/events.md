# Events

nagents uses an event-based streaming architecture. All events include optional usage information.

## Event Types

| Event | Description |
|-------|-------------|
| `TextChunkEvent` | Streaming text chunk |
| `TextDoneEvent` | Complete text (non-streaming) |
| `ToolCallEvent` | Model is calling a tool |
| `ToolResultEvent` | Tool execution completed |
| `ErrorEvent` | Error occurred |
| `DoneEvent` | Generation complete |

## Base Event

All events inherit from `Event` and include usage information:

```python
@dataclass
class Event:
    type: EventType
    timestamp: datetime
    usage: Usage  # Token usage (always present, never None)
```

## Usage Information

Usage is available on all events and includes both current generation and session totals:

```python
@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class Usage:
    prompt_tokens: int      # Current generation
    completion_tokens: int  # Current generation
    total_tokens: int       # Current generation
    session: TokenUsage | None  # Cumulative across tool rounds

    def has_usage(self) -> bool:
        """Check if this usage has any actual token counts."""
        ...
```

```python
async for event in agent.run("Hello"):
    # Usage is always present on events (never None)
    print(f"Current: {event.usage.total_tokens} tokens")
    if event.usage.session:
        print(f"Session total: {event.usage.session.total_tokens} tokens")
```

## TextChunkEvent

Emitted for each streaming text chunk:

```python
@dataclass
class TextChunkEvent:
    chunk: str  # The text chunk
```

```python
async for event in agent.run("Hello"):
    if isinstance(event, TextChunkEvent):
        print(event.chunk, end="", flush=True)
```

## TextDoneEvent

Emitted when complete text is available (non-streaming mode):

```python
@dataclass
class TextDoneEvent:
    text: str  # Complete text
```

## ToolCallEvent

Emitted when the model calls a tool:

```python
@dataclass
class ToolCallEvent:
    id: str               # Unique call identifier
    name: str             # Tool function name
    arguments: dict       # Tool arguments
```

```python
if isinstance(event, ToolCallEvent):
    print(f"Calling {event.name} with {event.arguments}")
```

## ToolResultEvent

Emitted after tool execution:

```python
@dataclass
class ToolResultEvent:
    id: str               # Matches ToolCallEvent.id
    name: str             # Tool function name
    result: Any           # Tool return value
    error: str | None     # Error message if failed
    duration_ms: float    # Execution time in milliseconds
```

```python
if isinstance(event, ToolResultEvent):
    if event.error:
        print(f"Tool failed: {event.error}")
    else:
        print(f"Tool result: {event.result} ({event.duration_ms}ms)")
```

## ErrorEvent

Emitted when an error occurs:

```python
@dataclass
class ErrorEvent:
    message: str           # Error message
    code: str | None       # Error code
    recoverable: bool      # Can the conversation continue?
```

## DoneEvent

Emitted when generation is complete:

```python
@dataclass
class DoneEvent:
    final_text: str        # Complete response text
    session_id: str | None # Session ID used
```

## Complete Example

```python
async for event in agent.run("What's 2+2?"):
    match event:
        case TextChunkEvent(chunk=chunk):
            print(chunk, end="")
        case ToolCallEvent(name=name, arguments=args):
            print(f"\n[Calling {name}...]")
        case ToolResultEvent(result=result, duration_ms=ms):
            print(f"[Result: {result} in {ms}ms]")
        case ErrorEvent(message=msg):
            print(f"\n[Error: {msg}]")
        case DoneEvent(final_text=text, usage=usage):
            print(f"\n[Done: {len(text)} chars]")
            if usage.session:
                print(f"[Total tokens: {usage.session.total_tokens}]")
```
