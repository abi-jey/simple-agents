# Tools

Tools allow your agent to call Python functions during conversations.

## Defining Tools

Any Python function can be used as a tool. The function's docstring and type hints are used to generate the tool schema:

```python
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.
    """
    # Your implementation
    return f"Weather in {city}: Sunny, 22°C"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like "2 + 2" or "10 * 5"
    """
    return str(eval(expression))
```

## Registering Tools

Pass tools when creating the agent:

```python
agent = Agent(
    provider=provider,
    tools=[get_weather, calculate],
)
```

## Tool Execution Flow

1. Agent receives user message
2. Model decides to call a tool
3. `ToolCallEvent` is emitted
4. Tool function is executed
5. `ToolResultEvent` is emitted with result
6. Model continues with tool result

```python
async for event in agent.run("What's 15 * 7?"):
    if isinstance(event, ToolCallEvent):
        print(f"Calling: {event.name}({event.arguments})")
    elif isinstance(event, ToolResultEvent):
        print(f"Result: {event.result}")
    elif isinstance(event, TextChunkEvent):
        print(event.chunk, end="")
```

## Async Tools

Async functions are fully supported:

```python
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

agent = Agent(provider=provider, tools=[fetch_data])
```

## Tool Errors

If a tool raises an exception, it's captured and sent to the model:

```python
def risky_operation(param: str) -> str:
    """A tool that might fail."""
    if not param:
        raise ValueError("param cannot be empty")
    return f"Success: {param}"
```

The `ToolResultEvent` will have `error` set instead of `result`.

## Complex Parameters

Tools can have complex parameter types:

```python
from typing import Optional

def search(
    query: str,
    limit: int = 10,
    filters: Optional[dict] = None,
) -> str:
    """Search for items.

    Args:
        query: Search query string
        limit: Maximum results to return
        filters: Optional filters to apply
    """
    ...
```
