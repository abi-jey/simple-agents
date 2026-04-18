"""
Example demonstrating OpenRouter with streaming and reasoning support.

This example shows how to use the Agent class with ProviderType.OPENROUTER,
which provides access to many LLM providers through a unified API with
optional reasoning/thinking support.
"""

import asyncio
import os
from datetime import UTC
from datetime import datetime
from logging import basicConfig
from logging import getLogger
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

from nagents import Agent
from nagents import DoneEvent
from nagents import ErrorEvent
from nagents import GenerationConfig
from nagents import Provider
from nagents import ProviderType
from nagents import SessionManager
from nagents import TextChunkEvent
from nagents import TextDoneEvent
from nagents import ToolCallEvent
from nagents import ToolResultEvent
from nagents.events.types import ReasoningChunkEvent

load_dotenv()

logger = getLogger(__name__)
console = Console()


async def order_coffee(quantity: int, store: str) -> str:
    """Simulate ordering a coffee."""
    await asyncio.sleep(1)
    return f"Ordered {quantity} coffee(s) from {store}."


def get_time(tz: str = "UTC") -> str:
    """Get the current time in a specific timezone."""
    now = datetime.now(UTC)
    return f"Current time in {tz}: {now.strftime('%Y-%m-%d %H:%M:%S')}"


async def main() -> None:
    """Main example demonstrating OpenRouter with streaming and reasoning."""
    console.print(Panel.fit("[bold blue]nagents OpenRouter Streaming Example[/bold blue]"))

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set. Set it in .env or environment.[/red]")
        return

    provider = Provider(
        provider_type=ProviderType.OPENROUTER,
        api_key=api_key,
        model="anthropic/claude-opus-4.7",
    )

    session_manager = SessionManager(Path("sessions.db"))
    session_id = f"session-openrouter-stream-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    log_file = Path("logs") / f"{session_id}.txt"

    config = GenerationConfig(
        reasoning={"enabled": True},
    )

    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        tools=[order_coffee, get_time],
        system_prompt="You are a helpful assistant with access to coffee ordering and time tools. "
        "Use them when appropriate to answer user questions.",
        streaming=True,
        log_file=log_file,
    )

    try:
        await agent.initialize()
        console.print(f"[green]Agent initialized, model: {provider.model}[/green]")
        console.print(f"[blue]HTTP logging to: {log_file}[/blue]")

        query = "Hey, what time is it, can you order 2 coffees for me from local shop?"

        console.print(Panel(f"[bold]User:[/bold] {query}", border_style="green"))
        console.print()
        response_text = ""

        async for event in agent.run(
            user_message=query,
            session_id=session_id,
            user_id="example-user",
            config=config,
        ):
            if isinstance(event, ReasoningChunkEvent):
                console.print(f"[dim italic]{event.chunk}[/dim italic]", end="")
            elif isinstance(event, TextChunkEvent):
                console.print(event.chunk, end="")
                response_text += event.chunk
            elif isinstance(event, TextDoneEvent):
                if not response_text:
                    response_text = event.text
                    console.print(event.text)
            elif isinstance(event, ToolCallEvent):
                console.print()
                tool_text = Text()
                tool_text.append("Tool Call: ", style="bold yellow")
                tool_text.append(event.name, style="cyan")
                tool_text.append("(", style="dim")
                args_str = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
                tool_text.append(args_str, style="white")
                tool_text.append(")", style="dim")
                console.print(tool_text)
            elif isinstance(event, ToolResultEvent):
                result_text = Text()
                result_text.append("Tool call result: ", style="bold green")
                result_text.append(f"`{event.name}` ", style="cyan")
                if event.error:
                    result_text.append(f"ERROR: {event.error}", style="red")
                else:
                    result_text.append(str(event.result), style="white")
                result_text.append(f" ({event.duration_ms:.1f}ms)", style="dim")
                console.print(result_text)
            elif isinstance(event, ErrorEvent):
                console.print()
                error_text = Text()
                error_text.append("ERROR: ", style="bold red")
                error_text.append(event.message, style="red")
                if event.code:
                    error_text.append(f" (code: {event.code})", style="dim red")
                if event.recoverable:
                    error_text.append(" [recoverable]", style="yellow")
                console.print(error_text)
            elif isinstance(event, DoneEvent):
                console.print()
                done_text = Text()
                done_text.append("Generation complete", style="bold green")
                done_text.append(" - ", style="dim")
                done_text.append(f"{len(event.final_text)} chars", style="dim")
                if event.session_id:
                    done_text.append(f" (session: {event.session_id})", style="dim blue")
                console.print(done_text)
                usage_text = Text()
                usage_text.append("Usage: ", style="bold blue")
                usage_text.append(
                    f"prompt={event.usage.prompt_tokens}, "
                    f"completion={event.usage.completion_tokens}, "
                    f"total={event.usage.total_tokens}",
                    style="dim",
                )
                if event.usage.session:
                    usage_text.append(
                        f" | Session total: {event.usage.session.total_tokens}",
                        style="dim cyan",
                    )
                console.print(usage_text)
        console.print()

        console.print(f"[dim]HTTP traffic logged to: {log_file.absolute()}[/dim]")
    finally:
        await agent.close()
    console.print("\n[dim]Example complete.[/dim]")


if __name__ == "__main__":
    basicConfig(
        level="INFO",
        format="[%(name)s] %(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    asyncio.run(main())
