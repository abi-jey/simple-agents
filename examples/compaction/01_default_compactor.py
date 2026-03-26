"""Default compactor example - using the built-in DEFAULT_COMPACTOR.

This example shows how the agent automatically uses context compaction
with sensible defaults when no compactor is specified.

The DEFAULT_COMPACTOR:
- Uses the main agent's provider/model for summarization
- Uses provider-specific context limits (80% of context window)
- Truncates large tool results automatically
"""

import asyncio
import logging
import os
from datetime import UTC
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from nagents import Agent
from nagents import CompactionDoneEvent
from nagents import CompactionStartedEvent
from nagents import DoneEvent
from nagents import Messages
from nagents import Provider
from nagents import ProviderType
from nagents import SessionManager
from nagents import TextChunkEvent

load_dotenv()

logger_name = __name__
console = Console()


def read_large_file(filename: str) -> str:
    """Read a file and return its contents.

    This simulates a tool that returns large output.
    """
    import random
    import string

    lines = []
    for i in range(50):
        line = "".join(random.choices(string.ascii_letters + string.digits, k=100))
        lines.append(f"Line {i}: {line}")
    return f"Contents of {filename}:\n" + "\n".join(lines)


async def main() -> None:
    """Run the example with default compactor."""
    console.print(Panel.fit("[bold blue]Default Compactor Example[/bold blue]"))

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set.[/red]")
        console.print("Set it in .env file or with: export OPENAI_API_KEY=your-key-here")
        return

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    console.print(f"[dim]Using model: {model}[/dim]")

    provider = Provider(
        provider_type=ProviderType.OPENAI_COMPATIBLE,
        api_key=api_key,
        model=model,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    session_manager = SessionManager(Path("sessions/default_compactor.db"))
    session_id = f"default-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    log_file = Path("logs") / f"{session_id}.txt"

    # No compactor specified - uses DEFAULT_COMPACTOR automatically
    # Use Messages trigger with low threshold to demonstrate compaction
    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        tools=[read_large_file],
        system_prompt="You are a helpful assistant. Keep responses concise.",
        streaming=True,
        log_file=log_file,
        compact_on=Messages(length=5),  # Compact after 5 messages
    )

    try:
        await agent.initialize()
        console.print("[green]Agent initialized[/green]")
        console.print(f"[blue]HTTP logging to: {log_file}[/blue]")
        console.print("[dim]Compaction: DEFAULT_COMPACTOR (auto)[/dim]")
        console.print("[dim]Trigger: Messages(length=5)[/dim]")

        conversations = [
            "Hello! What can you help me with?",
            "Please read the file 'data_file_1.txt'",
            "Now read 'data_file_2.txt'",
            "What files did I ask you to read?",
        ]

        for i, query in enumerate(conversations, 1):
            console.print(Panel(f"[bold]User ({i}):[/bold] {query}", border_style="green"))

            response_text = ""
            async for event in agent.run(
                user_message=query,
                session_id=session_id,
                user_id="compaction-user",
            ):
                if isinstance(event, TextChunkEvent):
                    console.print(event.chunk, end="")
                    response_text += event.chunk
                elif isinstance(event, CompactionStartedEvent):
                    console.print()
                    console.print(
                        f"[yellow]⚠ Compaction started: {event.message_count} messages, ~{event.estimated_tokens} tokens[/yellow]"
                    )
                elif isinstance(event, CompactionDoneEvent):
                    console.print()
                    console.print(
                        f"[green]✓ Compaction done: {event.original_message_count} → {event.new_message_count} messages[/green]"
                    )
                    console.print(f"[dim]Compactor: {event.compactor_used}[/dim]")
                    console.print(f"[dim]Summary: {event.summary_text[:150]}...[/dim]")
                elif isinstance(event, DoneEvent):
                    console.print()
                    console.print(f"[dim]Tokens: {event.usage.total_tokens}[/dim]")

            console.print()

        history = await session_manager.get_history(session_id)
        console.print(Panel(f"[bold]Final history: {len(history)} messages[/bold]", border_style="blue"))

    finally:
        await agent.close()

    console.print("\n[green]Done![/green]")


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", handlers=[RichHandler(console=console)])
    asyncio.run(main())
