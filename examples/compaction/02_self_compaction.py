"""Self compaction example - using the main agent for summarization.

This example shows how to use compactor="self" to have the main agent
summarize its own conversation history when context limits are reached.

Self compaction:
- Uses the same provider/model as the main agent
- Inserts a developer message with the compaction prompt
- The agent summarizes its own history
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

console = Console()


def read_large_file(filename: str) -> str:
    """Read a file and return its contents."""
    import random
    import string

    lines = []
    for i in range(50):
        line = "".join(random.choices(string.ascii_letters + string.digits, k=100))
        lines.append(f"Line {i}: {line}")
    return f"Contents of {filename}:\n" + "\n".join(lines)


async def main() -> None:
    """Run the example with self compaction."""
    console.print(Panel.fit("[bold blue]Self Compaction Example[/bold blue]"))

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set.[/red]")
        return

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    console.print(f"[dim]Using model: {model}[/dim]")

    provider = Provider(
        provider_type=ProviderType.OPENAI_COMPATIBLE,
        api_key=api_key,
        model=model,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    session_manager = SessionManager(Path("sessions/self_compaction.db"))
    session_id = f"self-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    log_file = Path("logs") / f"{session_id}.txt"

    # Use compactor="self" for self compaction
    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        tools=[read_large_file],
        system_prompt="You are a helpful assistant. Keep responses concise.",
        streaming=True,
        log_file=log_file,
        compactor="self",  # Use main agent for compaction
        compact_on=Messages(length=5),  # Compact after 5 messages
        compact_prompt="Summarize the conversation, highlighting key decisions and action items. Be concise.",
    )

    try:
        await agent.initialize()
        console.print("[green]Agent initialized[/green]")
        console.print(f"[blue]HTTP logging to: {log_file}[/blue]")
        console.print("[dim]Compaction: SELF (same agent)[/dim]")
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
                    console.print(f"[yellow]⚠ Self-compaction started: {event.message_count} messages[/yellow]")
                elif isinstance(event, CompactionDoneEvent):
                    console.print()
                    console.print(f"[green]✓ Self-compaction done by: {event.compactor_used}[/green]")
                    console.print(f"[dim]Summary ({event.summary_tokens} tokens): {event.summary_text[:150]}...[/dim]")
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
