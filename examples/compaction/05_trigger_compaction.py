"""Trigger compaction example - manually requesting compaction during run.

This example shows how to use trigger_compaction() to request compaction
before the next generation cycle. This is useful when you know a tool
has generated a lot of context and want to compact before continuing.

Use case: A tool that returns large results and you want to compact
before the next model call to save context.

Note: Since tools don't currently have access to the agent, this example
shows how you might structure your code to call trigger_compaction()
between rounds of interaction.
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
from nagents import Provider
from nagents import ProviderType
from nagents import SessionManager
from nagents import TextChunkEvent

load_dotenv()

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
    """Run the example demonstrating trigger_compaction()."""
    console.print(Panel.fit("[bold blue]Trigger Compaction Example[/bold blue]"))

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

    session_manager = SessionManager(Path("sessions/trigger_compaction.db"))
    session_id = f"trigger-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    log_file = Path("logs") / f"{session_id}.txt"

    # Create agent with compaction disabled (compact_on=None)
    # We'll trigger compaction manually
    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        tools=[read_large_file],
        system_prompt="You are a helpful assistant. Keep responses concise.",
        streaming=True,
        log_file=log_file,
        compactor="self",  # Use self for compaction
        compact_on=None,  # Disable automatic compaction
    )

    try:
        await agent.initialize()
        console.print("[green]Agent initialized[/green]")
        console.print(f"[blue]HTTP logging to: {log_file}[/blue]")
        console.print("[dim]Compaction: Manual trigger only[/dim]")

        conversations = [
            "Hello! What can you help me with?",
            "Please read the file 'large_data_file_1.txt'",
            "Now read 'large_data_file_2.txt'",
        ]

        for i, query in enumerate(conversations, 1):
            console.print(Panel(f"[bold]User ({i}):[/bold] {query}", border_style="green"))

            async for event in agent.run(
                user_message=query,
                session_id=session_id,
                user_id="trigger-user",
            ):
                if isinstance(event, TextChunkEvent):
                    console.print(event.chunk, end="")
                elif isinstance(event, DoneEvent):
                    console.print()
                    console.print(f"[dim]Tokens: {event.usage.total_tokens}[/dim]")

            console.print()

        # Now manually trigger compaction before next interaction
        # This is useful when you know context has grown large
        console.print(Panel("[bold yellow]Manually Triggering Compaction[/bold yellow]", border_style="yellow"))

        agent.trigger_compaction()
        console.print("[dim]Compaction requested. Will occur before next generation.[/dim]")

        # Continue with more conversation - compaction happens automatically
        # before the next model call
        console.print(Panel("[bold]User (4):[/bold] What files did you read?", border_style="green"))

        async for event in agent.run(
            user_message="What files did you read?",
            session_id=session_id,
            user_id="trigger-user",
        ):
            if isinstance(event, TextChunkEvent):
                console.print(event.chunk, end="")
            elif isinstance(event, CompactionStartedEvent):
                console.print()
                console.print(f"[yellow]⚠ Compaction triggered: {event.message_count} messages[/yellow]")
            elif isinstance(event, CompactionDoneEvent):
                console.print()
                console.print(
                    f"[green]✓ Compaction complete: {event.original_message_count} → {event.new_message_count} messages[/green]"
                )
                console.print(f"[dim]Summary tokens: {event.summary_tokens}[/dim]")
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
