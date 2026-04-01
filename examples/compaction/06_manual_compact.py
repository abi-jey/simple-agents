"""Manual compact example - calling compact() outside of run().

This example shows how to use agent.compact() to manually trigger
compaction for a session outside of the run loop. This is useful for:

1. Pre-emptively compacting before starting a new conversation
2. Compacting after a long conversation before continuing
3. Managing context across multiple sessions

The compact() method returns a CompactionDoneEvent with full details
about the compaction result.
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
from nagents import DoneEvent
from nagents import Provider
from nagents import ProviderType
from nagents import SessionManager
from nagents import TextChunkEvent

load_dotenv()

console = Console()


def generate_large_data(item_count: int) -> str:
    """Generate a large amount of data.

    This simulates a tool that generates large output.
    """
    import random
    import string

    lines = []
    for i in range(item_count):
        line = "".join(random.choices(string.ascii_letters + string.digits, k=80))
        lines.append(f"Item {i}: {line}")
    return "\n".join(lines)


async def main() -> None:
    """Run the example demonstrating manual compact()."""
    console.print(Panel.fit("[bold blue]Manual Compact Example[/bold blue]"))

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

    session_manager = SessionManager(Path("sessions/manual_compact.db"))
    session_id = f"manual-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    log_file = Path("logs") / f"{session_id}.txt"

    # Create agent - compaction will be done manually
    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        tools=[generate_large_data],
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
        console.print("[dim]Compaction: Manual only (compact_on=None)[/dim]")
        console.print()

        # Have a conversation to build up context
        conversations = [
            "Hello! Help me generate some data.",
            "Generate 30 items of data.",
            "Now generate 40 more items.",
        ]

        for i, query in enumerate(conversations, 1):
            console.print(Panel(f"[bold]User ({i}):[/bold] {query}", border_style="green"))

            async for event in agent.run(
                user_message=query,
                session_id=session_id,
                user_id="manual-user",
            ):
                if isinstance(event, TextChunkEvent):
                    console.print(event.chunk, end="")
                elif isinstance(event, DoneEvent):
                    console.print()
                    console.print(f"[dim]Tokens: {event.usage.total_tokens}[/dim]")

            console.print()

        # Check history size before compaction
        history_before = await session_manager.get_history(session_id)
        console.print(Panel(f"[bold]Before compaction: {len(history_before)} messages[/bold]", border_style="yellow"))

        # Manually compact the session
        console.print(Panel("[bold yellow]Calling agent.compact()[/bold yellow]", border_style="yellow"))

        result = await agent.compact(session_id)

        console.print(f"[green]✓ Compaction complete![/green]")
        console.print(f"[dim]Original messages: {result.original_message_count}[/dim]")
        console.print(f"[dim]New messages: {result.new_message_count}[/dim]")
        console.print(f"[dim]Original tokens: {result.original_token_count}[/dim]")
        console.print(f"[dim]Summary tokens: {result.summary_tokens}[/dim]")
        console.print(f"[dim]Compactor used: {result.compactor_used}[/dim]")
        console.print()
        console.print(Panel(f"[bold]Summary (first 300 chars):[/bold]\n{result.summary_text[:300]}...", border_style="blue"))

        # Check history size after compaction
        history_after = await session_manager.get_history(session_id)
        console.print(Panel(f"[bold]After compaction: {len(history_after)} messages[/bold]", border_style="green"))

        # Continue conversation - using compacted context
        console.print(Panel("[bold]User (4):[/bold] What data did we generate?", border_style="green"))

        async for event in agent.run(
            user_message="What data did we generate?",
            session_id=session_id,
            user_id="manual-user",
        ):
            if isinstance(event, TextChunkEvent):
                console.print(event.chunk, end="")
            elif isinstance(event, DoneEvent):
                console.print()
                console.print(f"[dim]Tokens: {event.usage.total_tokens}[/dim]")

        console.print()

        console.print("\n[green]Done![/green]")
        console.print("[dim]Note: The agent can still answer questions about the conversation using the summary.[/dim]")

    finally:
        await agent.close()


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", handlers=[RichHandler(console=console)))
    asyncio.run(main())