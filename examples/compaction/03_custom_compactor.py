"""Custom compactor example - using a separate agent for summarization.

This example shows how to use a custom Compactor with a different
model/provider for context summarization.

Custom compactor benefits:
- Use a cheaper/faster model for summarization
- Different context limits for compaction
- Separate provider configuration
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
from nagents import Compactor
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
    """Run the example with custom compactor."""
    console.print(Panel.fit("[bold blue]Custom Compactor Example[/bold blue]"))

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set.[/red]")
        return

    # Main agent uses gpt-4o-mini (cheap and fast)
    main_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    console.print(f"[dim]Main model: {main_model}[/dim]")

    main_provider = Provider(
        provider_type=ProviderType.OPENAI_COMPATIBLE,
        api_key=api_key,
        model=main_model,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    # Create a custom compactor with same model (for cost efficiency)
    # In production, you might use a different/cheaper model
    compactor = Compactor(
        model="gpt-4o-mini",  # Same model for cost efficiency
        compact_on=Messages(length=5),  # Compact after 5 messages
        system_prompt="""You are an expert at summarizing conversations.
Create a concise summary that preserves:
1. Key decisions and their rationale
2. Action items and next steps
3. Important context for continuing the conversation

Be thorough but brief. Use bullet points for clarity.""",
    )

    console.print("[dim]Compactor model: gpt-4o-mini[/dim]")

    session_manager = SessionManager(Path("sessions/custom_compactor.db"))
    session_id = f"custom-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    log_file = Path("logs") / f"{session_id}.txt"

    agent = Agent(
        provider=main_provider,
        session_manager=session_manager,
        tools=[read_large_file],
        system_prompt="You are a helpful assistant. Keep responses concise.",
        streaming=True,
        log_file=log_file,
        compactor=compactor,  # Custom compactor
    )

    try:
        await agent.initialize()
        console.print("[green]Agent initialized[/green]")
        console.print(f"[blue]HTTP logging to: {log_file}[/blue]")
        console.print("[dim]Compaction: CUSTOM (gpt-4o-mini)[/dim]")
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
                    console.print(f"[yellow]⚠ Custom compaction started: {event.message_count} messages[/yellow]")
                elif isinstance(event, CompactionDoneEvent):
                    console.print()
                    console.print(f"[green]✓ Compaction by: {event.compactor_used}[/green]")
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
