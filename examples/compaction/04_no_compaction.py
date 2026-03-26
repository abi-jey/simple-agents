"""No compaction example - explicitly disabling context compaction.

This example shows how to completely disable context compaction
by setting compactor=None.

Use this when:
- You want full control over context management
- Working with models that have very large context windows
- Testing/debugging without compaction interference
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
    """Run the example with compaction disabled."""
    console.print(Panel.fit("[bold blue]No Compaction Example[/bold blue]"))

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

    session_manager = SessionManager(Path("sessions/no_compaction.db"))
    session_id = f"no-compact-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    log_file = Path("logs") / f"{session_id}.txt"

    # Explicitly disable compaction with compactor=None
    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        tools=[read_large_file],
        system_prompt="You are a helpful assistant. Keep responses concise.",
        streaming=True,
        log_file=log_file,
        compactor=None,  # Explicitly disable compaction
    )

    try:
        await agent.initialize()
        console.print("[green]Agent initialized[/green]")
        console.print(f"[blue]HTTP logging to: {log_file}[/blue]")
        console.print("[red]Compaction: DISABLED[/red]")
        console.print("[dim]Context will grow without limit[/dim]")

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
                elif isinstance(event, DoneEvent):
                    console.print()
                    console.print(f"[dim]Tokens: {event.usage.total_tokens}[/dim]")

            console.print()

        history = await session_manager.get_history(session_id)
        console.print(
            Panel(
                f"[bold]Final history: {len(history)} messages[/bold]\n"
                f"[dim]No compaction occurred - full history preserved[/dim]",
                border_style="blue",
            )
        )

    finally:
        await agent.close()

    console.print("\n[green]Done![/green]")
    console.print("[yellow]Note: Without compaction, you may hit context limits with long conversations.[/yellow]")


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", handlers=[RichHandler(console=console)])
    asyncio.run(main())
