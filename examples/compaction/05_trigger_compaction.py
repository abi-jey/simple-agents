"""Trigger compaction example - manually requesting compaction during run.

This example shows how to use trigger_compaction() to request compaction
before the next generation cycle. Compaction can be triggered:

1. Within the event loop - during iteration over events (based on token usage)
2. Between run() calls - before starting a new conversation

The example uses "Pride and Prejudice" by Jane Austen from Project Gutenberg
to demonstrate realistic large context scenarios.
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
from nagents import Message
from nagents import Provider
from nagents import ProviderType
from nagents import SessionManager
from nagents import TextChunkEvent
from nagents import ToolResultEvent
from nagents import estimate_messages_tokens

load_dotenv()

console = Console()

# Token threshold for triggering compaction (adjust based on your model)
COMPACTION_TOKEN_THRESHOLD = 5000

# Book content cache (first chapter of Pride and Prejudice)
BOOK_CONTENT = """It is a truth universally acknowledged, that a single man in possession
of a good fortune must be in want of a wife.

However little known the feelings or views of such a man may be on his
first entering a neighbourhood, this truth is so well fixed in the minds
of the surrounding families, that he is considered as the rightful
property of some one or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard that
Netherfield Park is let at last?"

Mr. Bennet replied that he had not.

"But it is," returned she; "for Mrs. Long has just been here, and she
told me all about it."

Mr. Bennet made no answer.

"Do not you want to know who has taken it?" cried his wife, impatiently.

"You want to tell me, and I have no objection to hearing it."

This was invitation enough.

"Why, my dear, you must know, Mrs. Long says that Netherfield is taken
by a young man of large fortune from the north of England; that he came
down on Monday in a chaise and four to see the place, and was so much
delighted with it that he agreed with Mr. Morris immediately; that he is
to take possession before Michaelmas, and some of his servants are to be
in the house by the end of next week."

"What is his name?"

"Bingley."

"Is he married or single?"

"Oh, single, my dear, to be sure! A single man of large fortune; four or
five thousand a year. What a fine thing for our girls!"

"How so? how can it affect them?"

"My dear Mr. Bennet," replied his wife, "how can you be so tiresome? You
must know that I am thinking of his marrying one of them."

"Is that his design in settling here?"

"Design? Nonsense, how can you talk so! But it is very likely that he
may fall in love with one of them, and therefore you must visit him as
soon as he comes."

"I see no occasion for that. You and the girls may go--or you may send
them by themselves, which perhaps will be still better; for as you are
as handsome as any other of them, Mr. Bingley might like you the best of
the party."
"""


def read_book_chapter(chapter: int) -> str:
    """Read a chapter from the book.

    Simulates reading a book chapter and returning its content.
    Returns progressively longer content to simulate reading more.
    """
    # Each "chapter" adds more content
    content = BOOK_CONTENT
    if chapter > 1:
        # Add more content for subsequent chapters
        for i in range(2, chapter + 1):
            content += f"\n\n[Chapter {i}]\n" + BOOK_CONTENT[: 500 * i]
    return content


def summarize_text(text: str) -> str:
    """Summarize a piece of text.

    This simulates a tool that processes and summarizes text,
    returning a condensed version with analysis.
    """
    # Simulate processing time and return summary
    word_count = len(text.split())
    return (
        f"Text analysis complete.\n"
        f"- Word count: {word_count}\n"
        f"- Characters: {len(text)}\n"
        f"- Estimated reading time: {word_count // 200} minutes\n"
        f"- Key themes: Marriage, Social Class, Pride, Prejudice\n"
        f"- Main characters: Elizabeth Bennet, Mr. Darcy, Mr. Bingley, Jane Bennet"
    )


def search_text(query: str) -> str:
    """Search for text in the book.

    Returns matching passages and their context.
    """
    query_lower = query.lower()
    matches = []

    # Simple search simulation
    for i, line in enumerate(BOOK_CONTENT.split("\n")):
        if query_lower in line.lower():
            matches.append(f"Line {i}: {line.strip()}")

    if matches:
        return f"Found {len(matches)} matches for '{query}':\n" + "\n".join(matches[:5])
    return f"No matches found for '{query}'"


async def main() -> None:
    """Run the example demonstrating trigger_compaction()."""
    console.print(Panel.fit("[bold blue]Trigger Compaction Example[/bold blue]"))
    console.print("[dim]Using 'Pride and Prejudice' by Jane Austen for context[/dim]")
    console.print()

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

    # Create agent with compaction disabled by default
    # We'll trigger compaction manually when needed
    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        tools=[read_book_chapter, summarize_text, search_text],
        system_prompt=(
            "You are a literary analysis assistant. Help users analyze the book "
            "'Pride and Prejudice' by Jane Austen. Use the available tools to read "
            "chapters, search for text, and provide summaries. Keep responses concise."
        ),
        streaming=True,
        log_file=log_file,
        compactor="self",  # Use same agent for compaction
        compact_on=None,  # Disable automatic compaction - we'll trigger manually
    )

    try:
        await agent.initialize()
        console.print("[green]Agent initialized[/green]")
        console.print(f"[blue]HTTP logging to: {log_file}[/blue]")
        console.print(f"[dim]Compaction: Manual trigger (threshold: {COMPACTION_TOKEN_THRESHOLD} tokens)[/dim]")
        console.print()

        # Track if compaction was triggered during this session
        compaction_triggered = False

        # Sophisticated conversation demonstrating in-flow compaction
        conversations = [
            "Hello! I'd like to analyze Pride and Prejudice. Can you help me?",
            "Please read chapters 1 through 5 of the book.",
            "Now search for all mentions of 'marriage' in the text.",
            "Summarize what you've read so far about the main characters.",
            "What are the key themes emerging from these chapters?",
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
                elif isinstance(event, ToolResultEvent):
                    # Check token usage after each tool call
                    # Trigger compaction if we're approaching context limit
                    if (
                        event.usage.session
                        and event.usage.session.prompt_tokens > COMPACTION_TOKEN_THRESHOLD
                        and not compaction_triggered
                    ):
                        console.print()
                        console.print(
                            f"[yellow]⚠ Token usage ({event.usage.session.prompt_tokens}) exceeds threshold "
                            f"({COMPACTION_TOKEN_THRESHOLD}). Triggering compaction...[/yellow]"
                        )
                        agent.trigger_compaction()
                        compaction_triggered = True
                elif isinstance(event, CompactionStartedEvent):
                    console.print()
                    console.print(f"[yellow]⚠ Compaction started: {event.message_count} messages[/yellow]")
                elif isinstance(event, CompactionDoneEvent):
                    console.print()
                    console.print(
                        f"[green]✓ Compaction complete: {event.original_message_count} → "
                        f"{event.new_message_count} messages[/green]"
                    )
                    console.print(f"[dim]Summary: {event.summary_text[:150]}...[/dim]")
                    compaction_triggered = False  # Reset after compaction
                elif isinstance(event, DoneEvent):
                    console.print()
                    if event.usage.session:
                        console.print(
                            f"[dim]Tokens: {event.usage.total_tokens} "
                            f"(session: {event.usage.session.prompt_tokens})[/dim]"
                        )
                    else:
                        console.print(f"[dim]Tokens: {event.usage.total_tokens}[/dim]")

            console.print()

        # Show final session stats
        history = await session_manager.get_history(session_id)
        console.print(Panel(f"[bold]Final session: {len(history)} messages[/bold]", border_style="blue"))

        # Estimate final context size
        messages = [Message(role="system", content=agent.system_prompt or ""), *history]
        final_tokens = estimate_messages_tokens(messages)
        console.print(f"[dim]Estimated context: ~{final_tokens} tokens[/dim]")

    finally:
        await agent.close()

    console.print("\n[green]Done![/green]")
    console.print("[dim]Compaction was triggered when token usage exceeded the threshold.[/dim]")


if __name__ == "__main__":
    logging.basicConfig(level="INFO", handlers=[RichHandler(console=console)])
    asyncio.run(main())
