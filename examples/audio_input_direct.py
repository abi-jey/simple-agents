"""
Example demonstrating audio input directly to OpenAI GPT-4o (no STT transcription).

GPT-4o and similar models can natively process audio input without transcription.
This example shows how to send audio directly to the LLM.

Requirements:
------------
OPENAI_API_KEY=your-openai-api-key
(Or Azure equivalent for Azure OpenAI)

Supported Audio Formats for OpenAI:
----------------------------------
- WAV (recommended for best quality)
- MP3

How to Get Audio with Audacity:
------------------------------
1. Open Audacity and record or import audio
2. Export as WAV: File -> Export Audio -> WAV (16-bit PCM)
3. Save to examples/test_audio.wav
4. Run this example

Note: OGG/OGA format (Telegram voice) is NOT supported by OpenAI directly.
      Use audio_input_stt.py for those formats (transcribes via Gemini STT).
"""

import asyncio
import base64
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
from nagents import AudioContent
from nagents import ContentPart
from nagents import DoneEvent
from nagents import ErrorEvent
from nagents import Provider
from nagents import ProviderType
from nagents import SessionManager
from nagents import TextChunkEvent
from nagents import TextContent

load_dotenv()

logger = getLogger(__name__)
console = Console()


def get_openai_provider() -> Provider:
    """Create a Provider configured for OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    return Provider(
        provider_type=ProviderType.OPENAI_COMPATIBLE,
        api_key=api_key,
        model="gpt-4o-audio-preview",
    )


def get_azure_provider() -> Provider:
    """Create a Provider configured for Azure OpenAI V1 API."""
    endpoint = os.getenv("AZURE_DEPLOYED_MODEL_ENDPOINT")
    api_key = os.getenv("AZURE_DEPLOYED_MODEL_KEY")
    model_name = os.getenv("AZURE_DEPLOYED_MODEL_NAME")

    if not endpoint or not api_key or not model_name:
        raise ValueError("Azure environment variables not set")

    base_url = f"{endpoint.rstrip('/')}/openai"

    return Provider(
        provider_type=ProviderType.AZURE_OPENAI_COMPATIBLE_V1,
        api_key=api_key,
        model=model_name,
        base_url=base_url,
    )


def find_audio_file() -> Path | None:
    """Find an audio file in the examples directory."""
    examples_dir = Path("examples")
    for ext in [".wav", ".mp3"]:
        path = examples_dir / f"test_audio{ext}"
        if path.exists():
            return path
    return None


async def main() -> None:
    """Main example demonstrating direct audio input to LLM."""
    console.print(Panel.fit("[bold blue]nagents Direct Audio Input Example[/bold blue]"))
    console.print()
    console.print("[dim]This example sends audio directly to GPT-4o without transcription.[/dim]")
    console.print("[dim]The LLM processes the audio natively.[/dim]")
    console.print()

    openai_key = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_DEPLOYED_MODEL_ENDPOINT")

    if openai_key:
        try:
            provider = get_openai_provider()
            console.print("[green]Using OpenAI with gpt-4o-audio-preview[/green]")
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            return
    elif azure_endpoint:
        try:
            provider = get_azure_provider()
            console.print("[green]Using Azure OpenAI[/green]")
            console.print("[yellow]Note: Ensure your deployment supports audio (e.g., gpt-4o)[/yellow]")
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            return
    else:
        console.print("[bold red]Error:[/bold red] No OpenAI or Azure configuration found")
        console.print("\n[yellow]Set one of these in your .env:[/yellow]")
        console.print("  OPENAI_API_KEY=your-key")
        console.print("  # OR")
        console.print("  AZURE_DEPLOYED_MODEL_ENDPOINT=https://...")
        console.print("  AZURE_DEPLOYED_MODEL_KEY=your-key")
        console.print("  AZURE_DEPLOYED_MODEL_NAME=gpt-4o")
        return

    audio_file_path = find_audio_file()

    if not audio_file_path:
        console.print("\n[bold red]No audio file found![/bold red]")
        console.print("\n[yellow]Create audio with Audacity:[/yellow]")
        console.print("  1. Open Audacity and record or import audio")
        console.print("  2. Export as WAV: File -> Export Audio -> WAV (16-bit PCM)")
        console.print("  3. Save to examples/test_audio.wav")
        console.print("\n[dim]Supported formats: WAV, MP3[/dim]")
        return

    console.print(f"[dim]Using audio file: {audio_file_path}[/dim]")
    audio_data = audio_file_path.read_bytes()
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    format = audio_file_path.suffix.lower().lstrip(".")
    if format not in ("wav", "mp3"):
        console.print(f"[red]Unsupported format: {format}. Use WAV or MP3.[/red]")
        return

    session_manager = SessionManager(Path("sessions_audio_direct.db"))
    session_id = f"audio-direct-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    log_file = Path("logs") / f"{session_id}.txt"

    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        streaming=True,
        log_file=log_file,
    )

    user_content: list[ContentPart] = [
        TextContent(text="Listen to this audio and describe what you hear."),
        AudioContent(base64_data=audio_base64, format=format),
    ]

    try:
        await agent.initialize()
        console.print(f"[green]Agent initialized, model: {provider.model}[/green]")

        console.print(Panel("[bold]User sends audio with question...[/bold]", border_style="green"))
        console.print()

        response_text = ""

        async for event in agent.run(
            user_message=user_content,
            session_id=session_id,
        ):
            if isinstance(event, TextChunkEvent):
                console.print(event.chunk, end="")
                response_text += event.chunk
            elif isinstance(event, ErrorEvent):
                console.print(f"\n[bold red]Error:[/bold red] {event.message}")
                if event.code:
                    console.print(f"[dim red]Code: {event.code}[/dim red]")
            elif isinstance(event, DoneEvent):
                console.print()
                usage_text = Text()
                usage_text.append("Usage: ", style="bold blue")
                usage_text.append(
                    f"prompt={event.usage.prompt_tokens}, "
                    f"completion={event.usage.completion_tokens}, "
                    f"total={event.usage.total_tokens}",
                    style="dim",
                )
                console.print(usage_text)

        console.print(f"\n[dim]HTTP traffic logged to: {log_file.absolute()}[/dim]")

    finally:
        await agent.close()

    console.print("\n[dim]Direct audio input example complete.[/dim]")


if __name__ == "__main__":
    basicConfig(
        level="INFO",
        format="[%(name)s] %(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    asyncio.run(main())
