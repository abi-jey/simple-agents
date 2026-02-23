"""
Example demonstrating audio input with STT (Speech-to-Text) transcription.

This example shows how to use the Agent with audio content that gets
transcribed using Gemini STT before being sent to the LLM.

Requirements:
------------
1. GEMINI_API_KEY - for STT transcription
2. AZURE_DEPLOYED_MODEL_* - for the LLM (or modify to use OpenAI)

Audio Formats Supported:
----------------------
- wav, mp3, ogg, oga (Telegram voice), flac, m4a, webm, aac

Usage:
-----
1. Set environment variables in .env:
   GEMINI_API_KEY=your-gemini-api-key
   AZURE_DEPLOYED_MODEL_ENDPOINT=https://your-resource.cognitiveservices.azure.com
   AZURE_DEPLOYED_MODEL_KEY=your-api-key
   AZURE_DEPLOYED_MODEL_NAME=your-deployment-name

2. Place an audio file in the examples directory or provide a path
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
from nagents import GeminiSTTService
from nagents import Provider
from nagents import ProviderType
from nagents import SessionManager
from nagents import TextChunkEvent
from nagents import TextContent
from nagents import TextDoneEvent

load_dotenv()

logger = getLogger(__name__)
console = Console()


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


def get_gemini_stt() -> GeminiSTTService:
    """Create a Gemini STT service for audio transcription."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return GeminiSTTService(api_key=api_key)


async def main() -> None:
    """Main example demonstrating audio input with STT."""
    console.print(Panel.fit("[bold blue]nagents Audio Input with STT Example[/bold blue]"))

    # Check for required env vars
    gemini_key = os.getenv("GEMINI_API_KEY")
    azure_endpoint = os.getenv("AZURE_DEPLOYED_MODEL_ENDPOINT")

    if not gemini_key:
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY not set")
        console.print("\n[yellow]Please set in .env:[/yellow]")
        console.print("  GEMINI_API_KEY=your-gemini-api-key")
        return

    if not azure_endpoint:
        console.print("[yellow]Warning:[/yellow] Azure not configured, using OpenAI")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            console.print("[bold red]Error:[/bold red] Neither Azure nor OpenAI configured")
            return
        provider = Provider(
            provider_type=ProviderType.OPENAI_COMPATIBLE,
            api_key=openai_key,
            model="gpt-4o-mini",
        )
    else:
        try:
            provider = get_azure_provider()
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            return

    # Create STT service
    stt_service = get_gemini_stt()
    console.print("[green]STT service initialized (Gemini)[/green]")

    # Create session manager
    session_manager = SessionManager(Path("sessions_audio.db"))

    # Create log file
    session_id = f"audio-session-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    log_file = Path("logs") / f"{session_id}.txt"

    # Create agent with STT service
    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        stt_service=stt_service,
        streaming=True,
        log_file=log_file,
    )

    # Demo audio file - you need to provide one
    # Telegram voice messages are typically .oga or .ogg format
    audio_file_path = Path("examples/test_audio.ogg")

    if audio_file_path.exists():
        console.print(f"[dim]Using audio file: {audio_file_path}[/dim]")
        audio_data = audio_file_path.read_bytes()
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        # Determine format from extension
        format = audio_file_path.suffix.lower().lstrip(".")
        if format == "oga":
            format = "ogg"

        user_content: list[ContentPart] = [
            AudioContent(base64_data=audio_base64, format=format),
        ]
    else:
        console.print("[yellow]No audio file found, using simulated example[/yellow]")
        console.print("[dim]To test with real audio, place a file at examples/test_audio.ogg[/dim]")

        user_content = [
            TextContent(text="Hello, can you tell me a short joke?"),
        ]

    try:
        await agent.initialize()
        console.print(f"[green]Agent initialized, model: {provider.model}[/green]")

        console.print(Panel("[bold]User sends audio message...[/bold]", border_style="green"))
        console.print()

        response_text = ""

        async for event in agent.run(
            user_message=user_content,
            session_id=session_id,
        ):
            if isinstance(event, TextChunkEvent):
                console.print(event.chunk, end="")
                response_text += event.chunk
            elif isinstance(event, TextDoneEvent):
                if not response_text:
                    response_text = event.text
                    console.print(event.text)
            elif isinstance(event, ErrorEvent):
                console.print(f"\n[bold red]Error:[/bold red] {event.message}")
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
        await stt_service.close()

    console.print("\n[dim]Audio input example complete.[/dim]")


if __name__ == "__main__":
    basicConfig(
        level="INFO",
        format="[%(name)s] %(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    asyncio.run(main())
