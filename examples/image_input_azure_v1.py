"""
Example demonstrating image input with Azure OpenAI V1 API.

This example shows how to use the Agent with multimodal content including
images provided as base64-encoded data from local files.

Requirements:
------------
AZURE_DEPLOYED_MODEL_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_DEPLOYED_MODEL_KEY=your-api-key
AZURE_DEPLOYED_MODEL_NAME=your-deployment-name (must support vision, e.g., gpt-4o)

How to Get Test Images:
----------------------
1. Any image from your computer (JPG, PNG, GIF, WebP)
2. Place it at examples/test_image.jpg (or .png, .gif, .webp)
3. Run this example

Supported Image Formats:
-----------------------
- JPEG, PNG, GIF, WebP
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
from nagents import ContentPart
from nagents import DoneEvent
from nagents import ErrorEvent
from nagents import ImageContent
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


def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    """
    Encode an image file to base64.

    Returns:
        Tuple of (base64_data, media_type)
    """
    image_data = image_path.read_bytes()
    base64_data = base64.b64encode(image_data).decode("utf-8")

    ext = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/jpeg")

    return base64_data, media_type


def find_image_file() -> Path | None:
    """Find an image file in the examples directory."""
    examples_dir = Path("examples")
    for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
        path = examples_dir / f"test_image{ext}"
        if path.exists():
            return path
    return None


async def main() -> None:
    """Main example demonstrating image input with Azure OpenAI."""
    console.print(Panel.fit("[bold blue]nagents Image Input Example (Azure V1)[/bold blue]"))

    try:
        provider = get_azure_provider()
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        console.print("\n[yellow]Please ensure your .env file contains:[/yellow]")
        console.print("  AZURE_DEPLOYED_MODEL_ENDPOINT=https://your-resource.cognitiveservices.azure.com")
        console.print("  AZURE_DEPLOYED_MODEL_KEY=your-api-key")
        console.print("  AZURE_DEPLOYED_MODEL_NAME=your-deployment-name (must support vision)")
        return

    console.print(f"[dim]Azure V1 endpoint: {provider.base_url}[/dim]")
    console.print(f"[dim]Model: {provider.model}[/dim]")
    console.print("[yellow]Note: Model must support vision (e.g., gpt-4o)[/yellow]")

    # Find image file
    image_path = find_image_file()

    if not image_path:
        console.print("\n[bold red]No image file found![/bold red]")
        console.print("\n[yellow]Place an image file at one of these locations:[/yellow]")
        console.print("  examples/test_image.jpg")
        console.print("  examples/test_image.png")
        console.print("  examples/test_image.gif")
        console.print("  examples/test_image.webp")
        console.print("\n[dim]Or create a test image with:[/dim]")
        console.print("  poetry run python examples/create_test_files.py")
        return

    console.print(f"[dim]Using image: {image_path}[/dim]")
    base64_data, media_type = encode_image_to_base64(image_path)

    session_manager = SessionManager(Path("sessions_image.db"))
    session_id = f"image-session-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    log_file = Path("logs") / f"{session_id}.txt"

    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        streaming=True,
        log_file=log_file,
    )

    user_content: list[ContentPart] = [
        TextContent(text="What do you see in this image?"),
        ImageContent(base64_data=base64_data, media_type=media_type),
    ]

    try:
        await agent.initialize()
        console.print(f"[green]Agent initialized, model: {provider.model}[/green]")

        console.print(Panel("[bold]User sends image with question...[/bold]", border_style="green"))
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

    console.print("\n[dim]Image input example complete.[/dim]")


if __name__ == "__main__":
    basicConfig(
        level="INFO",
        format="[%(name)s] %(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    asyncio.run(main())
