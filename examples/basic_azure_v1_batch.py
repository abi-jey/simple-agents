"""
Example demonstrating batch processing with Azure OpenAI V1 API.

Azure OpenAI Batch API offers 50% cost discount compared to real-time API.
Batch requests are processed asynchronously with 24-hour target turnaround.

REQUIREMENTS:
1. You must create a Global-Batch deployment type in Azure OpenAI Studio
   - Standard deployments (GlobalStandard) do NOT support batch processing
   - Go to Azure OpenAI Studio > Deployments > Create new > Select "Global-Batch" type
2. The deployment name must be used as the model name
3. Models available for Global-Batch: gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o3-mini, o4-mini, gpt-5, gpt-5.1

Error "invalid_deployment_type" means you need to create a Global-Batch deployment.
"""

import asyncio
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
from nagents import DoneEvent
from nagents import ErrorEvent
from nagents import Provider
from nagents import ProviderType
from nagents import SessionManager
from nagents import TextDoneEvent

load_dotenv()

logger = getLogger(__name__)
console = Console()

# Configuration - edit these values for your Azure OpenAI resource
AZURE_RESOURCE_NAME = "agent-test-abbas"  # Your Azure OpenAI resource name
AZURE_API_KEY = os.getenv("AZURE_DEPLOYED_MODEL_KEY")  # API key
# NOTE: You need a Global-Batch deployment type for batch processing
# Check your deployments in Azure OpenAI Studio and use the name of a Global-Batch deployment
BATCH_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYED_MODEL_NAME", "Kimi-K2.5")  # Model/deployment name


def get_azure_batch_provider() -> Provider:
    """Create a Provider configured for Azure OpenAI V1 API with batch support."""
    api_key = AZURE_API_KEY
    if not api_key:
        raise ValueError("AZURE_DEPLOYED_MODEL_NAME (API key) not set in environment.")

    resource_name = AZURE_RESOURCE_NAME
    model_name = BATCH_DEPLOYMENT_NAME

    # Azure OpenAI V1 API base URL format
    # https://{resource-name}.openai.azure.com/openai/v1/
    base_url = f"https://{resource_name}.openai.azure.com/openai"

    console.print(f"[dim]Azure V1 endpoint: {base_url}[/dim]")
    console.print(f"[dim]Model (deployment): {model_name}[/dim]")

    return Provider(
        provider_type=ProviderType.AZURE_OPENAI_COMPATIBLE_V1,
        api_key=api_key,
        model=model_name,
        base_url=base_url,
    )


async def main() -> None:
    """Main example demonstrating batch processing with Azure V1."""
    console.print(Panel.fit("[bold blue]nagents Azure V1 Batch Processing Example[/bold blue]"))
    console.print("[dim]Using Azure OpenAI V1 provider with batch mode[/dim]\n")

    try:
        provider = get_azure_batch_provider()
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        return

    session_manager = SessionManager(Path("sessions_azure_v1_batch.db"))
    session_id = f"azure-v1-batch-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    log_file = Path("logs") / f"{session_id}.txt"

    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        tools=[],  # No tools for simple batch example
        system_prompt="You are a helpful assistant.",
        streaming=False,
        batch=True,
        batch_poll_interval=10.0,
        log_file=log_file,
    )

    try:
        await agent.initialize()
        console.print(f"[green]Agent initialized, model: {provider.model}[/green]")
        console.print(f"[blue]HTTP logging to: {log_file}[/blue]")
        console.print("[yellow]Batch mode enabled - 50% cost discount![/yellow]")

        query = "What is the capital of France? Give a brief answer."
        console.print(Panel(f"[bold]User:[/bold] {query}", border_style="green"))
        console.print()

        async for event in agent.run(user_message=query, session_id=session_id):
            if isinstance(event, TextDoneEvent):
                console.print(
                    Panel(
                        event.text,
                        title=f"[bold blue]Response[/bold blue] (finish_reason: {event.finish_reason.value})",
                        border_style="blue",
                    )
                )
                if event.extra:
                    console.print(f"[dim]Response ID: {event.extra.get('id', 'N/A')}[/dim]")
            elif isinstance(event, ErrorEvent):
                error_text = Text()
                error_text.append("ERROR: ", style="bold red")
                error_text.append(event.message, style="red")
                if event.code:
                    error_text.append(f" (code: {event.code})", style="dim red")
                console.print(error_text)
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
                console.print(f"[dim]Finish reason: {event.finish_reason.value}[/dim]")
                console.print("[green]Cost savings: 50% compared to real-time API[/green]")
                console.print(f"[dim]HTTP traffic logged to: {log_file.absolute()}[/dim]")

    finally:
        await agent.close()

    console.print("\n[dim]Azure V1 batch example complete.[/dim]")


if __name__ == "__main__":
    basicConfig(
        level="INFO",
        format="[%(name)s] %(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    asyncio.run(main())
