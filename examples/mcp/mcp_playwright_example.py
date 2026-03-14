"""
MCP Playwright Example - Agent with MCP Browser Tools.

This example demonstrates the nagents Agent with MCP (Model Context Protocol)
integration. The Agent connects to a Playwright MCP server, discovers its
browser automation tools, and uses them to browse the web autonomously via
the LLM.

The flow is fully integrated:
    1. Agent connects to Playwright MCP server during initialize()
    2. MCP tools are auto-discovered and prefixed (mcp_playwright_*)
    3. LLM calls browser tools as needed to fulfill the user query
    4. Agent routes tool calls to the MCP server and feeds results back

Prerequisites:
    - Install: pip install nagents python-dotenv rich
    - Node.js and npx installed (for Playwright MCP server)

Usage:
    python mcp_playwright_example.py
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
from nagents import ReasoningChunkEvent
from nagents import SessionManager
from nagents import StdioServerParameters
from nagents import TextChunkEvent
from nagents import TextDoneEvent
from nagents import ToolCallEvent
from nagents import ToolResultEvent

load_dotenv()

logger = getLogger(__name__)
console = Console()


def get_azure_provider() -> Provider:
    """Create a Provider configured for Azure OpenAI V1 API."""
    endpoint = os.getenv("AZURE_DEPLOYED_MODEL_ENDPOINT")
    api_key = os.getenv("AZURE_DEPLOYED_MODEL_KEY")
    model_name = os.getenv("AZURE_DEPLOYED_MODEL_NAME")

    if not endpoint:
        raise ValueError("AZURE_DEPLOYED_MODEL_ENDPOINT not set. Please set it in your .env file.")
    if not api_key:
        raise ValueError("AZURE_DEPLOYED_MODEL_KEY not set. Please set it in your .env file.")
    if not model_name:
        raise ValueError("AZURE_DEPLOYED_MODEL_NAME not set. Please set it in your .env file.")

    base_url = f"{endpoint.rstrip('/')}/openai"

    console.print(f"[dim]Azure V1 endpoint: {base_url}[/dim]")
    console.print(f"[dim]Model: {model_name}[/dim]")

    return Provider(
        provider_type=ProviderType.AZURE_OPENAI_COMPATIBLE_V1,
        api_key=api_key,
        model=model_name,
        base_url=base_url,
    )


async def main() -> None:
    """Run Agent with MCP Playwright integration."""
    console.print(Panel.fit("[bold blue]nagents MCP Playwright Example[/bold blue]"))

    try:
        provider = get_azure_provider()
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        console.print("\n[yellow]Please ensure your .env file contains:[/yellow]")
        console.print("  AZURE_DEPLOYED_MODEL_ENDPOINT=https://your-resource.cognitiveservices.azure.com")
        console.print("  AZURE_DEPLOYED_MODEL_KEY=your-api-key")
        console.print("  AZURE_DEPLOYED_MODEL_NAME=your-deployment-name")
        return

    # --- Session setup ---
    examples_dir = Path(__file__).parent.parent
    session_manager = SessionManager(examples_dir / "sessions.db")
    session_id = f"mcp-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    # --- Log file path ---
    log_file = examples_dir / "logs" / f"{session_id}.txt"

    # --- Create Agent with MCP server ---
    config_path = Path(__file__).parent / "mcp_playwright_config.json"

    agent = Agent(
        provider=provider,
        session_manager=session_manager,
        system_prompt=(
            "You are a helpful assistant with access to browser automation tools. "
            "Use the browser tools to navigate websites, take snapshots, and extract information. "
            "When browsing, always take a snapshot after navigating to see the page content. "
            "Be concise in your final answer."
        ),
        streaming=True,
        log_file=log_file,
        max_tool_rounds=15,
        mcp_servers={
            "playwright": StdioServerParameters(
                command="npx",
                args=["-y", "@playwright/mcp@latest", "--browser=chrome", "--config", str(config_path)],
                timeout=60.0,
            ),
        },
    )

    try:
        # Initialize explicitly to see MCP connection logs
        console.print("\n[bold]Initializing agent (connecting to MCP servers)...[/bold]")
        await agent.initialize()
        console.print(f"[green]Agent initialized, model: {provider.model}[/green]")
        console.print(f"[green]MCP tools registered: {len(agent.tool_registry.get_all())}[/green]")
        console.print(f"[blue]Traffic log: {log_file}[/blue]")

        # --- Run with user query ---
        query = (
            "What are the latest news today? "
            + "Ensure you accept cookies etc. "
            + "Wait for 10s after opening a webpage to ensure all content is loaded, "
            + "scroll very slowly downward the page to load all the content "
            + "then give me a summary."
            + "Check bbc.com and reuters.com are prefered sources for news. but head to x.com as well to catch a glance"
        )
        console.print()
        console.print(Panel(f"[bold]User:[/bold] {query}", border_style="green"))
        console.print()
        response_text = ""

        async for event in agent.run(
            user_message=query,
            session_id=session_id,
            user_id="mcp-example-user",
        ):
            if isinstance(event, ReasoningChunkEvent):
                console.print(event.chunk, end="", style="dim italic")
            elif isinstance(event, TextChunkEvent):
                console.print(event.chunk, end="")
                response_text += event.chunk
            elif isinstance(event, TextDoneEvent):
                if not response_text:
                    response_text = event.text
                    console.print(event.text)
            elif isinstance(event, ToolCallEvent):
                console.print()
                tool_text = Text()
                tool_text.append("Tool Call: ", style="bold yellow")
                tool_text.append(event.name, style="cyan")
                tool_text.append("(", style="dim")
                args_str = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
                if len(args_str) > 120:
                    args_str = args_str[:120] + "..."
                tool_text.append(args_str, style="white")
                tool_text.append(")", style="dim")
                console.print(tool_text)
            elif isinstance(event, ToolResultEvent):
                result_text = Text()
                result_text.append("Tool Result: ", style="bold green")
                result_text.append(f"`{event.name}` ", style="cyan")
                if event.error:
                    result_text.append(f"ERROR: {event.error[:100]}", style="red")
                else:
                    result_str = str(event.result) if event.result else ""
                    if len(result_str) > 150:
                        result_str = result_str[:150] + "..."
                    result_text.append(result_str, style="white")
                result_text.append(f" ({event.duration_ms:.1f}ms)", style="dim")
                console.print(result_text)
            elif isinstance(event, ErrorEvent):
                console.print()
                error_text = Text()
                error_text.append("ERROR: ", style="bold red")
                error_text.append(event.message, style="red")
                if event.code:
                    error_text.append(f" (code: {event.code})", style="dim red")
                if event.recoverable:
                    error_text.append(" [recoverable]", style="yellow")
                console.print(error_text)
            elif isinstance(event, DoneEvent):
                console.print()
                done_text = Text()
                done_text.append("Generation complete", style="bold green")
                done_text.append(" - ", style="dim")
                done_text.append(f"{len(event.final_text)} chars", style="dim")
                if event.session_id:
                    done_text.append(f" (session: {event.session_id})", style="dim blue")
                console.print(done_text)

                usage_text = Text()
                usage_text.append("Usage: ", style="bold blue")
                usage_text.append(
                    f"prompt={event.usage.prompt_tokens}, "
                    f"completion={event.usage.completion_tokens}, "
                    f"total={event.usage.total_tokens}",
                    style="dim",
                )
                if event.usage.session:
                    usage_text.append(
                        f" | Session total: {event.usage.session.total_tokens}",
                        style="dim cyan",
                    )
                console.print(usage_text)

        console.print()
        console.print(f"[dim]Traffic logged to: {log_file.absolute()}[/dim]")
    finally:
        await agent.close()

    console.print("[dim]Example complete.[/dim]")


if __name__ == "__main__":
    basicConfig(
        level="INFO",
        format="[%(name)s] %(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    asyncio.run(main())
