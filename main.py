#!/usr/bin/env python3
# main.py
"""
Smart Research Assistant — Entry Point

Usage:
    # Interactive mode (with stable user ID for memory)
    python main.py --user-id alice

    # Single query mode
    python main.py --query "What are the latest trends in agentic AI?" --user-id alice

    # Demo mode (runs 2 preset queries to demonstrate caching)
    python main.py --demo

    # User memory demo (3-query scenario: interests → "who am I?")
    python main.py --user-memory-demo --user-id alice
"""

import asyncio
import argparse
import logging
import sys
import uuid

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents.orchestrator import run_research_pipeline
from config.settings import settings

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.adk").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

console = Console()

DEMO_QUERIES = [
    "What are the key trends in multi-agent AI systems in 2025?",
    "What are the key trends in multi-agent AI systems in 2025?",  # Same query → should hit cache
]


# ─────────────────────────────────────────────────────────────────────────────
# Core run function
# ─────────────────────────────────────────────────────────────────────────────
async def run_query(query: str, session_id: str, label: str = "", user_id: str = "default-user") -> None:
    """Run a single query through the pipeline and pretty-print the result."""

    console.print(
        Panel(
            f"[bold cyan]{query}[/bold cyan]",
            title=f"[yellow]{'Query' if not label else label}[/yellow]",
            border_style="blue",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running research pipeline…", total=None)

        try:
            result = await run_research_pipeline(
                user_query=query,
                session_id=session_id,
                user_id=user_id,
            )
            progress.remove_task(task)
        except Exception as e:
            progress.remove_task(task)
            console.print(f"[bold red]Pipeline error:[/bold red] {e}")
            return

    console.print("\n")
    console.print(Markdown(result))
    console.print(
        Panel(
            f"[green]Session:[/green] {session_id}\n"
            f"[green]User ID:[/green] {user_id}\n"
            f"[green]Langfuse:[/green] {settings.LANGFUSE_HOST}",
            title="[dim]Run Info[/dim]",
            border_style="dim",
        )
    )
    console.print("\n" + "─" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode — shows caching in action
# ─────────────────────────────────────────────────────────────────────────────
async def run_demo() -> None:
    console.print(
        Panel(
            "[bold]Smart Research Assistant — Demo Mode[/bold]\n\n"
            "Running 2 queries. The second is identical to the first,\n"
            "demonstrating [green]LiteLLM Redis cache[/green] — it should return faster.\n\n"
            "Traces visible in [cyan]Langfuse[/cyan] at: " + settings.LANGFUSE_HOST,
            title="[yellow]Demo[/yellow]",
            border_style="yellow",
        )
    )

    session_id = str(uuid.uuid4())

    for i, query in enumerate(DEMO_QUERIES, 1):
        label = f"Query {i}" + (" (⚡ should hit Redis cache)" if i == 2 else " (first call → populates cache)")
        await run_query(query, session_id=session_id, label=label, user_id="demo-user")
        if i < len(DEMO_QUERIES):
            console.print("[dim]Pausing 2s before next query…[/dim]")
            await asyncio.sleep(2)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────
async def run_user_memory_demo(user_id: str) -> None:
    """
    3-query demo that shows user memory in action:
      Q1: Research Data Science
      Q2: Research MLOps
      Q3: "Who am I? What is my passion?"  ← answered from stored history
    """
    console.print(
        Panel(
            f"[bold]User Memory Demo[/bold]\n\n"
            f"User ID: [cyan]{user_id}[/cyan]\n\n"
            "Q1 and Q2 store your interests in PostgreSQL.\n"
            "Q3 asks [green]'Who am I?'[/green] — the orchestrator answers from history\n"
            "without calling any research sub-agents.",
            title="[magenta]User Memory Demo[/magenta]",
            border_style="magenta",
        )
    )

    queries = [
        ("What is the role of a Data Scientist? What skills and responsibilities do they have?",
         "Q1 — Data Scientist (stores interest)"),
        ("What is MLOps? What are the key practices and tools in the MLOps ecosystem?",
         "Q2 — MLOps (stores interest)"),
        ("Who am I? Based on what I've been asking about, what seems to be my passion and career focus?",
         "Q3 — Who am I? (answered from memory)"),
    ]

    for query, label in queries:
        session_id = str(uuid.uuid4())
        await run_query(query, session_id=session_id, label=label, user_id=user_id)
        if query != queries[-1][0]:
            console.print("[dim]Storing interaction… moving to next question.[/dim]\n")
            await asyncio.sleep(1)


async def run_interactive(user_id: str = "default-user") -> None:
    console.print(
        Panel(
            "[bold]Smart Research Assistant[/bold]\n"
            "Multi-agent research powered by Google ADK + AWS Bedrock + LiteLLM\n\n"
            "Type your research question and press Enter.\n"
            "[dim]Commands: 'quit' or 'exit' to stop, 'new' to start a new session[/dim]",
            title="[cyan]Interactive Mode[/cyan]",
            border_style="cyan",
        )
    )

    session_id = str(uuid.uuid4())
    console.print(f"[dim]Session ID: {session_id} | User ID: {user_id}[/dim]\n")

    while True:
        try:
            query = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting.[/yellow]")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if query.lower() == "new":
            session_id = str(uuid.uuid4())
            console.print(f"[dim]New session: {session_id}[/dim]\n")
            continue

        await run_query(query, session_id=session_id, user_id=user_id)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Smart Research Assistant — Multi-Agent Pipeline"
    )
    parser.add_argument("--query", "-q", type=str, help="Run a single query and exit")
    parser.add_argument("--demo", action="store_true", help="Run the caching demo (2 identical queries)")
    parser.add_argument(
        "--user-memory-demo", action="store_true",
        help="Run the 3-query user memory demo (Data Scientist → MLOps → Who am I?)"
    )
    parser.add_argument(
        "--user-id", type=str, default="default-user",
        help="Stable user identifier for memory recall (default: default-user)"
    )
    parser.add_argument(
        "--clear-user-memory", action="store_true",
        help="Delete all stored history for --user-id and exit"
    )
    args = parser.parse_args()

    if args.clear_user_memory:
        async def _clear():
            from config.memory_store import user_memory_store
            deleted = await user_memory_store.clear(args.user_id)
            console.print(f"[green]Cleared {deleted} memory record(s) for user '{args.user_id}'[/green]")
        asyncio.run(_clear())
    elif args.demo:
        asyncio.run(run_demo())
    elif args.user_memory_demo:
        asyncio.run(run_user_memory_demo(user_id=args.user_id))
    elif args.query:
        session_id = str(uuid.uuid4())
        asyncio.run(run_query(args.query, session_id=session_id, user_id=args.user_id))
    else:
        asyncio.run(run_interactive(user_id=args.user_id))


if __name__ == "__main__":
    main()
