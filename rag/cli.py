"""CLI for RAG Assistant.

Provides command-line interface for ingestion, querying, and management.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.config import Config
from rag.ingest import DocumentIngester
from rag.pipeline import RAGPipeline, PipelineError

# Initialize Typer app
app = typer.Typer(
    name="rag",
    help="RAG Assistant - Question answering over local knowledge bases",
    add_completion=False,
)

console = Console()

# Global state for storing last query results
_pipeline: Optional[RAGPipeline] = None


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    log_level = Config.LOG_LEVEL if not verbose else "DEBUG"

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def get_pipeline() -> RAGPipeline:
    """Get or create the pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@app.command()
def ingest(
    kb_path: Optional[Path] = typer.Option(
        None,
        "--kb-path",
        "-k",
        help="Path to knowledge base directory",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-ingestion even if index exists",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Ingest documents from the knowledge base into the vector store."""
    setup_logging(verbose)

    kb_dir = kb_path or Config.KB_PATH

    if not kb_dir.exists():
        console.print(f"[red]Error:[/red] Knowledge base path not found: {kb_dir}")
        raise typer.Exit(1)

    console.print(f"[blue]Ingesting documents from:[/blue] {kb_dir}")

    try:
        ingester = DocumentIngester(kb_path=kb_dir)
        num_chunks = ingester.ingest()

        if num_chunks > 0:
            console.print(
                f"\n[green]Success![/green] Ingested {num_chunks} chunks "
                f"from {len(list(kb_dir.glob('*.md')))} documents."
            )
        else:
            console.print(
                "\n[yellow]Warning:[/yellow] No documents found to ingest. "
                f"Add markdown files to {kb_dir}"
            )

    except Exception as e:
        console.print(f"\n[red]Error during ingestion:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask"),
    top_k: int = typer.Option(
        None,
        "--top-k",
        "-k",
        help="Number of documents to retrieve",
    ),
    temperature: float = typer.Option(
        None,
        "--temperature",
        "-t",
        help="LLM temperature (0-1)",
    ),
    no_sources: bool = typer.Option(
        False,
        "--no-sources",
        help="Don't show source documents",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Ask a question and get an answer from the knowledge base."""
    setup_logging(verbose)

    pipeline = get_pipeline()

    # Check health first
    health = pipeline.check_health()
    if not health["healthy"]:
        if not health["retriever"]:
            console.print(
                "[red]Error:[/red] No ingested documents found. "
                "Run 'rag ingest' first."
            )
            raise typer.Exit(1)
        if not health["llm"]:
            console.print(
                f"[red]Error:[/red] Cannot connect to Ollama. "
                f"{health.get('llm_error', 'Check if Ollama is running.')}"
            )
            raise typer.Exit(1)

    try:
        answer = pipeline.ask(
            question=question,
            top_k=top_k,
            temperature=temperature,
        )

        # Display answer
        console.print()
        console.print(Panel(answer, title="Answer", border_style="green"))

        # Display sources if requested
        if not no_sources:
            sources = pipeline.get_sources()
            if sources:
                console.print("\n[dim]Sources:[/dim]")
                for source in sources:
                    source_name = source["metadata"].get("source", "Unknown")
                    chunk_idx = source["metadata"].get("chunk_index", 0) + 1
                    score = source["score"]
                    console.print(
                        f"  [dim]•[/dim] {source_name} "
                        f"(chunk {chunk_idx}, score: {score:.2f})"
                    )

    except PipelineError as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("show-sources")
def show_sources(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Show detailed sources from the last query."""
    setup_logging(verbose)

    pipeline = get_pipeline()
    sources = pipeline.get_sources()

    if not sources:
        console.print(
            "[yellow]No sources available.[/yellow] Run 'rag ask' first."
        )
        return

    console.print("\n[bold]Retrieved Sources[/bold]")
    console.print("=" * 50)

    for source in sources:
        source_name = source["metadata"].get("source", "Unknown")
        chunk_idx = source["metadata"].get("chunk_index", 0)
        chunk_total = source["metadata"].get("chunk_total", "?")
        score = source["score"]
        content = source["content"]

        console.print(
            f"\n[cyan][{source['rank']}][/cyan] {source_name} "
            f"(chunk {chunk_idx + 1}/{chunk_total})"
        )
        console.print(f"    [dim]Score:[/dim] {score:.3f}")
        console.print(f"    [dim]Preview:[/dim] {content[:200]}...")


@app.command()
def config(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Show current configuration."""
    setup_logging(verbose)

    table = Table(title="RAG Assistant Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    config_items = [
        ("Ollama Model", Config.OLLAMA_MODEL),
        ("Ollama URL", Config.OLLAMA_BASE_URL),
        ("Embedding Model", Config.EMBEDDING_MODEL),
        ("Chunk Size", str(Config.CHUNK_SIZE)),
        ("Chunk Overlap", str(Config.CHUNK_OVERLAP)),
        ("Top-K", str(Config.RETRIEVER_TOP_K)),
        ("Score Threshold", str(Config.RETRIEVER_SCORE_THRESHOLD)),
        ("Temperature", str(Config.LLM_TEMPERATURE)),
        ("Max Tokens", str(Config.LLM_MAX_TOKENS)),
        ("KB Path", str(Config.KB_PATH)),
        ("ChromaDB Path", str(Config.CHROMA_PERSIST_DIR)),
    ]

    for name, value in config_items:
        table.add_row(name, value)

    console.print(table)


@app.command()
def health(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Check health of all components."""
    setup_logging(verbose)

    console.print("[bold]Checking RAG Assistant health...[/bold]\n")

    # Check knowledge base
    kb_path = Config.KB_PATH
    if kb_path.exists():
        md_files = list(kb_path.glob("*.md"))
        console.print(f"[green]✓[/green] Knowledge base: {len(md_files)} files in {kb_path}")
    else:
        console.print(f"[red]✗[/red] Knowledge base: Path not found ({kb_path})")

    # Check ChromaDB
    try:
        pipeline = get_pipeline()
        health_status = pipeline.check_health()

        if health_status["retriever"]:
            count = health_status.get("collection_count", 0)
            console.print(f"[green]✓[/green] Vector store: {count} chunks indexed")
        else:
            console.print(
                f"[red]✗[/red] Vector store: {health_status.get('retriever_error', 'Not ready')}"
            )

        # Check Ollama
        if health_status["llm"]:
            console.print(f"[green]✓[/green] Ollama: Connected ({Config.OLLAMA_MODEL})")
        else:
            console.print(
                f"[red]✗[/red] Ollama: {health_status.get('llm_error', 'Not connected')}"
            )

        # Overall status
        if health_status["healthy"]:
            console.print("\n[green]All systems operational![/green]")
        else:
            console.print("\n[yellow]Some components need attention.[/yellow]")

    except Exception as e:
        console.print(f"[red]✗[/red] Health check failed: {e}")


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit",
    ),
) -> None:
    """RAG Assistant - Question answering over local knowledge bases."""
    if version:
        from rag import __version__
        console.print(f"RAG Assistant v{__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
