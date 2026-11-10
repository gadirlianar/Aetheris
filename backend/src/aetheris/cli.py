"""CLI (`aetheris research`, `aetheris replay`, `aetheris eval`, `aetheris serve`)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from aetheris.agents.renderer import compute_metrics
from aetheris.core.config import Settings
from aetheris.core.events import Event, EventType
from aetheris.core.models import RetrievalMode
from aetheris.demo import DEMO_PAPERS, DemoLLM, DemoSource
from aetheris.evals.runner import format_ablation_markdown, run_ablation_suite
from aetheris.llm.hashing import HashingEmbedder
from aetheris.orchestrator.engine import RunStatus
from aetheris.pipeline import build_research_graph, replay_research, run_research
from aetheris.tracing.store import TraceStore

app = typer.Typer(
    name="aetheris",
    help="Claim-level verifiable deep research engine over scientific literature.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console(stderr=True)
stdout = Console()


def _live_printer(event: Event) -> None:
    t = event.type
    d = event.data
    if t is EventType.NODE_STARTED:
        console.print(f"[bold cyan]▸ {event.node}[/]")
    elif t is EventType.PROGRESS:
        console.print(f"  [dim]{d.get('message', '')}[/]")
    elif t is EventType.SOURCE_QUERY and not d.get("ok", True):
        console.print(f"  [yellow]! {d.get('source')} query failed: {d.get('error')}[/]")
    elif t is EventType.NODE_RETRY:
        console.print(
            f"  [yellow]↻ retrying {event.node} in {d.get('delay_s')}s ({d.get('error')})[/]"
        )
    elif t is EventType.NODE_FAILED:
        console.print(f"  [bold red]✗ {event.node}: {d.get('error')}[/]")


@app.command()
def research(
    question: Annotated[str, typer.Argument(help="Research question to investigate.")],
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Write the Markdown report to this file."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Override LLM model id.")] = None,
    demo: Annotated[
        bool,
        typer.Option(
            "--demo",
            help="Run offline against the built-in curated scientific corpus & deterministic LLM.",
        ),
    ] = False,
    retrieval_mode: Annotated[
        RetrievalMode,
        typer.Option(help="Retrieval mode: dense, bm25, hybrid, or graph_hybrid."),
    ] = "graph_hybrid",
    skeptic_rounds: Annotated[
        int, typer.Option(help="Max adversarial Skeptic revision loops.")
    ] = 1,
) -> None:
    """Run the full research, GraphRAG, and claim-verification pipeline on QUESTION."""
    settings = Settings(retrieval_mode=retrieval_mode, max_skeptic_rounds=skeptic_rounds)
    if model:
        settings.llm_model = model
    store = TraceStore(settings.trace_db)
    try:
        if demo:
            result = asyncio.run(
                run_research(
                    question,
                    settings=settings,
                    llm=DemoLLM(inject_initial_overclaim=True),
                    embedder=HashingEmbedder(dim=512),
                    sources=(
                        DemoSource(DEMO_PAPERS[:5], name="arxiv"),
                        DemoSource(DEMO_PAPERS[3:], name="openalex"),
                    ),
                    trace_store=store,
                    on_event=_live_printer,
                )
            )
        else:
            result = asyncio.run(
                run_research(question, settings=settings, trace_store=store, on_event=_live_printer)
            )
    finally:
        store.close()

    if result.status is not RunStatus.COMPLETED or not result.state.report_markdown:
        console.print(f"[bold red]Run {result.status.value}:[/] {result.error}")
        raise typer.Exit(code=1)

    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    saved = settings.runs_dir / f"{result.state.run_id}.md"
    saved.write_text(result.state.report_markdown, encoding="utf-8")
    if output is not None:
        output.write_text(result.state.report_markdown, encoding="utf-8")

    m = compute_metrics(result.state)
    console.print(
        Panel.fit(
            f"run [bold]{result.state.run_id}[/] · "
            f"[green]{m.supported} supported[/], [red]{m.refuted} refuted[/], "
            f"[yellow]{m.not_enough_info} NEI[/] · "
            f"citation accuracy [bold]{m.citation_accuracy:.0%}[/] · "
            f"skeptic rounds [bold]{m.skeptic_rounds}[/] · "
            f"conflicts [bold]{m.conflicts_resolved}[/] · "
            f"cost [bold]${m.total_cost_usd:.5f}[/] · saved to {saved}",
            border_style="green",
        )
    )
    stdout.print(Markdown(result.state.report_markdown))


@app.command("replay")
def replay(
    run_id: Annotated[str, typer.Argument(help="Existing run_id to resume or fork from disk.")],
    from_node: Annotated[
        str,
        typer.Option(
            "--from-node",
            help=(
                "Graph node to resume from (plan, retrieve, synthesize, "
                "extract_claims, verify, skeptic, contradiction, render)."
            ),
        ),
    ] = "verify",
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Write the replayed Markdown report to this file."),
    ] = None,
    demo: Annotated[
        bool,
        typer.Option(
            "--demo",
            help="Replay using the deterministic offline LLM and corpus.",
        ),
    ] = False,
) -> None:
    """Deterministically replay or fork a recorded research run from any checkpointed node."""
    settings = Settings()
    if not settings.trace_db.exists():
        console.print(f"[bold red]Trace database not found at {settings.trace_db}[/]")
        raise typer.Exit(code=1)

    store = TraceStore(settings.trace_db)
    try:
        use_demo = demo or settings.gemini_api_key is None
        if use_demo:
            result = asyncio.run(
                replay_research(
                    run_id,
                    from_node=from_node,
                    trace_store=store,
                    settings=settings,
                    llm=DemoLLM(inject_initial_overclaim=False),
                    embedder=HashingEmbedder(dim=512),
                    sources=(
                        DemoSource(DEMO_PAPERS[:5], name="arxiv"),
                        DemoSource(DEMO_PAPERS[3:], name="openalex"),
                    ),
                    on_event=_live_printer,
                )
            )
        else:
            result = asyncio.run(
                replay_research(
                    run_id,
                    from_node=from_node,
                    trace_store=store,
                    settings=settings,
                    on_event=_live_printer,
                )
            )
    except ValueError as exc:
        console.print(f"[bold red]Replay error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        store.close()

    if result.status is not RunStatus.COMPLETED or not result.state.report_markdown:
        console.print(f"[bold red]Replay {result.status.value}:[/] {result.error}")
        raise typer.Exit(code=1)

    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    saved = settings.runs_dir / f"{result.state.run_id}.md"
    saved.write_text(result.state.report_markdown, encoding="utf-8")
    if output is not None:
        output.write_text(result.state.report_markdown, encoding="utf-8")

    m = compute_metrics(result.state)
    console.print(
        Panel.fit(
            f"replayed [bold]{run_id}[/] -> [bold]{result.state.run_id}[/] "
            f"(from [cyan]{from_node}[/], steps={len(result.steps)}) · "
            f"[green]{m.supported} supported[/], [red]{m.refuted} refuted[/], "
            f"[yellow]{m.not_enough_info} NEI[/] · saved to {saved}",
            border_style="cyan",
        )
    )
    stdout.print(Markdown(result.state.report_markdown))


@app.command("eval")
def run_eval(
    output_md: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Write Markdown ablation report to this path."),
    ] = None,
    output_json: Annotated[
        Path | None,
        typer.Option("--json", help="Write full JSON benchmark results to this path."),
    ] = None,
) -> None:
    """Run the 5-way SciFact-Aetheris retrieval & verification ablation benchmark suite."""
    summaries = asyncio.run(run_ablation_suite())
    table = Table(title="SciFact-Aetheris Ablation Study Results (60 Claims, 1000x Bootstrap CIs)")
    for col in (
        "Configuration",
        "Retrieval",
        "Guard",
        "Verdict Acc",
        "Macro-F1 (95% CI)",
        "Recall@K",
        "MRR",
        "Span IoU",
        "ECE (↓)",
        "Brier (↓)",
        "Caught",
    ):
        table.add_column(col)

    for s in summaries:
        ci_lo, ci_hi = s.macro_f1_ci_95
        table.add_row(
            s.config_name,
            s.retrieval_mode,
            "Yes" if s.grounding_guard else "No",
            f"{s.verdict_accuracy:.1%}",
            f"{s.verdict_macro_f1:.1%} [{ci_lo:.1%}, {ci_hi:.1%}]",
            f"{s.retrieval_recall_at_k:.1%}",
            f"{s.retrieval_mrr:.3f}",
            f"{s.mean_span_iou:.1%}",
            f"{s.ece:.3f}",
            f"{s.brier_score:.3f}",
            str(s.hallucinations_caught),
        )
    stdout.print(table)

    md = format_ablation_markdown(summaries)
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(md, encoding="utf-8")
        console.print(f"Saved Markdown ablation table to {output_md}")
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = [s.model_dump(mode="json") for s in summaries]
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Saved JSON benchmark report to {output_json}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Start the FastAPI backend server for the interactive Next.js dashboard."""
    uvicorn.run("aetheris.api.app:app", host=host, port=port, reload=False)


@app.command("runs")
def list_runs(limit: Annotated[int, typer.Option(help="Max rows.")] = 20) -> None:
    """List recorded runs from the local SQLite trace store."""
    settings = Settings()
    if not settings.trace_db.exists():
        console.print("No runs recorded yet.")
        return
    store = TraceStore(settings.trace_db)
    rows = store.list_runs(limit=limit)
    store.close()

    table = Table(title="Recorded runs")
    for col in (
        "run_id",
        "status",
        "llm_calls",
        "total_tokens",
        "cost_usd",
        "duration_ms",
        "question",
    ):
        table.add_column(col)
    for r in rows:
        cost = float(r.get("cost_usd") or 0.0)
        table.add_row(
            r["run_id"],
            r["status"],
            str(r["llm_calls"]),
            str(r["total_tokens"]),
            f"${cost:.5f}",
            str(r["duration_ms"] or ""),
            (r["question"] or "")[:60],
        )
    stdout.print(table)


@app.command()
def inspect(run_id: Annotated[str, typer.Argument(help="Run id to inspect.")]) -> None:
    """Print the event timeline of a recorded run."""
    store = TraceStore(Settings().trace_db)
    events = store.get_events(run_id)
    store.close()
    if not events:
        console.print(f"No events found for run {run_id}")
        raise typer.Exit(code=1)
    for ev in events:
        node = f"[{ev.node}] " if ev.node else ""
        stdout.print(f"{ev.seq:03d}  {ev.type.value:<16} {node}{ev.data}")


@app.command()
def graph() -> None:
    """Print the pipeline graph as a Mermaid diagram."""
    stdout.print(build_research_graph().to_mermaid())
