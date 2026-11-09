"""FastAPI server with REST endpoints and Server-Sent Events (SSE) for live agent runs."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aetheris import __version__
from aetheris.agents.renderer import compute_metrics
from aetheris.core.config import Settings
from aetheris.core.events import Event, EventType
from aetheris.core.models import ResearchState, RetrievalMode
from aetheris.demo import DEMO_PAPERS, PRESET_QUESTIONS, DemoLLM, DemoSource
from aetheris.evals.metrics import BenchmarkSummary
from aetheris.evals.runner import format_ablation_markdown, run_ablation_suite
from aetheris.llm.hashing import HashingEmbedder
from aetheris.pipeline import build_research_graph, run_research
from aetheris.tracing.store import TraceStore


class CreateRunRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    mode: Literal["auto", "demo", "live"] = "auto"
    retrieval_mode: RetrievalMode = "graph_hybrid"
    max_skeptic_rounds: int = Field(default=1, ge=0, le=3)


class EvalResponse(BaseModel):
    summaries: list[BenchmarkSummary]
    markdown: str


class AppState:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.queues: dict[str, list[asyncio.Queue[Event | None]]] = {}
        self.tasks: set[asyncio.Task[Any]] = set()
        self.eval_cache: EvalResponse | None = None

    def open_store(self) -> TraceStore:
        return TraceStore(self.settings.trace_db)

    def broadcast(self, event: Event) -> None:
        for q in self.queues.get(event.run_id, []):
            q.put_nowait(event)
        if event.type is EventType.RUN_FINISHED:
            for q in self.queues.get(event.run_id, []):
                q.put_nowait(None)


async def _execute_run(
    app_state: AppState,
    run_id: str,
    req: CreateRunRequest,
    *,
    use_demo: bool,
) -> dict[str, Any]:
    cfg = app_state.settings.model_copy(
        update={
            "retrieval_mode": req.retrieval_mode,
            "max_skeptic_rounds": req.max_skeptic_rounds,
        }
    )
    store = app_state.open_store()
    try:
        if use_demo:
            result = await run_research(
                req.question,
                settings=cfg,
                llm=DemoLLM(inject_initial_overclaim=True),
                embedder=HashingEmbedder(dim=512),
                sources=(
                    DemoSource(DEMO_PAPERS[:5], name="arxiv"),
                    DemoSource(DEMO_PAPERS[3:], name="openalex"),
                ),
                trace_store=store,
                on_event=app_state.broadcast,
                run_id=run_id,
            )
        else:
            result = await run_research(
                req.question,
                settings=cfg,
                trace_store=store,
                on_event=app_state.broadcast,
                run_id=run_id,
            )
        row = store.get_run(run_id)
        events = [e.model_dump(mode="json") for e in store.get_events(run_id)]
    finally:
        store.close()

    return _build_run_payload(row, result.state, events)


def _build_run_payload(
    row: dict[str, Any] | None, state: ResearchState, events: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = compute_metrics(state).to_dict()
    return {
        "run_id": state.run_id,
        "question": state.question,
        "status": (row or {}).get("status", "completed"),
        "created_at": (row or {}).get("created_at"),
        "finished_at": (row or {}).get("finished_at"),
        "llm_calls": (row or {}).get("llm_calls", 0),
        "total_tokens": (row or {}).get("total_tokens", 0),
        "duration_ms": (row or {}).get("duration_ms", 0),
        "error": (row or {}).get("error"),
        "metrics": metrics,
        "state": state.model_dump(mode="json"),
        "events": events,
    }


def create_app(settings: Settings | None = None, *, seed_demo: bool = True) -> FastAPI:
    app_state = AppState(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if seed_demo:
            store = app_state.open_store()
            existing = store.list_runs(limit=1)
            store.close()
            if not existing:
                await _execute_run(
                    app_state,
                    "seed-rag-01",
                    CreateRunRequest(
                        question=PRESET_QUESTIONS[0]["question"],
                        mode="demo",
                        retrieval_mode="graph_hybrid",
                        max_skeptic_rounds=1,
                    ),
                    use_demo=True,
                )
        yield

    app = FastAPI(
        title="Aetheris API",
        description="Claim-level verifiable multi-agent scientific deep research engine.",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        s = app_state.settings
        return {
            "status": "ok",
            "version": __version__,
            "live_llm_configured": s.gemini_api_key is not None,
            "llm_model": s.llm_model,
            "embedding_model": s.embedding_model,
            "retrieval_mode": s.retrieval_mode,
            "graph_mermaid": build_research_graph().to_mermaid(),
        }

    @app.get("/api/presets")
    async def presets() -> list[dict[str, str]]:
        return list(PRESET_QUESTIONS)

    @app.get("/api/runs")
    async def list_runs(limit: int = 25) -> list[dict[str, Any]]:
        store = app_state.open_store()
        try:
            rows = store.list_runs(limit=limit)
            enriched: list[dict[str, Any]] = []
            for r in rows:
                full = store.get_run(r["run_id"])
                metrics: dict[str, Any] | None = None
                if full and full.get("final_state"):
                    state = ResearchState.model_validate_json(full["final_state"])
                    metrics = compute_metrics(state).to_dict()
                enriched.append({**r, "metrics": metrics})
            return enriched
        finally:
            store.close()

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        store = app_state.open_store()
        try:
            row = store.get_run(run_id)
            if row is None:
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
            events = [e.model_dump(mode="json") for e in store.get_events(run_id)]
            if row.get("final_state"):
                state = ResearchState.model_validate_json(row["final_state"])
            else:
                state = ResearchState(run_id=run_id, question=row["question"])
            return _build_run_payload(row, state, events)
        finally:
            store.close()

    @app.post("/api/runs")
    async def start_run(req: CreateRunRequest) -> dict[str, Any]:
        use_demo = req.mode == "demo" or (
            req.mode == "auto" and app_state.settings.gemini_api_key is None
        )
        run_id = uuid.uuid4().hex[:12]
        app_state.queues[run_id] = []
        task = asyncio.create_task(_execute_run(app_state, run_id, req, use_demo=use_demo))
        app_state.tasks.add(task)
        task.add_done_callback(app_state.tasks.discard)
        return {
            "run_id": run_id,
            "status": "running",
            "mode": "demo" if use_demo else "live",
        }

    @app.post("/api/runs/sync")
    async def run_sync(req: CreateRunRequest) -> dict[str, Any]:
        use_demo = req.mode == "demo" or (
            req.mode == "auto" and app_state.settings.gemini_api_key is None
        )
        run_id = uuid.uuid4().hex[:12]
        return await _execute_run(app_state, run_id, req, use_demo=use_demo)

    @app.get("/api/runs/{run_id}/stream")
    async def stream_run(run_id: str) -> StreamingResponse:
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        app_state.queues.setdefault(run_id, []).append(queue)

        async def event_generator() -> AsyncIterator[str]:
            store = app_state.open_store()
            try:
                existing = store.get_events(run_id)
                row = store.get_run(run_id)
            finally:
                store.close()

            seen_seq = 0
            for ev in existing:
                seen_seq = max(seen_seq, ev.seq)
                payload = json.dumps(ev.model_dump(mode="json", exclude={"snapshot"}))
                yield f"data: {payload}\n\n"
                if ev.type is EventType.RUN_FINISHED:
                    return

            if row and row.get("status") not in (None, "running"):
                return

            try:
                while True:
                    ev_or_end = await asyncio.wait_for(queue.get(), timeout=60.0)
                    if ev_or_end is None:
                        break
                    if ev_or_end.seq <= seen_seq:
                        continue
                    seen_seq = ev_or_end.seq
                    payload = json.dumps(ev_or_end.model_dump(mode="json", exclude={"snapshot"}))
                    yield f"data: {payload}\n\n"
                    if ev_or_end.type is EventType.RUN_FINISHED:
                        break
            except TimeoutError:
                yield 'data: {"type": "timeout"}\n\n'
            finally:
                if run_id in app_state.queues and queue in app_state.queues[run_id]:
                    app_state.queues[run_id].remove(queue)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/api/evals", response_model=EvalResponse)
    async def get_evals(refresh: bool = False) -> EvalResponse:
        if app_state.eval_cache is None or refresh:
            summaries = await run_ablation_suite()
            app_state.eval_cache = EvalResponse(
                summaries=summaries, markdown=format_ablation_markdown(summaries)
            )
        return app_state.eval_cache

    return app


app = create_app()
