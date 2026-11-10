"""Tests for the 60-claim SciFact-Aetheris evaluation suite, FastAPI endpoints, and CLI commands."""

from __future__ import annotations

from pathlib import Path

import httpx
from typer.testing import CliRunner

from aetheris.api.app import create_app
from aetheris.cli import app as cli_app
from aetheris.core.config import Settings
from aetheris.evals.metrics import span_iou
from aetheris.evals.runner import format_ablation_markdown, run_ablation_suite


def test_span_iou_calculation() -> None:
    assert span_iou(10, 20, 10, 20) == 1.0
    assert span_iou(10, 20, 20, 30) == 0.0
    assert span_iou(10, 30, 20, 40) == 10 / 30
    assert span_iou(20, 10, 10, 20) == 0.0


async def test_ablation_suite_monotonic_gains() -> None:
    summaries = await run_ablation_suite()
    by_name = {s.config_name: s for s in summaries}

    full = by_name["full_aetheris"]
    dense = by_name["dense_only"]
    bm25 = by_name["bm25_only"]
    hybrid = by_name["hybrid_2way_rrf"]
    no_guard = by_name["no_grounding_guard"]

    assert full.total_examples == 60
    assert 0.80 <= full.verdict_macro_f1 <= 0.92
    assert full.verdict_macro_f1 > hybrid.verdict_macro_f1
    assert hybrid.verdict_macro_f1 > bm25.verdict_macro_f1 > dense.verdict_macro_f1
    assert full.verdict_macro_f1 > no_guard.verdict_macro_f1
    assert full.hallucinations_caught == 5
    assert no_guard.hallucinations_caught == 0
    assert full.brier_score < no_guard.brier_score

    # Verify 95% Bootstrap CIs, 3x3 Confusion Matrix, and Error Taxonomy
    assert full.macro_f1_ci_95[0] < full.verdict_macro_f1 < full.macro_f1_ci_95[1]
    assert sum(sum(row.values()) for row in full.confusion_matrix.values()) == 60
    assert sum(full.error_taxonomy.values()) == sum(1 for p in full.predictions if not p.correct)

    md = format_ablation_markdown(summaries)
    assert "full_aetheris" in md
    assert "## Confusion Matrix" in md
    assert "Error Taxonomy & Failure Mode Breakdown" in md


async def test_fastapi_endpoints(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "api_data")
    fastapi_app = create_app(settings, seed_demo=False)

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        presets = await client.get("/api/presets")
        assert presets.status_code == 200
        assert len(presets.json()) >= 4

        sync_run = await client.post(
            "/api/runs/sync",
            json={
                "question": "Do retrieval-augmented models reduce hallucinations?",
                "mode": "demo",
                "retrieval_mode": "graph_hybrid",
                "max_skeptic_rounds": 1,
            },
        )
        assert sync_run.status_code == 200
        payload = sync_run.json()
        run_id = payload["run_id"]
        assert payload["metrics"]["support_rate"] == 1.0
        assert payload["metrics"]["conflicts_resolved"] >= 1

        replay_res = await client.post(
            f"/api/runs/{run_id}/replay",
            json={"from_node": "verify", "mode": "demo"},
        )
        assert replay_res.status_code == 200
        assert replay_res.json()["status"] == "completed"
        assert replay_res.json()["run_id"] != run_id

        runs_list = await client.get("/api/runs")
        assert runs_list.status_code == 200
        assert any(r["run_id"] == run_id for r in runs_list.json())

        detail = await client.get(f"/api/runs/{run_id}")
        assert detail.status_code == 200
        assert detail.json()["run_id"] == run_id

        stream = await client.get(f"/api/runs/{run_id}/stream")
        assert stream.status_code == 200
        assert "run_finished" in stream.text

        evals = await client.get("/api/evals")
        assert evals.status_code == 200
        assert len(evals.json()["summaries"]) == 5


def test_cli_commands(tmp_path: Path, monkeypatch: object) -> None:
    import pytest

    mp = monkeypatch if isinstance(monkeypatch, pytest.MonkeyPatch) else None
    if mp is not None:
        mp.setenv("AETHERIS_DATA_DIR", str(tmp_path / "cli_data"))

    runner = CliRunner()
    out_md = tmp_path / "report.md"
    replay_md = tmp_path / "replay_report.md"
    eval_md = tmp_path / "ablation.md"
    eval_json = tmp_path / "ablation.json"

    res_graph = runner.invoke(cli_app, ["graph"])
    assert res_graph.exit_code == 0
    assert "flowchart TD" in res_graph.stdout
    assert "contradiction" in res_graph.stdout

    res_eval = runner.invoke(cli_app, ["eval", "-o", str(eval_md), "--json", str(eval_json)])
    assert res_eval.exit_code == 0
    assert eval_md.exists()
    assert eval_json.exists()

    res_research = runner.invoke(
        cli_app,
        [
            "research",
            "How does GraphRAG improve multi-hop retrieval?",
            "--demo",
            "-o",
            str(out_md),
        ],
    )
    assert res_research.exit_code == 0
    assert out_md.exists()

    res_runs = runner.invoke(cli_app, ["runs"])
    assert res_runs.exit_code == 0

    store = Settings().trace_db
    from aetheris.tracing.store import TraceStore

    ts = TraceStore(store)
    recorded_id = ts.list_runs(limit=1)[0]["run_id"]
    ts.close()

    res_inspect = runner.invoke(cli_app, ["inspect", recorded_id])
    assert res_inspect.exit_code == 0

    res_replay = runner.invoke(
        cli_app,
        [
            "replay",
            recorded_id,
            "--from-node",
            "verify",
            "--demo",
            "-o",
            str(replay_md),
        ],
    )
    assert res_replay.exit_code == 0
    assert replay_md.exists()
