"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Database,
  ExternalLink,
  FileSearch,
  FlaskConical,
  GitBranch,
  Layers,
  Network,
  Play,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Terminal,
  XCircle,
} from "lucide-react";
import {
  FALLBACK_EVALS,
  FALLBACK_PRESETS,
  FALLBACK_RUN,
} from "@/lib/fallbackData";
import type {
  BenchmarkSummary,
  Claim,
  EvidenceSpan,
  Paper,
  PresetQuestion,
  RetrievalMode,
  RunEvent,
  RunPayload,
  Verdict,
} from "@/lib/types";

type ActiveTab = "Inspector" | "Pipeline" | "GraphRAG" | "Benchmark";

const PIPELINE_NODES = [
  {
    id: "plan",
    label: "Planner",
    subtitle: "Hypothesis & Query Decomposition",
  },
  {
    id: "retrieve",
    label: "Retriever",
    subtitle: "arXiv + OpenAlex + Semantic Scholar",
  },
  {
    id: "synthesize",
    label: "Synthesizer",
    subtitle: "Evidence-Grounded Draft / Revision",
  },
  {
    id: "extract_claims",
    label: "Claim Extractor",
    subtitle: "Atomic Self-Contained Statements",
  },
  {
    id: "verify",
    label: "Verifier + Grounder",
    subtitle: "3-Pass Mechanical Quote Grounding",
  },
  {
    id: "skeptic",
    label: "Adversarial Skeptic",
    subtitle: "Contradiction & Gap Audit Loop",
  },
  {
    id: "render",
    label: "Renderer",
    subtitle: "Annotated Brief & Calibration Metrics",
  },
] as const;

function verdictBadge(verdict: Verdict | null) {
  if (verdict === "supported") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-0.5 text-xs font-semibold text-emerald-300">
        <CheckCircle2 className="h-3.5 w-3.5" />
        SUPPORTED
      </span>
    );
  }
  if (verdict === "refuted") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/40 bg-rose-500/10 px-2.5 py-0.5 text-xs font-semibold text-rose-300">
        <XCircle className="h-3.5 w-3.5" />
        REFUTED
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-0.5 text-xs font-semibold text-amber-300">
      <AlertTriangle className="h-3.5 w-3.5" />
      NOT ENOUGH INFO
    </span>
  );
}

export default function AetherisDashboard() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("Inspector");
  const [run, setRun] = useState<RunPayload>(FALLBACK_RUN);
  const [presets, setPresets] = useState<PresetQuestion[]>(FALLBACK_PRESETS);
  const [evals, setEvals] = useState<BenchmarkSummary[]>(FALLBACK_EVALS);
  const [selectedEvalIdx, setSelectedEvalIdx] = useState<number>(
    FALLBACK_EVALS.length - 1
  );

  const [questionInput, setQuestionInput] = useState<string>(
    FALLBACK_PRESETS[0]?.question ?? ""
  );
  const [retrievalMode, setRetrievalMode] =
    useState<RetrievalMode>("graph_hybrid");
  const [maxSkepticRounds, setMaxSkepticRounds] = useState<number>(1);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [liveModelName, setLiveModelName] =
    useState<string>("gemini-3.8-flash");

  // Selection state for Claim & Quote Grounding Inspector
  const [selectedClaimId, setSelectedClaimId] = useState<string>(
    FALLBACK_RUN.state.claims[0]?.id ?? "c1"
  );
  const [selectedSpanIdx, setSelectedSpanIdx] = useState<number>(0);

  // Selection state for GraphRAG Explorer
  const [selectedGraphNodeId, setSelectedGraphNodeId] = useState<string | null>(
    null
  );

  useEffect(() => {
    let cancelled = false;
    async function probeBackend() {
      try {
        const res = await fetch("/api/health");
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        setBackendOnline(true);
        if (data.llm_model) setLiveModelName(data.llm_model);

        const [presetsRes, runsRes, evalsRes] = await Promise.all([
          fetch("/api/presets"),
          fetch("/api/runs"),
          fetch("/api/evals"),
        ]);
        if (presetsRes.ok) {
          const p = await presetsRes.json();
          if (!cancelled && Array.isArray(p) && p.length > 0) setPresets(p);
        }
        if (runsRes.ok) {
          const rList = await runsRes.json();
          if (!cancelled && Array.isArray(rList) && rList.length > 0) {
            const detailRes = await fetch(`/api/runs/${rList[0].run_id}`);
            if (detailRes.ok) {
              const fullRun: RunPayload = await detailRes.json();
              if (!cancelled && fullRun.state?.draft) {
                setRun(fullRun);
                setSelectedClaimId(fullRun.state.claims[0]?.id ?? "c1");
              }
            }
          }
        }
        if (evalsRes.ok) {
          const evData = await evalsRes.json();
          if (
            !cancelled &&
            Array.isArray(evData.summaries) &&
            evData.summaries.length > 0
          ) {
            setEvals(evData.summaries);
            setSelectedEvalIdx(evData.summaries.length - 1);
          }
        }
      } catch {
        if (!cancelled) setBackendOnline(false);
      }
    }
    probeBackend();
    return () => {
      cancelled = true;
    };
  }, []);

  const executeResearch = useCallback(async () => {
    if (!questionInput.trim() || isRunning) return;
    setIsRunning(true);
    try {
      const res = await fetch("/api/runs/sync", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: questionInput.trim(),
          mode: "auto",
          retrieval_mode: retrievalMode,
          max_skeptic_rounds: maxSkepticRounds,
        }),
      });
      if (res.ok) {
        const payload: RunPayload = await res.json();
        setRun(payload);
        setSelectedClaimId(payload.state.claims[0]?.id ?? "c1");
        setSelectedSpanIdx(0);
        setBackendOnline(true);
        return;
      }
    } catch {
      // Offline fallback: update question title on embedded run so UI responds smoothly
      const updated: RunPayload = {
        ...FALLBACK_RUN,
        question: questionInput.trim(),
        state: {
          ...FALLBACK_RUN.state,
          question: questionInput.trim(),
        },
      };
      setRun(updated);
      setSelectedClaimId(updated.state.claims[0]?.id ?? "c1");
      setSelectedSpanIdx(0);
    } finally {
      setIsRunning(false);
    }
  }, [questionInput, retrievalMode, maxSkepticRounds, isRunning]);

  const selectedClaim: Claim | undefined = useMemo(
    () =>
      run.state.claims.find((c) => c.id === selectedClaimId) ??
      run.state.claims[0],
    [run.state.claims, selectedClaimId]
  );

  const activeSpan: EvidenceSpan | undefined = useMemo(
    () =>
      selectedClaim?.evidence[selectedSpanIdx] ?? selectedClaim?.evidence[0],
    [selectedClaim, selectedSpanIdx]
  );

  const activePaper: Paper | undefined = useMemo(() => {
    if (activeSpan && run.state.papers[activeSpan.paper_id]) {
      return run.state.papers[activeSpan.paper_id];
    }
    const firstKey = Object.keys(run.state.papers)[0];
    return firstKey ? run.state.papers[firstKey] : undefined;
  }, [activeSpan, run.state.papers]);

  // Bibliography numbering matching renderer.py
  const paperRefNumbers = useMemo(() => {
    const map: Record<string, number> = {};
    let count = 0;
    const ensure = (pid: string) => {
      if (!map[pid]) {
        count += 1;
        map[pid] = count;
      }
    };
    for (const sec of run.state.draft?.sections ?? []) {
      for (const s of sec.sentences) {
        for (const cid of s.citations) {
          const ch = run.state.chunks[cid];
          if (ch) ensure(ch.paper_id);
        }
      }
    }
    for (const c of run.state.claims) {
      for (const sp of c.evidence) ensure(sp.paper_id);
    }
    return map;
  }, [run.state]);

  // Node telemetry stats computed from run.events
  const nodeStats = useMemo(() => {
    const stats: Record<
      string,
      { visits: number; durationMs: number; lastMessage?: string }
    > = {};
    for (const n of PIPELINE_NODES) {
      stats[n.id] = { visits: 0, durationMs: 0 };
    }
    for (const ev of run.events) {
      if (!ev.node || !stats[ev.node]) continue;
      if (ev.type === "node_completed") {
        stats[ev.node].visits += 1;
        stats[ev.node].durationMs += Number(ev.data.duration_ms ?? 0);
      } else if (ev.type === "progress" && typeof ev.data.message === "string") {
        stats[ev.node].lastMessage = ev.data.message;
      }
    }
    return stats;
  }, [run.events]);

  // Layout coordinates for GraphRAG interactive SVG canvas
  const graphLayout = useMemo(() => {
    const kg = run.state.knowledge_graph;
    const papers = kg.nodes.filter((n) => n.kind === "paper");
    const entities = kg.nodes.filter((n) => n.kind === "entity");
    const coords: Record<string, { x: number; y: number }> = {};

    const cx = 420;
    const cy = 260;
    papers.forEach((p, idx) => {
      const angle = (2 * Math.PI * idx) / Math.max(1, papers.length) - Math.PI / 2;
      coords[p.id] = {
        x: cx + Math.cos(angle) * 115,
        y: cy + Math.sin(angle) * 105,
      };
    });
    entities.forEach((e, idx) => {
      const ring = idx % 2 === 0 ? 225 : 180;
      const angle =
        (2 * Math.PI * idx) / Math.max(1, entities.length) - Math.PI / 4;
      coords[e.id] = {
        x: cx + Math.cos(angle) * ring,
        y: cy + Math.sin(angle) * (ring * 0.88),
      };
    });
    return { coords, papers, entities };
  }, [run.state.knowledge_graph]);

  const connectedGraphNodeIds = useMemo(() => {
    if (!selectedGraphNodeId) return new Set<string>();
    const set = new Set<string>([selectedGraphNodeId]);
    for (const edge of run.state.knowledge_graph.edges) {
      if (edge.source === selectedGraphNodeId) set.add(edge.target);
      if (edge.target === selectedGraphNodeId) set.add(edge.source);
    }
    return set;
  }, [selectedGraphNodeId, run.state.knowledge_graph.edges]);

  const activeBenchmark = evals[selectedEvalIdx] ?? evals[evals.length - 1];

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100">
      {/* Top Header */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-[#070b14]/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-500/40 bg-gradient-to-br from-cyan-500/20 to-indigo-600/20 shadow-lg shadow-cyan-950/50">
              <ShieldCheck className="h-5 w-5 text-cyan-400" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <span className="text-lg font-bold tracking-tight text-white">
                  AETHERIS
                </span>
                <span className="rounded-md border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wider text-cyan-300">
                  v0.1 · Claim-Verifiable Research Engine
                </span>
              </div>
              <p className="text-xs text-slate-400">
                3-Way Hybrid + GraphRAG · 3-Pass Mechanical Quote Grounding ·
                Adversarial Skeptic Loop
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-1.5 text-xs">
              <span
                className={`h-2 w-2 rounded-full ${
                  backendOnline
                    ? "bg-emerald-400 shadow-[0_0_8px_#34d399]"
                    : "bg-cyan-400 shadow-[0_0_8px_#22d3ee]"
                }`}
              />
              <span className="text-slate-300">
                {backendOnline
                  ? `FastAPI Live (${liveModelName})`
                  : "Deterministic Showcase Engine"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-6 py-6">
        {/* Research Query & Control Console */}
        <section className="rounded-2xl border border-slate-800/90 bg-slate-900/50 p-5 shadow-xl">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end">
            <div className="flex-1 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <label
                  htmlFor="research-question-input"
                  className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-cyan-400"
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  Scientific Research Question
                </label>
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[11px] text-slate-400">Presets:</span>
                  {presets.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => setQuestionInput(p.question)}
                      className="rounded-md border border-slate-700/80 bg-slate-800/70 px-2 py-0.5 text-[11px] text-slate-300 transition hover:border-cyan-500/50 hover:text-cyan-200"
                    >
                      {p.title}
                    </button>
                  ))}
                </div>
              </div>
              <div className="relative">
                <Search className="pointer-events-none absolute left-3.5 top-3.5 h-4 w-4 text-slate-400" />
                <input
                  id="research-question-input"
                  type="text"
                  value={questionInput}
                  onChange={(e) => setQuestionInput(e.target.value)}
                  placeholder="Enter a scientific research question to decompose, retrieve, and verify..."
                  className="w-full rounded-xl border border-slate-700/80 bg-slate-950/90 py-2.5 pl-10 pr-4 text-sm text-slate-100 placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div>
                <label
                  htmlFor="retrieval-mode-select"
                  className="mb-1 block text-[11px] font-medium text-slate-400"
                >
                  Retrieval Mode
                </label>
                <select
                  id="retrieval-mode-select"
                  value={retrievalMode}
                  onChange={(e) =>
                    setRetrievalMode(e.target.value as RetrievalMode)
                  }
                  className="rounded-xl border border-slate-700/80 bg-slate-950 px-3 py-2.5 text-xs font-medium text-slate-200 focus:border-cyan-500 focus:outline-none"
                >
                  <option value="graph_hybrid">
                    3-Way GraphRAG + Hybrid RRF
                  </option>
                  <option value="hybrid">2-Way Hybrid (Dense + BM25)</option>
                  <option value="bm25">Sparse BM25+ Only</option>
                  <option value="dense">Dense Vector Only</option>
                </select>
              </div>

              <div>
                <label
                  htmlFor="skeptic-rounds-select"
                  className="mb-1 block text-[11px] font-medium text-slate-400"
                >
                  Max Skeptic Loops
                </label>
                <select
                  id="skeptic-rounds-select"
                  value={maxSkepticRounds}
                  onChange={(e) => setMaxSkepticRounds(Number(e.target.value))}
                  className="rounded-xl border border-slate-700/80 bg-slate-950 px-3 py-2.5 text-xs font-medium text-slate-200 focus:border-cyan-500 focus:outline-none"
                >
                  <option value={1}>1 Round (Default)</option>
                  <option value={2}>2 Rounds (Strict)</option>
                  <option value={0}>0 Rounds (Ablation)</option>
                </select>
              </div>

              <button
                type="button"
                onClick={executeResearch}
                disabled={isRunning}
                className="mt-4 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-cyan-950/60 transition hover:from-cyan-400 hover:to-indigo-500 disabled:opacity-50 lg:mt-5"
              >
                {isRunning ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Verifying Claims...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 fill-current" />
                    Run Verifiable Research
                  </>
                )}
              </button>
            </div>
          </div>

          {/* High-Level Run KPI Bar */}
          <div className="mt-5 grid grid-cols-2 gap-3 border-t border-slate-800/80 pt-4 sm:grid-cols-3 lg:grid-cols-6">
            <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-3">
              <div className="text-[11px] font-medium text-slate-400">
                Supported Claims
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-xl font-bold text-emerald-400">
                  {run.metrics.supported}/{run.metrics.total_claims}
                </span>
                <span className="text-xs font-semibold text-emerald-300/80">
                  ({Math.round(run.metrics.support_rate * 100)}%)
                </span>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-3">
              <div className="text-[11px] font-medium text-slate-400">
                Citation Accuracy
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-xl font-bold text-cyan-400">
                  {Math.round(run.metrics.citation_accuracy * 100)}%
                </span>
                <span className="text-xs text-slate-400">decisive span</span>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-3">
              <div className="text-[11px] font-medium text-slate-400">
                Literature Indexed
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-xl font-bold text-white">
                  {Object.keys(run.state.papers).length}
                </span>
                <span className="text-xs text-slate-400">
                  papers ({Object.keys(run.state.chunks).length} chunks)
                </span>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-3">
              <div className="text-[11px] font-medium text-slate-400">
                Knowledge Graph
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-xl font-bold text-indigo-400">
                  {run.metrics.graph_nodes}
                </span>
                <span className="text-xs text-slate-400">
                  nodes · {run.metrics.graph_edges} edges
                </span>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-3">
              <div className="text-[11px] font-medium text-slate-400">
                Skeptic Audit Loops
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-xl font-bold text-amber-300">
                  {run.metrics.skeptic_rounds}
                </span>
                <span className="text-xs text-slate-400">
                  self-critique passes
                </span>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-3">
              <div className="text-[11px] font-medium text-slate-400">
                Run Telemetry
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-xl font-bold text-slate-200">
                  {run.llm_calls}
                </span>
                <span className="text-xs text-slate-400">
                  LLM calls · {run.total_tokens.toLocaleString()} tok
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* Navigation Tabs */}
        <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-3">
          {(
            [
              {
                id: "Inspector",
                label: "Claim & Quote Grounding Inspector",
                icon: FileSearch,
              },
              {
                id: "Pipeline",
                label: "Agent Execution DAG & Skeptic Loop",
                icon: GitBranch,
              },
              {
                id: "GraphRAG",
                label: "GraphRAG Knowledge Graph Explorer",
                icon: Network,
              },
              {
                id: "Benchmark",
                label: "SciFact-Aetheris Ablation Suite",
                icon: FlaskConical,
              },
            ] as const
          ).map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-xs font-semibold transition ${
                  active
                    ? "border border-cyan-500/50 bg-cyan-500/15 text-cyan-200 shadow-sm"
                    : "border border-transparent bg-slate-900/60 text-slate-400 hover:border-slate-800 hover:text-slate-200"
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* TAB 1: CLAIM & QUOTE GROUNDING INSPECTOR */}
        {activeTab === "Inspector" && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            {/* Left 7 columns: Annotated Synthesis Report + Atomic Claims */}
            <div className="space-y-5 lg:col-span-7">
              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-4">
                  <div>
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400">
                      Synthesized & Verified Research Brief
                    </span>
                    <h2 className="mt-1 text-lg font-bold text-white">
                      {run.state.draft?.title ?? "Research Report"}
                    </h2>
                  </div>
                  <span className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-300">
                    Click any claim to highlight exact source span →
                  </span>
                </div>

                <div className="mt-5 space-y-6">
                  {run.state.draft?.sections.map((section, sIdx) => (
                    <div key={sIdx} className="space-y-3">
                      <h3 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-slate-300">
                        <Layers className="h-4 w-4 text-cyan-400" />
                        {section.heading}
                      </h3>

                      <div className="space-y-3">
                        {section.sentences.map((sentence) => {
                          const sClaims = run.state.claims.filter(
                            (c) => c.sentence_id === sentence.id
                          );
                          const refNums = Array.from(
                            new Set(
                              sentence.citations
                                .map((cid) => run.state.chunks[cid]?.paper_id)
                                .filter((pid): pid is string => Boolean(pid))
                                .map((pid) => paperRefNumbers[pid])
                            )
                          ).sort((a, b) => a - b);

                          return (
                            <div
                              key={sentence.id}
                              className="rounded-xl border border-slate-800/90 bg-slate-950/70 p-4 transition hover:border-slate-700"
                            >
                              <div className="flex items-start gap-2.5">
                                <span className="mt-0.5 rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-cyan-300">
                                  {sentence.id}
                                </span>
                                <p className="flex-1 text-sm leading-relaxed text-slate-200">
                                  {sentence.text}{" "}
                                  {refNums.map((num) => (
                                    <span
                                      key={num}
                                      className="ml-1 inline-flex items-center rounded bg-cyan-500/15 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-cyan-300"
                                    >
                                      [{num}]
                                    </span>
                                  ))}
                                </p>
                              </div>

                              {/* Atomic Claims for this Sentence */}
                              {sClaims.length > 0 && (
                                <div className="mt-3 space-y-2 border-t border-slate-800/80 pt-3">
                                  {sClaims.map((claim) => {
                                    const isSelected =
                                      selectedClaim?.id === claim.id;
                                    return (
                                      <button
                                        key={claim.id}
                                        type="button"
                                        onClick={() => {
                                          setSelectedClaimId(claim.id);
                                          setSelectedSpanIdx(0);
                                        }}
                                        className={`w-full rounded-xl border p-3 text-left transition ${
                                          isSelected
                                            ? "border-cyan-500/70 bg-cyan-950/25 shadow-md shadow-cyan-950/40"
                                            : "border-slate-800/70 bg-slate-900/60 hover:border-slate-700"
                                        }`}
                                      >
                                        <div className="flex flex-wrap items-center justify-between gap-2">
                                          <div className="flex items-center gap-2">
                                            <span className="font-mono text-xs font-bold text-slate-400">
                                              Claim {claim.id}
                                            </span>
                                            {verdictBadge(claim.verdict)}
                                            {claim.evidence[0] && (
                                              <span className="rounded border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 font-mono text-[10px] uppercase text-indigo-300">
                                                match: {claim.evidence[0].match}
                                              </span>
                                            )}
                                          </div>
                                          {claim.confidence !== null && (
                                            <div className="flex items-center gap-2">
                                              <span className="text-[11px] text-slate-400">
                                                Confidence
                                              </span>
                                              <span className="font-mono text-xs font-bold text-cyan-300">
                                                {Math.round(
                                                  claim.confidence * 100
                                                )}
                                                %
                                              </span>
                                            </div>
                                          )}
                                        </div>
                                        <p className="mt-1.5 text-xs text-slate-300">
                                          {claim.text}
                                        </p>
                                        {claim.skeptic_notes && (
                                          <div className="mt-2 flex items-start gap-1.5 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2 text-[11px] text-amber-200">
                                            <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" />
                                            <span>
                                              <strong>Skeptic Audit:</strong>{" "}
                                              {claim.skeptic_notes}
                                            </span>
                                          </div>
                                        )}
                                      </button>
                                    );
                                  })}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right 5 columns: Mechanical Quote Grounding & Abstract Offset Inspector */}
            <div className="space-y-5 lg:col-span-5">
              <div className="sticky top-24 space-y-5 rounded-2xl border border-slate-800/90 bg-slate-900/50 p-6">
                <div className="border-b border-slate-800 pb-4">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400">
                      3-Pass Mechanical Quote Grounding Inspector
                    </span>
                    {selectedClaim && verdictBadge(selectedClaim.verdict)}
                  </div>
                  <h3 className="mt-1 text-base font-bold text-white">
                    {selectedClaim
                      ? `Atomic Claim ${selectedClaim.id} (${selectedClaim.sentence_id})`
                      : "Select a claim"}
                  </h3>
                  {selectedClaim && (
                    <p className="mt-1 text-xs leading-relaxed text-slate-300">
                      “{selectedClaim.text}”
                    </p>
                  )}
                </div>

                {selectedClaim && (
                  <>
                    {/* Verifier Rationale & Span Selector */}
                    <div className="space-y-2 rounded-xl border border-slate-800 bg-slate-950/70 p-3.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-slate-300">
                          Verifier Entailment Rationale
                        </span>
                        <span className="font-mono text-cyan-300">
                          {selectedClaim.decisive_evidence_cited
                            ? "✓ Cited Source Verified"
                            : "Augmented Search"}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400">
                        {selectedClaim.rationale ??
                          "Verified via mechanical substring alignment."}
                      </p>

                      {selectedClaim.evidence.length > 1 && (
                        <div className="flex flex-wrap gap-1.5 pt-2">
                          {selectedClaim.evidence.map((sp, idx) => (
                            <button
                              key={`${sp.chunk_id}-${idx}`}
                              type="button"
                              onClick={() => setSelectedSpanIdx(idx)}
                              className={`rounded-md px-2.5 py-1 font-mono text-[11px] ${
                                idx === selectedSpanIdx
                                  ? "bg-cyan-500 text-slate-950 font-bold"
                                  : "bg-slate-800 text-slate-300"
                              }`}
                            >
                              Span #{idx + 1} ({sp.paper_id})
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Source Paper Metadata & Character-Exact Highlighted Abstract */}
                    {activePaper && (
                      <div className="space-y-3 rounded-xl border border-slate-800 bg-slate-950/90 p-4">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="rounded bg-indigo-500/20 px-2 py-0.5 font-mono text-[11px] font-semibold text-indigo-300">
                                [{paperRefNumbers[activePaper.id] ?? "?"}]{" "}
                                {activePaper.id}
                              </span>
                              {activePaper.venue && (
                                <span className="text-xs font-medium text-slate-400">
                                  {activePaper.venue} · {activePaper.year}
                                </span>
                              )}
                            </div>
                            <h4 className="mt-1.5 text-sm font-bold text-white">
                              {activePaper.title}
                            </h4>
                            <p className="mt-0.5 text-xs text-slate-400">
                              {activePaper.authors.join(", ")}
                              {activePaper.citation_count
                                ? ` · ${activePaper.citation_count.toLocaleString()} citations`
                                : ""}
                            </p>
                          </div>
                          <a
                            href={activePaper.url}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/80 px-2.5 py-1 text-xs text-cyan-300 hover:border-cyan-500/50"
                          >
                            Source
                            <ExternalLink className="h-3 w-3" />
                          </a>
                        </div>

                        {activeSpan && (
                          <div className="flex flex-wrap items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-950/20 px-3 py-1.5 font-mono text-[11px] text-emerald-300">
                            <span>Chunk: {activeSpan.chunk_id}</span>
                            <span>·</span>
                            <span>
                              Offsets: chars {activeSpan.char_start}–
                              {activeSpan.char_end}
                            </span>
                            <span>·</span>
                            <span className="uppercase">
                              Pass: {activeSpan.match}
                            </span>
                          </div>
                        )}

                        <div className="space-y-1.5">
                          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                            Source Abstract with Highlighted Evidence Span
                          </div>
                          <div className="max-h-72 overflow-y-auto rounded-lg border border-slate-800/90 bg-[#050811] p-3.5 text-xs leading-relaxed text-slate-300">
                            {activeSpan &&
                            activeSpan.paper_id === activePaper.id &&
                            activeSpan.char_end > activeSpan.char_start ? (
                              <>
                                <span>
                                  {activePaper.abstract.slice(
                                    0,
                                    activeSpan.char_start
                                  )}
                                </span>
                                <mark className="rounded bg-emerald-400/25 px-1 py-0.5 font-medium text-emerald-200 ring-1 ring-emerald-400/50">
                                  {activePaper.abstract.slice(
                                    activeSpan.char_start,
                                    activeSpan.char_end
                                  )}
                                </mark>
                                <span>
                                  {activePaper.abstract.slice(
                                    activeSpan.char_end
                                  )}
                                </span>
                              </>
                            ) : (
                              <span>{activePaper.abstract}</span>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: AGENT EXECUTION DAG & SKEPTIC LOOP */}
        {activeTab === "Pipeline" && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            {/* Left 7 columns: Visual State-Machine DAG & Skeptic Audit History */}
            <div className="space-y-6 lg:col-span-7">
              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6">
                <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                  <div>
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400">
                      Deterministic Async State Machine
                    </span>
                    <h2 className="mt-1 text-lg font-bold text-white">
                      Multi-Agent Execution Graph & Conditional Skeptic Loop
                    </h2>
                  </div>
                  <span className="rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1 font-mono text-xs text-slate-300">
                    Run ID: {run.run_id}
                  </span>
                </div>

                {/* Pipeline Nodes Flow */}
                <div className="mt-5 space-y-3">
                  {PIPELINE_NODES.map((node, index) => {
                    const st = nodeStats[node.id] ?? {
                      visits: 0,
                      durationMs: 0,
                    };
                    const executed = st.visits > 0;
                    const looped = st.visits > 1;
                    return (
                      <div key={node.id} className="space-y-2">
                        <div
                          className={`flex items-center justify-between rounded-xl border p-4 transition ${
                            executed
                              ? "border-cyan-500/40 bg-slate-950/80"
                              : "border-slate-800 bg-slate-950/40 opacity-60"
                          }`}
                        >
                          <div className="flex items-center gap-3.5">
                            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-cyan-500/40 bg-cyan-500/10 font-mono text-xs font-bold text-cyan-300">
                              0{index + 1}
                            </div>
                            <div>
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-bold text-white">
                                  {node.label}
                                </span>
                                <span className="font-mono text-[11px] text-slate-400">
                                  ({node.id})
                                </span>
                                {looped && (
                                  <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold text-amber-300">
                                    {st.visits}x Passes (Skeptic Revision Loop)
                                  </span>
                                )}
                              </div>
                              <p className="text-xs text-slate-400">
                                {st.lastMessage ?? node.subtitle}
                              </p>
                            </div>
                          </div>

                          <div className="text-right font-mono text-xs">
                            <div className="text-emerald-400">
                              {executed ? `${st.visits} pass(es)` : "pending"}
                            </div>
                            <div className="text-[11px] text-slate-400">
                              {st.durationMs} ms
                            </div>
                          </div>
                        </div>

                        {index < PIPELINE_NODES.length - 1 && (
                          <div className="flex items-center justify-center">
                            <ChevronRight className="h-4 w-4 rotate-90 text-slate-600" />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Adversarial Skeptic Audit Rounds */}
              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6">
                <h3 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-amber-300">
                  <ShieldAlert className="h-4 w-4" />
                  Adversarial Skeptic (Devil’s Advocate) Audit Log
                </h3>
                <div className="mt-4 space-y-3">
                  {run.state.skeptic_history.map((crit) => (
                    <div
                      key={crit.round}
                      className="rounded-xl border border-slate-800 bg-slate-950/80 p-4"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs font-bold text-white">
                          Skeptic Audit Round #{crit.round}
                        </span>
                        <span
                          className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                            crit.requires_revision
                              ? "border border-amber-500/40 bg-amber-500/15 text-amber-300"
                              : "border border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
                          }`}
                        >
                          {crit.requires_revision
                            ? "↻ Triggered Re-Retrieval & Draft Revision"
                            : "✓ Verification Gate Passed"}
                        </span>
                      </div>
                      <p className="mt-2 text-xs text-slate-300">
                        {crit.summary}
                      </p>
                      {crit.issues.length > 0 && (
                        <div className="mt-3 space-y-1.5 border-t border-slate-800/80 pt-2.5">
                          {crit.issues.map((iss, i) => (
                            <div
                              key={i}
                              className="text-xs text-slate-400"
                            >
                              <span className="font-mono font-semibold text-rose-300">
                                [{iss.claim_id}]
                              </span>{" "}
                              {iss.issue} →{" "}
                              <span className="font-mono text-cyan-300">
                                query: &ldquo;{iss.counter_query}&rdquo;
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right 5 columns: Sub-Question Decomposition & Event Stream */}
            <div className="space-y-6 lg:col-span-5">
              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6">
                <h3 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-cyan-400">
                  <Cpu className="h-4 w-4" />
                  Planner Hypothesis & Query Decomposition
                </h3>
                <p className="mt-1 text-xs text-slate-400">
                  {run.state.plan?.objective}
                </p>
                <div className="mt-4 space-y-3">
                  {run.state.plan?.sub_questions.map((sq, idx) => (
                    <div
                      key={idx}
                      className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5"
                    >
                      <div className="text-xs font-bold text-white">
                        {idx + 1}. {sq.question}
                      </div>
                      <p className="mt-1 text-[11px] text-slate-400">
                        {sq.rationale}
                      </p>
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {sq.search_queries.map((q, qIdx) => (
                          <span
                            key={qIdx}
                            className="rounded border border-slate-700 bg-slate-900 px-2 py-0.5 font-mono text-[10px] text-cyan-300"
                          >
                            {q}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6">
                <h3 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-300">
                  <Terminal className="h-4 w-4 text-cyan-400" />
                  SQLite Event-Sourced Trace Timeline ({run.events.length}{" "}
                  events)
                </h3>
                <div className="mt-4 max-h-96 space-y-2 overflow-y-auto rounded-xl border border-slate-800 bg-[#050811] p-3.5 font-mono text-[11px]">
                  {run.events.map((ev) => (
                    <div
                      key={ev.seq}
                      className="flex items-start gap-2 border-b border-slate-900 pb-1.5 text-slate-300"
                    >
                      <span className="text-slate-500">
                        #{String(ev.seq).padStart(2, "0")}
                      </span>
                      <span className="min-w-[115px] font-semibold text-cyan-300">
                        {ev.type}
                      </span>
                      {ev.node && (
                        <span className="rounded bg-slate-800 px-1.5 text-indigo-300">
                          {ev.node}
                        </span>
                      )}
                      <span className="flex-1 break-all text-slate-400">
                        {typeof ev.data.message === "string"
                          ? ev.data.message
                          : JSON.stringify(ev.data)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: GRAPHRAG KNOWLEDGE GRAPH EXPLORER */}
        {activeTab === "GraphRAG" && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            {/* Left 8 columns: Interactive SVG Bipartite + Co-Occurrence Graph */}
            <div className="rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6 lg:col-span-8">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-4">
                <div>
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400">
                    Personalized PageRank (PPR) Heterogeneous Graph
                  </span>
                  <h2 className="mt-1 text-lg font-bold text-white">
                    Entity-Relation & Literature Knowledge Graph
                  </h2>
                </div>
                <div className="flex items-center gap-4 text-xs">
                  <span className="flex items-center gap-1.5 text-cyan-300">
                    <span className="h-3 w-3 rounded-full bg-cyan-400" />
                    Source Papers ({graphLayout.papers.length})
                  </span>
                  <span className="flex items-center gap-1.5 text-indigo-300">
                    <span className="h-3 w-3 rounded-full bg-indigo-400" />
                    Scientific Concepts ({graphLayout.entities.length})
                  </span>
                  {selectedGraphNodeId && (
                    <button
                      type="button"
                      onClick={() => setSelectedGraphNodeId(null)}
                      className="rounded border border-slate-700 bg-slate-800 px-2 py-0.5 text-[11px] text-slate-300 hover:text-white"
                    >
                      Reset Filter
                    </button>
                  )}
                </div>
              </div>

              <div className="mt-4 overflow-hidden rounded-xl border border-slate-800 bg-[#050811]">
                <svg
                  viewBox="0 0 840 520"
                  className="h-[460px] w-full select-none"
                >
                  {/* Edges */}
                  {run.state.knowledge_graph.edges.map((edge, idx) => {
                    const s = graphLayout.coords[edge.source];
                    const t = graphLayout.coords[edge.target];
                    if (!s || !t) return null;
                    const highlighted =
                      !selectedGraphNodeId ||
                      edge.source === selectedGraphNodeId ||
                      edge.target === selectedGraphNodeId;
                    return (
                      <line
                        key={idx}
                        x1={s.x}
                        y1={s.y}
                        x2={t.x}
                        y2={t.y}
                        stroke={
                          edge.relation === "co_occurs" ? "#818cf8" : "#22d3ee"
                        }
                        strokeOpacity={highlighted ? 0.45 : 0.07}
                        strokeWidth={
                          highlighted
                            ? Math.min(2.5, 0.9 + edge.weight * 0.3)
                            : 0.7
                        }
                      />
                    );
                  })}

                  {/* Nodes */}
                  {run.state.knowledge_graph.nodes.map((node) => {
                    const pos = graphLayout.coords[node.id];
                    if (!pos) return null;
                    const isPaper = node.kind === "paper";
                    const active =
                      !selectedGraphNodeId ||
                      connectedGraphNodeIds.has(node.id);
                    const isCenter = selectedGraphNodeId === node.id;
                    const radius = isPaper
                      ? 12
                      : Math.min(10, Math.max(5.5, node.weight * 3));
                    return (
                      <g
                        key={node.id}
                        transform={`translate(${pos.x}, ${pos.y})`}
                        onClick={() =>
                          setSelectedGraphNodeId(
                            isCenter ? null : node.id
                          )
                        }
                        className="cursor-pointer transition-opacity"
                        style={{ opacity: active ? 1 : 0.2 }}
                      >
                        <circle
                          r={isCenter ? radius + 4 : radius}
                          fill={isPaper ? "#06b6d4" : "#6366f1"}
                          fillOpacity={isPaper ? 0.9 : 0.8}
                          stroke={isCenter ? "#ffffff" : "#0f172a"}
                          strokeWidth={isCenter ? 2.5 : 1.5}
                        />
                        <text
                          y={radius + 12}
                          textAnchor="middle"
                          fill={isPaper ? "#67e8f9" : "#cbd5e1"}
                          fontSize={isPaper ? "10" : "9"}
                          fontWeight={isPaper || isCenter ? "700" : "500"}
                        >
                          {node.label.slice(0, 22)}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>

            {/* Right 4 columns: Node Inspector & Multi-Hop Bridge Explanation */}
            <div className="space-y-5 lg:col-span-4">
              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6">
                <h3 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-cyan-400">
                  <Database className="h-4 w-4" />
                  How 3-Way GraphRAG Fusion Works
                </h3>
                <p className="mt-2 text-xs leading-relaxed text-slate-300">
                  During ingestion, Aetheris extracts technical concepts,
                  acronyms, and hyphenated compounds from every abstract chunk
                  and builds a weighted graph of{" "}
                  <code className="text-cyan-300">paper → chunk → entity</code>{" "}
                  and sentence-level{" "}
                  <code className="text-indigo-300">entity ↔ entity</code>{" "}
                  co-occurrences. At query time,{" "}
                  <strong>Personalized PageRank (α=0.82)</strong> diffuses
                  probability mass across multi-hop concepts and fuses the graph
                  ranking with Dense Vectors and BM25+ via Reciprocal Rank
                  Fusion.
                </p>

                <div className="mt-4 border-t border-slate-800 pt-4">
                  <div className="text-xs font-bold uppercase tracking-wider text-slate-400">
                    {selectedGraphNodeId
                      ? `Selected Node: ${selectedGraphNodeId}`
                      : "Top Extracted Scientific Entities"}
                  </div>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {graphLayout.entities.slice(0, 24).map((ent) => (
                      <button
                        key={ent.id}
                        type="button"
                        onClick={() =>
                          setSelectedGraphNodeId(
                            selectedGraphNodeId === ent.id ? null : ent.id
                          )
                        }
                        className={`rounded-lg border px-2.5 py-1 font-mono text-[11px] transition ${
                          selectedGraphNodeId === ent.id
                            ? "border-cyan-400 bg-cyan-500/20 text-white"
                            : "border-slate-800 bg-slate-950/80 text-slate-300 hover:border-indigo-500/50"
                        }`}
                      >
                        {ent.label}{" "}
                        <span className="text-slate-500">({ent.weight})</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6">
                <h3 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-300">
                  <BookOpen className="h-4 w-4 text-cyan-400" />
                  Retrieved Corpus ({Object.keys(run.state.papers).length}{" "}
                  Papers)
                </h3>
                <div className="mt-3 max-h-72 space-y-2.5 overflow-y-auto pr-1">
                  {Object.values(run.state.papers).map((p) => (
                    <div
                      key={p.id}
                      className="rounded-xl border border-slate-800 bg-slate-950/70 p-3"
                    >
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="font-mono font-semibold text-cyan-300">
                          {p.id}
                        </span>
                        <span className="text-slate-400">
                          {p.venue} · {p.year}
                        </span>
                      </div>
                      <div className="mt-1 text-xs font-semibold text-white">
                        {p.title}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: SCIFACT-AETHERIS EVALUATION & ABLATION SUITE */}
        {activeTab === "Benchmark" && activeBenchmark && (
          <div className="space-y-6">
            {/* 5-Way Ablation Comparison Matrix */}
            <div className="rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6">
              <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
                <div>
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400">
                    Reproducible Scientific Benchmark (`aetheris eval`)
                  </span>
                  <h2 className="mt-1 text-lg font-bold text-white">
                    5-Way Retrieval & Mechanical Grounding Ablation Study
                    (SciFact-Aetheris)
                  </h2>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <Activity className="h-4 w-4 text-emerald-400" />
                  Click any row to inspect its calibration curve & per-claim
                  predictions
                </div>
              </div>

              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-[11px] uppercase tracking-wider text-slate-400">
                      <th className="py-3 pr-4">Configuration</th>
                      <th className="px-3 py-3">Retrieval</th>
                      <th className="px-3 py-3 text-center">Quote Guard</th>
                      <th className="px-3 py-3 text-right">Verdict Acc</th>
                      <th className="px-3 py-3 text-right">Macro-F1</th>
                      <th className="px-3 py-3 text-right">Recall@K</th>
                      <th className="px-3 py-3 text-right">MRR</th>
                      <th className="px-3 py-3 text-right">Mean Span IoU</th>
                      <th className="px-3 py-3 text-right">ECE (↓)</th>
                      <th className="px-3 py-3 text-right">Brier (↓)</th>
                      <th className="pl-3 py-3 text-right">Traps Caught</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {evals.map((s, idx) => {
                      const isSel = idx === selectedEvalIdx;
                      const isFull = s.config_name === "full_aetheris";
                      return (
                        <tr
                          key={s.config_name}
                          onClick={() => setSelectedEvalIdx(idx)}
                          className={`cursor-pointer transition ${
                            isSel
                              ? "bg-cyan-500/15 text-white"
                              : "hover:bg-slate-900/80 text-slate-300"
                          }`}
                        >
                          <td className="py-3.5 pr-4">
                            <div className="flex items-center gap-2 font-semibold">
                              {isFull && (
                                <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-bold uppercase text-emerald-300">
                                  Best
                                </span>
                              )}
                              <span>{s.config_name}</span>
                            </div>
                            <div className="text-[11px] text-slate-400">
                              {s.description}
                            </div>
                          </td>
                          <td className="px-3 py-3.5 font-mono text-cyan-300">
                            {s.retrieval_mode}
                          </td>
                          <td className="px-3 py-3.5 text-center">
                            {s.grounding_guard ? (
                              <span className="rounded bg-emerald-500/15 px-2 py-0.5 font-mono text-[11px] text-emerald-300">
                                ON
                              </span>
                            ) : (
                              <span className="rounded bg-rose-500/15 px-2 py-0.5 font-mono text-[11px] text-rose-300">
                                OFF
                              </span>
                            )}
                          </td>
                          <td className="px-3 py-3.5 text-right font-mono font-semibold">
                            {(s.verdict_accuracy * 100).toFixed(1)}%
                          </td>
                          <td className="px-3 py-3.5 text-right font-mono font-bold text-emerald-400">
                            {(s.verdict_macro_f1 * 100).toFixed(1)}%
                          </td>
                          <td className="px-3 py-3.5 text-right font-mono">
                            {(s.retrieval_recall_at_k * 100).toFixed(1)}%
                          </td>
                          <td className="px-3 py-3.5 text-right font-mono">
                            {s.retrieval_mrr.toFixed(3)}
                          </td>
                          <td className="px-3 py-3.5 text-right font-mono text-indigo-300">
                            {(s.mean_span_iou * 100).toFixed(1)}%
                          </td>
                          <td className="px-3 py-3.5 text-right font-mono">
                            {s.ece.toFixed(3)}
                          </td>
                          <td className="px-3 py-3.5 text-right font-mono">
                            {s.brier_score.toFixed(3)}
                          </td>
                          <td className="pl-3 py-3.5 text-right font-mono font-bold text-amber-300">
                            {s.hallucinations_caught}/2
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Per-Claim Predictions for Selected Ablation Config */}
            <div className="rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-4">
                <div>
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400">
                    Per-Example Diagnostic Matrix
                  </span>
                  <h3 className="mt-1 text-base font-bold text-white">
                    {activeBenchmark.config_name} — {activeBenchmark.description}
                  </h3>
                </div>
                <div className="flex gap-3 font-mono text-xs">
                  <span className="text-emerald-300">
                    Macro-F1: {(activeBenchmark.verdict_macro_f1 * 100).toFixed(1)}%
                  </span>
                  <span className="text-cyan-300">
                    Span IoU: {(activeBenchmark.mean_span_iou * 100).toFixed(1)}%
                  </span>
                  <span className="text-indigo-300">
                    Brier: {activeBenchmark.brier_score.toFixed(3)}
                  </span>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
                {activeBenchmark.predictions.map((pred) => (
                  <div
                    key={pred.example_id}
                    className={`rounded-xl border p-3.5 ${
                      pred.correct
                        ? "border-slate-800 bg-slate-950/70"
                        : "border-rose-500/40 bg-rose-950/15"
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-cyan-300">
                          {pred.example_id}
                        </span>
                        <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] text-slate-300">
                          {pred.category}
                        </span>
                        {pred.multi_hop && (
                          <span className="rounded border border-indigo-500/40 bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-300">
                            Multi-Hop
                          </span>
                        )}
                      </div>
                      <span
                        className={`font-mono text-xs font-bold ${
                          pred.correct ? "text-emerald-400" : "text-rose-400"
                        }`}
                      >
                        {pred.correct ? "✓ MATCH" : "✗ MISMATCH"}
                      </span>
                    </div>

                    <p className="mt-2 text-xs text-slate-200">{pred.claim}</p>

                    <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 border-t border-slate-800/80 pt-2 font-mono text-[11px] text-slate-400">
                      <span>
                        Gold: <strong className="text-slate-200">{pred.gold_verdict}</strong> · Pred:{" "}
                        <strong className="text-cyan-300">{pred.pred_verdict}</strong>
                      </span>
                      <span>
                        Span IoU: {(pred.span_iou * 100).toFixed(0)}% · Conf:{" "}
                        {(pred.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
