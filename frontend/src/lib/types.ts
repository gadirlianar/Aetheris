export type Verdict = "supported" | "refuted" | "not_enough_info";
export type MatchKind = "exact" | "normalized" | "fuzzy";
export type RetrievalMode = "dense" | "bm25" | "hybrid" | "graph_hybrid";

export interface Paper {
  id: string;
  source: "arxiv" | "openalex" | "semantic_scholar";
  title: string;
  abstract: string;
  authors: string[];
  year: number | null;
  venue: string | null;
  doi: string | null;
  url: string;
  citation_count: number | null;
}

export interface Chunk {
  id: string;
  paper_id: string;
  text: string;
  char_start: number;
  char_end: number;
}

export interface GraphNode {
  id: string;
  label: string;
  kind: "paper" | "chunk" | "entity";
  weight: number;
  paper_id?: string | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  weight: number;
}

export interface KnowledgeGraphSnapshot {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface SubQuestion {
  question: string;
  rationale: string;
  search_queries: string[];
}

export interface ResearchPlan {
  objective: string;
  sub_questions: SubQuestion[];
}

export interface ReportSentence {
  id: string;
  text: string;
  citations: string[];
  invalid_citations: string[];
}

export interface ReportSection {
  heading: string;
  sentences: ReportSentence[];
}

export interface DraftReport {
  title: string;
  sections: ReportSection[];
}

export interface EvidenceSpan {
  chunk_id: string;
  paper_id: string;
  quote: string;
  char_start: number;
  char_end: number;
  match: MatchKind;
}

export interface Claim {
  id: string;
  sentence_id: string;
  text: string;
  verdict: Verdict | null;
  confidence: number | null;
  rationale: string | null;
  evidence: EvidenceSpan[];
  considered_chunk_ids: string[];
  grounding_failed: boolean;
  decisive_evidence_cited: boolean | null;
  skeptic_notes: string | null;
}

export interface SkepticIssue {
  claim_id: string;
  issue: string;
  counter_query: string;
}

export interface SkepticCritique {
  round: number;
  summary: string;
  requires_revision: boolean;
  issues: SkepticIssue[];
  counter_queries: string[];
}

export interface ResearchState {
  run_id: string;
  question: string;
  plan: ResearchPlan | null;
  papers: Record<string, Paper>;
  chunks: Record<string, Chunk>;
  knowledge_graph: KnowledgeGraphSnapshot;
  executed_queries: string[];
  pending_queries: string[];
  draft: DraftReport | null;
  claims: Claim[];
  skeptic_rounds: number;
  skeptic_history: SkepticCritique[];
  report_markdown: string | null;
}

export interface VerificationMetrics {
  total_claims: number;
  supported: number;
  refuted: number;
  not_enough_info: number;
  support_rate: number;
  citation_accuracy: number;
  grounding_failures: number;
  invalid_citations: number;
  skeptic_rounds: number;
  graph_nodes: number;
  graph_edges: number;
}

export interface RunEvent {
  run_id: string;
  seq: number;
  ts: string;
  type: string;
  node: string | null;
  data: Record<string, unknown>;
}

export interface RunPayload {
  run_id: string;
  question: string;
  status: string;
  created_at?: string;
  finished_at?: string;
  llm_calls: number;
  total_tokens: number;
  duration_ms: number;
  error?: string | null;
  metrics: VerificationMetrics;
  state: ResearchState;
  events: RunEvent[];
}

export interface PresetQuestion {
  id: string;
  title: string;
  question: string;
  domain: string;
}

export interface CalibrationBin {
  bin_lower: number;
  bin_upper: number;
  avg_confidence: number;
  accuracy: number;
  count: number;
}

export interface ExamplePrediction {
  example_id: string;
  claim: string;
  category: string;
  multi_hop: boolean;
  gold_verdict: Verdict;
  pred_verdict: Verdict;
  confidence: number;
  gold_paper_id: string | null;
  retrieved_paper_ids: string[];
  grounded_paper_id: string | null;
  span_iou: number;
  grounding_failed: boolean;
  correct: boolean;
}

export interface BenchmarkSummary {
  config_name: string;
  description: string;
  retrieval_mode: string;
  grounding_guard: boolean;
  total_examples: number;
  verdict_accuracy: number;
  verdict_macro_f1: number;
  per_class: Record<
    string,
    { precision: number; recall: number; f1: number; support: number }
  >;
  retrieval_recall_at_k: number;
  retrieval_mrr: number;
  mean_span_iou: number;
  brier_score: number;
  ece: number;
  hallucinations_caught: number;
  calibration_bins: CalibrationBin[];
  predictions: ExamplePrediction[];
}

