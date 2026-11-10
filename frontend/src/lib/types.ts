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

export interface WeightedEvidenceSource {
  paper_id: string;
  citation_label: string;
  stance: "supports" | "refutes" | "qualifies";
  year: number | null;
  citation_count: number | null;
  venue: string | null;
  authority_score: number;
  quote: string;
}

export interface ConflictResolution {
  id: string;
  topic: string;
  claim_ids: string[];
  supporting_sources: WeightedEvidenceSource[];
  counter_sources: WeightedEvidenceSource[];
  support_authority_weight: number;
  counter_authority_weight: number;
  recency_gap_years: number;
  citation_ratio: number;
  consensus_verdict:
    | "support_dominant"
    | "refute_dominant"
    | "qualified_consensus";
  synthesis: string;
}

export interface LatencyPercentiles {
  count: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  mean_ms: number;
  max_ms?: number;
}

export interface TierUsageSummary {
  calls: number;
  input_tokens?: number;
  output_tokens?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens: number;
  cost_usd: number;
}

export interface CostTelemetry {
  llm_calls: number;
  input_tokens?: number;
  output_tokens?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens: number;
  total_cost_usd: number;
  elapsed_s?: number;
  llm_latency: LatencyPercentiles;
  node_latencies: Record<string, LatencyPercentiles>;
  tier_usage: Record<string, TierUsageSummary>;
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
  conflicts?: ConflictResolution[];
  cost_telemetry?: CostTelemetry;
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
  conflicts_resolved?: number;
  total_cost_usd?: number;
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
  snapshot?: Record<string, unknown> | null;
}

export interface RunPayload {
  run_id: string;
  question: string;
  status: string;
  created_at?: string;
  finished_at?: string;
  llm_calls: number;
  total_tokens: number;
  cost_usd?: number;
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
  difficulty?: string;
  challenge_type?: string;
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
  error_category?: string | null;
  error_bucket?: string | null;
}

export interface BenchmarkSummary {
  config_name: string;
  description: string;
  retrieval_mode: string;
  grounding_guard: boolean;
  total_examples: number;
  verdict_accuracy: number;
  accuracy_ci_95?: [number, number];
  verdict_macro_f1: number;
  macro_f1_ci_95?: [number, number];
  per_class: Record<
    string,
    { precision: number; recall: number; f1: number; support: number }
  >;
  confusion_matrix?: Record<string, Record<string, number>>;
  error_taxonomy?: Record<string, number>;
  per_challenge_accuracy?: Record<string, number>;
  retrieval_recall_at_k: number;
  retrieval_mrr: number;
  mean_span_iou: number;
  brier_score: number;
  ece: number;
  hallucinations_caught: number;
  calibration_bins: CalibrationBin[];
  predictions: ExamplePrediction[];
}
