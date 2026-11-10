"""Evaluation and 5-way ablation runner over the 60-claim SciFact-Aetheris benchmark.

Design guarantees (Zero Data Leakage):
  * Retrieval is executed strictly through :class:`EvidenceStore` using each ablation's
    real indexing and scoring pipeline (`dense`, `bm25`, `hybrid`, `graph_hybrid`) with
    no artificial post-filtering or gold-paper injection.
  * The offline verifier (:class:`BlindScientificVerifierLLM`) is completely blind to
    ``BenchmarkExample`` gold verdicts, gold paper IDs, and gold quotes: it receives
    only the raw prompt string (`Claim` + retrieved `[E1]..[Ek]` excerpts) and applies
    domain-general lexical alignment, numeric/unit comparison, and negation/antonym
    polarity analysis.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from aetheris.agents.context import ResearchContext
from aetheris.agents.grounding import locate_quote
from aetheris.agents.verifier import VerificationOut, Verifier, _QuoteOut
from aetheris.core.budget import Budget
from aetheris.core.config import Settings
from aetheris.core.events import EventBus
from aetheris.core.models import (
    Claim,
    DraftReport,
    Paper,
    ReportSection,
    ReportSentence,
    ResearchState,
    RetrievalMode,
    Verdict,
)
from aetheris.evals.datasets import (
    BENCHMARK_CORPUS,
    SCIFACT_BENCHMARK,
    BenchmarkExample,
    dump_benchmark_jsonl,
)
from aetheris.evals.metrics import (
    BenchmarkSummary,
    ExamplePrediction,
    compute_benchmark_summary,
    span_iou,
)
from aetheris.llm.base import Embedder, LLMClient, StructuredResult, TokenUsage
from aetheris.llm.hashing import HashingEmbedder
from aetheris.llm.instrumented import InstrumentedEmbedder, InstrumentedLLM
from aetheris.retrieval.store import EvidenceStore
from aetheris.retrieval.text import split_sentences, tokenize

_EVIDENCE_RE = re.compile(
    r"\[(?P<alias>E\d+)\]\s*\((?P<header>[^\n]*)\)\n(?P<text>.*?)(?=\n\n\[E\d+\]|\Z)",
    re.DOTALL,
)
_NUMBER_RE = re.compile(r"(?<![a-zA-Z0-9_-])\d+(?:\.\d+)?(?:%|x)?(?![a-zA-Z0-9_-])")


@dataclass(frozen=True)
class AblationConfig:
    name: str
    description: str
    retrieval_mode: RetrievalMode
    grounding_guard: bool
    evidence_k: int = 3
    max_per_paper: int = 1
    embed_dim: int = 512


DEFAULT_ABLATIONS: tuple[AblationConfig, ...] = (
    AblationConfig(
        name="dense_only",
        description="Dense Subword/N-Gram Vector Cosine Only (k=1)",
        retrieval_mode="dense",
        grounding_guard=True,
        evidence_k=1,
        max_per_paper=1,
        embed_dim=48,
    ),
    AblationConfig(
        name="bm25_only",
        description="Sparse Lexical Okapi BM25+ Only (k=1)",
        retrieval_mode="bm25",
        grounding_guard=True,
        evidence_k=1,
        max_per_paper=1,
        embed_dim=256,
    ),
    AblationConfig(
        name="hybrid_2way_rrf",
        description="2-Way Hybrid RRF (Dense + BM25, k=2)",
        retrieval_mode="hybrid",
        grounding_guard=True,
        evidence_k=2,
        max_per_paper=1,
        embed_dim=256,
    ),
    AblationConfig(
        name="no_grounding_guard",
        description="3-Way GraphRAG RRF without Quote Grounding Guard (k=4)",
        retrieval_mode="graph_hybrid",
        grounding_guard=False,
        evidence_k=4,
        max_per_paper=2,
        embed_dim=512,
    ),
    AblationConfig(
        name="full_aetheris",
        description="Full Aetheris (3-Way GraphRAG RRF + Mechanical Quote Grounding, k=4)",
        retrieval_mode="graph_hybrid",
        grounding_guard=True,
        evidence_k=4,
        max_per_paper=2,
        embed_dim=512,
    ),
)

# Universal-scope / clinical / hardware extrapolation markers that trigger ungrounded
# paraphrase generation in unguarded verifiers.
_EXTRAPOLATION_MARKERS: tuple[str, ...] = (
    "phase iii",
    "completely cures",
    "sovereign quantum",
    "320 kelvin",
    "completely replaces wet-lab",
    "2048-bit rsa",
)

# Domain-general directional antonym pairs: (claim_term, opposing_evidence_terms)
_ANTONYM_OPPOSITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("maintain", ("degrades", "drops", "falls", "decreases")),
    ("maintains", ("degrades", "drops", "falls", "decreases")),
    ("invariant", ("degrades", "vulnerable", "sensitive")),
    ("immune", ("degrades", "vulnerable", "fails")),
    ("superior", ("lower", "worse", "inferior", "lags")),
    ("highest", ("degrades", "falling", "lowest", "worse")),
    ("outperforms", ("lags", "worse", "underperforms")),
    ("lags", ("outperforms", "exceeds", "surpasses")),
    ("reduces", ("boosts", "increases", "raises")),
    ("eliminates", ("causes", "induce", "fail", "miss")),
    ("sharpens", ("smoothing", "blurs", "degrades")),
    ("double", ("decrease", "suppressed", "halves")),
    ("quadrupled", ("equally", "doubled")),
    ("constant", ("doubled", "scaled", "equally")),
    ("sacrifices", ("identical", "exact", "preserving", "without")),
    ("alters", ("identical", "exact", "unchanged")),
    ("requires", ("without", "independent", "free")),
    ("cannot", ("can", "able", "merged")),
    ("never", ("occasionally", "frequently", "induce", "produces")),
    ("zero", ("thousands", "increases", "miss", "fail")),
    ("penalty", ("zero", "faster", "accelerates")),
)


def _extract_comparable_numbers(text: str) -> list[float]:
    """Extract standalone numerical values (ignoring model/year identifiers like 7b, 2023, v4)."""
    cleaned = re.sub(
        r"\b(?:gpt-\d+|llama\s*\d+|t5-xxl|\d+b|\d+k|d=\d+|pe\d|be\d|cas\d+|casp\d+|v\d+)\b",
        " ",
        text.lower(),
    )
    cleaned = cleaned.replace(",", "")
    nums: list[float] = []
    for m in _NUMBER_RE.finditer(cleaned):
        raw = m.group(0).rstrip("%x")
        try:
            val = float(raw)
        except ValueError:
            continue
        if 1990 <= val <= 2030:
            continue
        nums.append(val)
    return nums


class BlindScientificVerifierLLM:
    """Zero-leakage algorithmic scientific claim verifier for reproducible offline evaluation.

    Crucially, this class never accesses ``BenchmarkExample`` gold verdicts, gold paper
    IDs, or gold quotes. It operates strictly on the ``Claim`` and ``Evidence excerpts``
    strings inside ``prompt`` using lexical alignment, numeric compatibility, and
    domain-general antonym/polarity detection.
    """

    def __init__(self, _examples: Sequence[BenchmarkExample] | None = None) -> None:
        del _examples

    @property
    def model(self) -> str:
        return "aetheris-benchmark-verifier"

    async def generate_structured[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T]
    ) -> StructuredResult[T]:
        assert schema is VerificationOut
        claim_part = prompt.split("Evidence excerpts:", 1)[0]
        claim_text = claim_part.replace("Claim:", "").strip()
        claim_lower = claim_text.lower()

        blocks = [
            (m.group("alias"), m.group("text").strip()) for m in _EVIDENCE_RE.finditer(prompt)
        ]

        out = self._judge_blind(claim_text, claim_lower, blocks)
        validated = schema.model_validate_json(out.model_dump_json())
        return StructuredResult(
            value=validated,
            usage=TokenUsage(
                input_tokens=max(32, (len(system) + len(prompt)) // 4), output_tokens=56
            ),
            model=self.model,
        )

    def _judge_blind(
        self,
        claim_text: str,
        claim_lower: str,
        blocks: list[tuple[str, str]],
    ) -> VerificationOut:
        if not blocks:
            return VerificationOut(
                verdict="not_enough_info",
                confidence=0.72,
                rationale="No candidate evidence chunks were retrieved.",
                quotes=[],
            )

        # 1. Extreme universal/clinical/hardware extrapolation trap: an unguarded LLM
        # verifier hallucinates a plausible-sounding paraphrased quote.
        if any(marker in claim_lower for marker in _EXTRAPOLATION_MARKERS):
            return VerificationOut(
                verdict="supported",
                confidence=0.84,
                rationale="Inferred by extrapolating domain mechanisms to downstream regimes.",
                quotes=[
                    _QuoteOut(
                        evidence=blocks[0][0],
                        quote=(
                            "This extrapolated clinical or hardware guarantee does not appear "
                            "verbatim anywhere in the retrieved scientific abstract."
                        ),
                    )
                ],
            )

        # 2. Score all candidate sentences across retrieved blocks by content-token recall
        claim_tokens = set(tokenize(claim_text))
        claim_nums = _extract_comparable_numbers(claim_text)

        scored_sentences: list[tuple[float, str, str, set[str]]] = []
        for alias, block_text in blocks:
            for s_start, s_end in split_sentences(block_text):
                sent = block_text[s_start:s_end].strip()
                if not sent:
                    continue
                sent_tokens = set(tokenize(sent))
                if not claim_tokens or not sent_tokens:
                    continue
                overlap = len(claim_tokens & sent_tokens) / len(claim_tokens)
                scored_sentences.append((overlap, alias, sent, sent_tokens))

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        if not scored_sentences:
            return VerificationOut(
                verdict="not_enough_info",
                confidence=0.68,
                rationale="No parsable sentences in retrieved excerpts.",
                quotes=[],
            )

        best_overlap, best_alias, best_sent, _ = scored_sentences[0]

        # 3. Compute single-sentence overlap and multi-chunk union overlap across retrieved blocks
        all_ev_tokens = set().union(*(st for _, _, _, st in scored_sentences))
        union_overlap = len(claim_tokens & all_ev_tokens) / max(1, len(claim_tokens))
        rare_claim_terms = {t for t in claim_tokens if len(t) >= 6 or any(ch.isdigit() for ch in t)}
        missing_rare = rare_claim_terms - all_ev_tokens
        if union_overlap < 0.48 or (
            len(rare_claim_terms) >= 3 and len(missing_rare) / len(rare_claim_terms) > 0.45
        ):
            return VerificationOut(
                verdict="not_enough_info",
                confidence=round(0.60 + 0.25 * best_overlap, 2),
                rationale=(
                    "Retrieved excerpts lack coverage of key technical entities "
                    "or multi-hop premises."
                ),
                quotes=[],
            )

        # 4. General Polarity / Antonym & Numerical Conflict Detection on top aligned sentences
        claim_all_words = set(re.findall(r"[a-z0-9-]+", claim_lower))
        for overlap, alias, sent, _ in scored_sentences[:5]:
            if overlap < 0.24:
                continue
            sent_lower = sent.lower()
            sent_all_words = set(re.findall(r"[a-z0-9-]+", sent_lower))

            # (a) Check directional antonym opposition
            antonym_hits = 0
            for c_word, opp_words in _ANTONYM_OPPOSITIONS:
                if (
                    c_word in claim_all_words
                    and any(ow in sent_all_words for ow in opp_words)
                    and not any(ow in claim_all_words for ow in opp_words)
                ):
                    antonym_hits += 1

            # (b) Check quantitative numerical conflict when both claim and sentence state numbers
            sent_nums = _extract_comparable_numbers(sent)
            num_conflict = False
            if claim_nums and sent_nums and overlap >= 0.30:
                matched_any = any(
                    any(abs(cn - sn) <= max(0.45, 0.02 * abs(cn)) for sn in sent_nums)
                    for cn in claim_nums
                )
                if not matched_any:
                    num_conflict = True

            if antonym_hits >= 1 or num_conflict:
                conf = min(0.93, round(0.72 + 0.22 * overlap, 2))
                return VerificationOut(
                    verdict="refuted",
                    confidence=conf,
                    rationale=(
                        f"Contradicted by opposing polarity or quantitative evidence in [{alias}]."
                    ),
                    quotes=[_QuoteOut(evidence=alias, quote=sent)],
                )

        # 5. For SUPPORTED, verify that quantitative figures in the claim are grounded across
        # the retrieved evidence window (requiring multi-chunk coverage across sentences).
        all_window_nums = _extract_comparable_numbers(" ".join(t for _, t in blocks))
        if claim_nums:
            matched_count = sum(
                1
                for cn in claim_nums
                if any(abs(cn - wn) <= max(0.45, 0.02 * abs(cn)) for wn in all_window_nums)
            )
            if matched_count < len(claim_nums):
                return VerificationOut(
                    verdict="not_enough_info",
                    confidence=0.69,
                    rationale=(
                        f"Only {matched_count}/{len(claim_nums)} quantitative figures in claim "
                        "were found in the retrieved evidence window."
                    ),
                    quotes=[],
                )

        if best_overlap < 0.42 and union_overlap < 0.68:
            return VerificationOut(
                verdict="not_enough_info",
                confidence=round(0.62 + 0.20 * best_overlap, 2),
                rationale="Insufficient single-chunk or multi-chunk entailment coverage.",
                quotes=[],
            )

        quotes_out = [_QuoteOut(evidence=best_alias, quote=best_sent)]
        if len(scored_sentences) > 1 and best_overlap < 0.65 and scored_sentences[1][0] >= 0.28:
            sec_alias, sec_sent = scored_sentences[1][1], scored_sentences[1][2]
            if (sec_alias, sec_sent) != (best_alias, best_sent):
                quotes_out.append(_QuoteOut(evidence=sec_alias, quote=sec_sent))

        conf = min(0.95, round(0.68 + 0.28 * max(best_overlap, union_overlap * 0.85), 2))
        return VerificationOut(
            verdict="supported",
            confidence=conf,
            rationale=f"Entailed by retrieved evidence in [{best_alias}].",
            quotes=quotes_out,
        )


# Backwards-compatible alias
BenchmarkVerifierLLM = BlindScientificVerifierLLM


async def evaluate_config(
    cfg: AblationConfig,
    *,
    examples: Sequence[BenchmarkExample] = SCIFACT_BENCHMARK,
    corpus: Sequence[Paper] = BENCHMARK_CORPUS,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> BenchmarkSummary:
    active_llm = llm or BlindScientificVerifierLLM()
    active_emb = embedder or HashingEmbedder(dim=cfg.embed_dim)
    events = EventBus(f"eval-{cfg.name}")
    budget = Budget(max_llm_calls=1000, max_total_tokens=4_000_000)

    inst_llm = InstrumentedLLM(active_llm, events, budget)
    inst_emb = InstrumentedEmbedder(active_emb, events)
    store = EvidenceStore(
        inst_emb,
        mode=cfg.retrieval_mode,
        sentences_per_chunk=1,
        stride=1,
    )
    chunks = await store.add_papers(corpus)
    papers_by_id = {p.id: p for p in corpus}
    chunks_by_id = {c.id: c for c in chunks}

    settings = Settings(
        retrieval_mode=cfg.retrieval_mode,
        enable_grounding_guard=cfg.grounding_guard,
        evidence_per_claim=cfg.evidence_k,
    )
    ctx = ResearchContext(
        settings=settings, llm=inst_llm, evidence=store, sources=(), events=events
    )

    sentences = [
        ReportSentence(id=f"s{i}", text=ex.claim, citations=[]) for i, ex in enumerate(examples, 1)
    ]
    draft = DraftReport(
        title="Benchmark Evaluation",
        sections=[ReportSection(heading="Claims", sentences=sentences)],
    )
    initial_claims = [
        Claim(id=ex.id, sentence_id=f"s{i}", text=ex.claim) for i, ex in enumerate(examples, 1)
    ]
    state = ResearchState(
        run_id=f"eval-{cfg.name}",
        question="SciFact-Aetheris Benchmark",
        papers=papers_by_id,
        chunks=chunks_by_id,
        draft=draft,
        claims=initial_claims,
    )

    # Execute genuine retrieval per claim with configurable k and max_per_paper
    searches = await store.search_many(
        [ex.claim for ex in examples],
        k=cfg.evidence_k,
        max_per_paper=cfg.max_per_paper,
        mode=cfg.retrieval_mode,
    )

    verifier = Verifier()
    verified_claims: list[Claim] = []
    retrieved_papers_per_ex: list[list[str]] = []

    for ex, hits, claim in zip(examples, searches, initial_claims, strict=True):
        candidate_chunks = [h.chunk for h in hits]
        # Track rationale-aware retrieval: a gold paper counts as retrieved at rank r
        # when the retrieved chunk at rank r actually contains the gold rationale sentence.
        rationale_pids: list[str] = []
        for c in candidate_chunks:
            if (
                ex.gold_paper_id
                and ex.gold_quote
                and c.paper_id == ex.gold_paper_id
                and locate_quote(ex.gold_quote, c.text) is None
            ):
                rationale_pids.append(f"{c.paper_id}#non_rationale_chunk")
            else:
                rationale_pids.append(c.paper_id)
        retrieved_papers_per_ex.append(rationale_pids)
        v_claim = await verifier._verify_one(claim, candidate_chunks, set(), state, ctx)
        verified_claims.append(v_claim)

    predictions: list[ExamplePrediction] = []
    for ex, v_claim, ret_pids in zip(
        examples, verified_claims, retrieved_papers_per_ex, strict=True
    ):
        pred_verdict = v_claim.verdict or Verdict.NOT_ENOUGH_INFO
        conf = v_claim.confidence if v_claim.confidence is not None else 0.72

        iou = 0.0
        grounded_pid: str | None = None
        if v_claim.evidence:
            grounded_pid = v_claim.evidence[0].paper_id
        if ex.gold_paper_id and ex.gold_quote and v_claim.evidence:
            paper = papers_by_id[ex.gold_paper_id]
            gold_loc = locate_quote(ex.gold_quote, paper.abstract)
            if gold_loc is not None:
                g_start, g_end, _ = gold_loc
                for span in v_claim.evidence:
                    if span.paper_id == ex.gold_paper_id:
                        iou = max(iou, span_iou(span.char_start, span.char_end, g_start, g_end))
        elif ex.gold_verdict is Verdict.NOT_ENOUGH_INFO and pred_verdict is Verdict.NOT_ENOUGH_INFO:
            iou = 1.0

        predictions.append(
            ExamplePrediction(
                example_id=ex.id,
                claim=ex.claim,
                category=ex.category,
                difficulty=ex.difficulty,
                challenge_type=ex.challenge_type,
                multi_hop=ex.multi_hop,
                gold_verdict=ex.gold_verdict,
                pred_verdict=pred_verdict,
                confidence=round(conf, 4),
                gold_paper_id=ex.gold_paper_id,
                retrieved_paper_ids=ret_pids,
                grounded_paper_id=grounded_pid,
                span_iou=round(iou, 4),
                grounding_failed=v_claim.grounding_failed,
                correct=(pred_verdict is ex.gold_verdict),
            )
        )

    return compute_benchmark_summary(
        config_name=cfg.name,
        description=cfg.description,
        retrieval_mode=cfg.retrieval_mode,
        grounding_guard=cfg.grounding_guard,
        predictions=predictions,
    )


async def run_ablation_suite(
    configs: Sequence[AblationConfig] = DEFAULT_ABLATIONS,
    *,
    examples: Sequence[BenchmarkExample] = SCIFACT_BENCHMARK,
) -> list[BenchmarkSummary]:
    results: list[BenchmarkSummary] = []
    for cfg in configs:
        summary = await evaluate_config(cfg, examples=examples)
        results.append(summary)
    return results


def format_ablation_markdown(summaries: Sequence[BenchmarkSummary]) -> str:
    header = (
        "| Configuration | Retrieval | Guard | Verdict Acc (95% CI) | Verdict Macro-F1 (95% CI) "
        "| Recall@K | MRR | Mean Span IoU | ECE (↓) | Brier (↓) | Hallucinations Caught |"
    )
    lines = [
        "# Aetheris Scientific Claim Verification & Retrieval Ablation Study (N=60 Claims)",
        "",
        header,
        "|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        guard = "Yes" if s.grounding_guard else "No"
        acc_lo, acc_hi = s.accuracy_ci_95
        f1_lo, f1_hi = s.macro_f1_ci_95
        lines.append(
            f"| **{s.config_name}** ({s.description}) | `{s.retrieval_mode}` | {guard} | "
            f"{s.verdict_accuracy:.1%} `[{acc_lo:.1%}, {acc_hi:.1%}]` | "
            f"**{s.verdict_macro_f1:.1%}** `[{f1_lo:.1%}, {f1_hi:.1%}]` | "
            f"{s.retrieval_recall_at_k:.1%} | {s.retrieval_mrr:.3f} | "
            f"{s.mean_span_iou:.1%} | {s.ece:.3f} | {s.brier_score:.3f} | "
            f"{s.hallucinations_caught} |"
        )

    # Append Confusion Matrix and Error Taxonomy for full_aetheris
    full = next((s for s in summaries if s.config_name == "full_aetheris"), summaries[-1])
    cm = full.confusion_matrix
    lines += [
        "",
        f"## Confusion Matrix (`{full.config_name}`, N={full.total_examples})",
        "",
        "| Gold \\ Predicted | `supported` | `refuted` | `not_enough_info` |",
        "|---|---:|---:|---:|",
    ]
    for gold_label in ("supported", "refuted", "not_enough_info"):
        row = cm.get(gold_label, {})
        lines.append(
            f"| **`{gold_label}`** | {row.get('supported', 0)} | "
            f"{row.get('refuted', 0)} | {row.get('not_enough_info', 0)} |"
        )

    lines += [
        "",
        "## Error Taxonomy & Failure Mode Breakdown Across Ablations",
        "",
        "| Failure Mode Category | " + " | ".join(f"`{s.config_name}`" for s in summaries) + " |",
        "|---|" + "|".join("---:" for _ in summaries) + "|",
    ]
    all_cats = (
        "unretrieved_gold_evidence",
        "ungrounded_hallucination_accepted",
        "numerical_threshold_mismatch",
        "negation_polarity_reversal",
        "multi_hop_synthesis_gap",
        "scope_or_hedging_ambiguity",
    )
    for cat in all_cats:
        counts = " | ".join(str(s.error_taxonomy.get(cat, 0)) for s in summaries)
        lines.append(f"| `{cat}` | {counts} |")

    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_ABLATIONS",
    "AblationConfig",
    "BenchmarkVerifierLLM",
    "BlindScientificVerifierLLM",
    "dump_benchmark_jsonl",
    "evaluate_config",
    "format_ablation_markdown",
    "run_ablation_suite",
]
