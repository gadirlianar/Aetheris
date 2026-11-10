"""Evaluation metrics: Macro-F1, 95% Bootstrap CIs, Confusion Matrix, Error Taxonomy, and ECE."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from pydantic import BaseModel, Field

from aetheris.core.models import Verdict


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class CalibrationBin(BaseModel):
    bin_lower: float
    bin_upper: float
    avg_confidence: float
    accuracy: float
    count: int


class ExamplePrediction(BaseModel):
    example_id: str
    claim: str
    category: str
    difficulty: str = "standard"
    challenge_type: str = "direct_entailment"
    multi_hop: bool
    gold_verdict: Verdict
    pred_verdict: Verdict
    confidence: float
    gold_paper_id: str | None = None
    retrieved_paper_ids: list[str]
    grounded_paper_id: str | None = None
    span_iou: float = 0.0
    grounding_failed: bool = False
    correct: bool
    error_category: str | None = None


class BenchmarkSummary(BaseModel):
    config_name: str
    description: str
    retrieval_mode: str
    grounding_guard: bool
    total_examples: int
    verdict_accuracy: float
    accuracy_ci_95: tuple[float, float] = (0.0, 0.0)
    verdict_macro_f1: float
    macro_f1_ci_95: tuple[float, float] = (0.0, 0.0)
    per_class: dict[str, ClassMetrics]
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    error_taxonomy: dict[str, int] = Field(default_factory=dict)
    per_challenge_accuracy: dict[str, float] = Field(default_factory=dict)
    retrieval_recall_at_k: float
    retrieval_mrr: float
    mean_span_iou: float
    brier_score: float
    ece: float
    hallucinations_caught: int
    calibration_bins: list[CalibrationBin]
    predictions: list[ExamplePrediction]


def span_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Character-level Intersection-over-Union between two half-open intervals."""
    if a_end <= a_start or b_end <= b_start:
        return 0.0
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def _compute_macro_f1(preds: Sequence[ExamplePrediction]) -> float:
    f1_scores: list[float] = []
    for v in Verdict:
        tp = sum(p.gold_verdict is v and p.pred_verdict is v for p in preds)
        fp = sum(p.gold_verdict is not v and p.pred_verdict is v for p in preds)
        fn = sum(p.gold_verdict is v and p.pred_verdict is not v for p in preds)
        support = sum(p.gold_verdict is v for p in preds)
        if support == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(float(sorted_vals[0]), 4)
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return round(float(sorted_vals[lo]), 4)
    w = pos - lo
    return round(float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w), 4)


def bootstrap_confidence_intervals(
    predictions: Sequence[ExamplePrediction],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute 95% nonparametric bootstrap CIs for (Accuracy, Macro-F1)."""
    n = len(predictions)
    if n == 0:
        return (0.0, 0.0), (0.0, 0.0)
    rng = random.Random(seed)
    acc_samples: list[float] = []
    f1_samples: list[float] = []
    pred_list = list(predictions)
    for _ in range(n_resamples):
        sample = [pred_list[rng.randrange(n)] for _ in range(n)]
        acc_samples.append(sum(p.correct for p in sample) / n)
        f1_samples.append(_compute_macro_f1(sample))
    acc_samples.sort()
    f1_samples.sort()
    acc_ci = (_percentile(acc_samples, 2.5), _percentile(acc_samples, 97.5))
    f1_ci = (_percentile(f1_samples, 2.5), _percentile(f1_samples, 97.5))
    return acc_ci, f1_ci


def classify_error_category(pred: ExamplePrediction, *, grounding_guard: bool) -> str | None:
    """Assign a root-cause error taxonomy label when ``pred.correct`` is False."""
    if pred.correct:
        return None
    if pred.gold_paper_id and pred.gold_paper_id not in pred.retrieved_paper_ids:
        return "unretrieved_gold_evidence"
    if not grounding_guard and pred.grounding_failed:
        return "ungrounded_hallucination_accepted"
    if pred.challenge_type == "numerical_threshold_trap":
        return "numerical_threshold_mismatch"
    if pred.challenge_type == "negation_antonym_reversal":
        return "negation_polarity_reversal"
    if pred.multi_hop or pred.challenge_type == "multi_hop_synthesis":
        return "multi_hop_synthesis_gap"
    return "scope_or_hedging_ambiguity"


def compute_benchmark_summary(
    *,
    config_name: str,
    description: str,
    retrieval_mode: str,
    grounding_guard: bool,
    predictions: Sequence[ExamplePrediction],
    num_bins: int = 5,
) -> BenchmarkSummary:
    n = len(predictions)
    if n == 0:
        raise ValueError("cannot compute summary over zero predictions")

    enriched_preds = [
        p.model_copy(
            update={"error_category": classify_error_category(p, grounding_guard=grounding_guard)}
        )
        for p in predictions
    ]

    correct_count = sum(p.correct for p in enriched_preds)
    accuracy = correct_count / n

    per_class: dict[str, ClassMetrics] = {}
    f1_scores: list[float] = []
    for v in Verdict:
        tp = sum(p.gold_verdict is v and p.pred_verdict is v for p in enriched_preds)
        fp = sum(p.gold_verdict is not v and p.pred_verdict is v for p in enriched_preds)
        fn = sum(p.gold_verdict is v and p.pred_verdict is not v for p in enriched_preds)
        support = sum(p.gold_verdict is v for p in enriched_preds)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[v.value] = ClassMetrics(
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            support=support,
        )
        if support > 0:
            f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    acc_ci, f1_ci = bootstrap_confidence_intervals(enriched_preds)

    # 3x3 Confusion Matrix: gold_verdict (row) -> pred_verdict (col)
    confusion: dict[str, dict[str, int]] = {g.value: {p.value: 0 for p in Verdict} for g in Verdict}
    for p in enriched_preds:
        confusion[p.gold_verdict.value][p.pred_verdict.value] += 1

    # Error Taxonomy breakdown
    error_taxonomy: dict[str, int] = {
        "unretrieved_gold_evidence": 0,
        "ungrounded_hallucination_accepted": 0,
        "numerical_threshold_mismatch": 0,
        "negation_polarity_reversal": 0,
        "multi_hop_synthesis_gap": 0,
        "scope_or_hedging_ambiguity": 0,
    }
    for p in enriched_preds:
        if p.error_category is not None:
            error_taxonomy[p.error_category] = error_taxonomy.get(p.error_category, 0) + 1

    # Per-challenge-type accuracy
    challenge_groups: dict[str, list[bool]] = {}
    for p in enriched_preds:
        challenge_groups.setdefault(p.challenge_type, []).append(p.correct)
    per_challenge_accuracy = {
        ct: round(sum(vals) / len(vals), 4) for ct, vals in sorted(challenge_groups.items())
    }

    with_gold_paper = [p for p in enriched_preds if p.gold_paper_id is not None]
    hits = 0
    rr_sum = 0.0
    for p in with_gold_paper:
        if p.gold_paper_id in p.retrieved_paper_ids:
            hits += 1
            rank = p.retrieved_paper_ids.index(p.gold_paper_id) + 1
            rr_sum += 1.0 / rank
    recall_at_k = hits / len(with_gold_paper) if with_gold_paper else 0.0
    mrr = rr_sum / len(with_gold_paper) if with_gold_paper else 0.0

    iou_candidates = [p.span_iou for p in with_gold_paper]
    mean_iou = sum(iou_candidates) / len(iou_candidates) if iou_candidates else 0.0

    # Brier score & Expected Calibration Error (ECE)
    brier = sum((p.confidence - (1.0 if p.correct else 0.0)) ** 2 for p in enriched_preds) / n
    bins: list[CalibrationBin] = []
    ece = 0.0
    for b_idx in range(num_bins):
        low = round(b_idx / num_bins, 2)
        high = round((b_idx + 1) / num_bins, 2)
        in_bin = [
            p
            for p in enriched_preds
            if (low <= p.confidence < high) or (b_idx == num_bins - 1 and p.confidence == high)
        ]
        if in_bin:
            avg_conf = sum(p.confidence for p in in_bin) / len(in_bin)
            bin_acc = sum(p.correct for p in in_bin) / len(in_bin)
            ece += (len(in_bin) / n) * abs(avg_conf - bin_acc)
        else:
            avg_conf, bin_acc = (low + high) / 2, 0.0
        bins.append(
            CalibrationBin(
                bin_lower=low,
                bin_upper=high,
                avg_confidence=round(avg_conf, 4),
                accuracy=round(bin_acc, 4),
                count=len(in_bin),
            )
        )

    return BenchmarkSummary(
        config_name=config_name,
        description=description,
        retrieval_mode=retrieval_mode,
        grounding_guard=grounding_guard,
        total_examples=n,
        verdict_accuracy=round(accuracy, 4),
        accuracy_ci_95=acc_ci,
        verdict_macro_f1=round(macro_f1, 4),
        macro_f1_ci_95=f1_ci,
        per_class=per_class,
        confusion_matrix=confusion,
        error_taxonomy=error_taxonomy,
        per_challenge_accuracy=per_challenge_accuracy,
        retrieval_recall_at_k=round(recall_at_k, 4),
        retrieval_mrr=round(mrr, 4),
        mean_span_iou=round(mean_iou, 4),
        brier_score=round(brier, 4),
        ece=round(ece, 4),
        hallucinations_caught=(
            sum(p.grounding_failed for p in enriched_preds) if grounding_guard else 0
        ),
        calibration_bins=bins,
        predictions=enriched_preds,
    )
