"""Evaluation metrics: Verdict Macro-F1, Retrieval Recall/MRR, Span IoU, Brier, and ECE."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

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


class BenchmarkSummary(BaseModel):
    config_name: str
    description: str
    retrieval_mode: str
    grounding_guard: bool
    total_examples: int
    verdict_accuracy: float
    verdict_macro_f1: float
    per_class: dict[str, ClassMetrics]
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

    correct_count = sum(p.correct for p in predictions)
    accuracy = correct_count / n

    per_class: dict[str, ClassMetrics] = {}
    f1_scores: list[float] = []
    for v in Verdict:
        tp = sum(p.gold_verdict is v and p.pred_verdict is v for p in predictions)
        fp = sum(p.gold_verdict is not v and p.pred_verdict is v for p in predictions)
        fn = sum(p.gold_verdict is v and p.pred_verdict is not v for p in predictions)
        support = sum(p.gold_verdict is v for p in predictions)
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

    with_gold_paper = [p for p in predictions if p.gold_paper_id is not None]
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
    brier = sum((p.confidence - (1.0 if p.correct else 0.0)) ** 2 for p in predictions) / n
    bins: list[CalibrationBin] = []
    ece = 0.0
    for b_idx in range(num_bins):
        low = round(b_idx / num_bins, 2)
        high = round((b_idx + 1) / num_bins, 2)
        in_bin = [
            p
            for p in predictions
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
        verdict_macro_f1=round(macro_f1, 4),
        per_class=per_class,
        retrieval_recall_at_k=round(recall_at_k, 4),
        retrieval_mrr=round(mrr, 4),
        mean_span_iou=round(mean_iou, 4),
        brier_score=round(brier, 4),
        ece=round(ece, 4),
        hallucinations_caught=(
            sum(p.grounding_failed for p in predictions) if grounding_guard else 0
        ),
        calibration_bins=bins,
        predictions=list(predictions),
    )
