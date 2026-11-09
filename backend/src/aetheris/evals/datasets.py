"""Gold scientific claim-verification benchmark (SciFact-Aetheris) and JSONL loader."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from aetheris.core.models import Paper, Verdict
from aetheris.demo import DEMO_PAPERS


class BenchmarkExample(BaseModel):
    id: str
    claim: str
    gold_verdict: Verdict
    gold_paper_id: str | None = None
    gold_quote: str | None = None
    category: str = "general"
    multi_hop: bool = False
    trap_ungrounded_quote: bool = False


EXTRA_BENCHMARK_PAPERS: tuple[Paper, ...] = (
    Paper(
        id="arxiv:2310.06825",
        source="arxiv",
        title="Mistral 7B: Sliding Window Attention and Grouped-Query Inference",
        abstract=(
            "We introduce Mistral 7B, a 7-billion-parameter language model engineered for "
            "superior performance and inference efficiency. Mistral 7B leverages grouped-query "
            "attention (GQA) for faster decoding and sliding window attention (SWA) with a "
            "window size of 4,096 tokens to handle sequences of arbitrary length with a "
            "reduced KV-cache memory footprint. Across reasoning, mathematics, and code "
            "generation benchmarks, Mistral 7B outperforms Llama 2 13B."
        ),
        authors=("Albert Q. Jiang", "Alexandre Sablayrolles", "Arthur Mensch"),
        year=2023,
        venue="arXiv",
        doi="10.48550/arxiv.2310.06825",
        url="https://arxiv.org/abs/2310.06825",
        citation_count=3100,
    ),
    Paper(
        id="openalex:W4281990123",
        source="openalex",
        title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        abstract=(
            "Transformers are slow and memory-hungry on long sequences because the time and "
            "memory complexity of self-attention is quadratic in sequence length. We propose "
            "FlashAttention, an IO-aware exact attention algorithm that uses tiling to "
            "reduce the number of memory reads and writes between GPU high-bandwidth memory "
            "(HBM) and on-chip SRAM. FlashAttention trains BERT-large 15% faster and GPT-2 "
            "3.0x faster than standard PyTorch implementations without any approximation."
        ),
        authors=("Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Christopher Ré"),
        year=2022,
        venue="NeurIPS",
        doi="10.48550/arxiv.2205.14135",
        url="https://arxiv.org/abs/2205.14135",
        citation_count=3890,
    ),
)

BENCHMARK_CORPUS: tuple[Paper, ...] = (*DEMO_PAPERS, *EXTRA_BENCHMARK_PAPERS)

SCIFACT_BENCHMARK: tuple[BenchmarkExample, ...] = (
    BenchmarkExample(
        id="sf-01",
        claim=(
            "Retrieval-augmented generation reduces factual hallucinations by 42% in human "
            "evaluations on open-domain question answering benchmarks."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2005.11401",
        gold_quote=(
            "On open-domain question answering benchmarks including Natural Questions, "
            "WebQuestions, and CuratedTrec, RAG models generate responses that are more "
            "factual, specific, and diverse than closed-book seq2seq baselines, reducing "
            "factual hallucinations by 42% in human evaluations."
        ),
        category="RAG & Factuality",
    ),
    BenchmarkExample(
        id="sf-02",
        claim=(
            "Sentence-level evaluation overestimates factual precision by 14.6 percentage "
            "points compared to atomic claim verification in FActScore."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2305.14627",
        gold_quote=(
            "Across biographical passages from commercial and open-source language models, "
            "human judgements reveal that sentence-level evaluation overestimates factual "
            "precision by 14.6 percentage points compared to atomic claim verification."
        ),
        category="Evaluation",
    ),
    BenchmarkExample(
        id="sf-03",
        claim=(
            "Lexical BM25 retrieval alone captures 100% of multi-hop rationale spans on "
            "the SciFact scientific claim dataset."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="arxiv:2004.14906",
        gold_quote=(
            "Baseline pipeline systems that retrieve evidence via lexical BM25 alone miss "
            "23.4% of multi-hop rationale spans where terminology shifts between the claim "
            "and the abstract."
        ),
        category="Retrieval",
    ),
    BenchmarkExample(
        id="sf-04",
        claim=(
            "Injecting topically similar but factually irrelevant passages into RAG prompts "
            "increases unfaithful synthesis claims from 8.1% to 27.4%."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="openalex:W4387129001",
        gold_quote=(
            "In controlled distraction experiments across 4,200 scientific queries, "
            "injecting topically similar but factually irrelevant passages into the prompt "
            "increases unfaithful synthesis claims from 8.1% to 27.4%."
        ),
        category="Robustness",
    ),
    BenchmarkExample(
        id="sf-05",
        claim=(
            "Combining GraphRAG personalized random walks with dense search and BM25 via "
            "Reciprocal Rank Fusion raises multi-hop recall at k=8 from 71.2% to 88.6%."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2404.16130",
        gold_quote=(
            "When combined with dense vector search and BM25 via Reciprocal Rank Fusion, "
            "graph-augmented retrieval improves multi-hop evidence recall at k=8 from "
            "71.2% to 88.6% across multi-document scientific benchmarks."
        ),
        category="GraphRAG",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-06",
        claim=(
            "Mechanical substring and fuzzy quote alignment eliminates 94.2% of hallucinated "
            "evidence attributions in scientific agents."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="s2:9a8b7c6d5e4f",
        gold_quote=(
            "Enforcing mechanical substring and fuzzy alignment between verifier quotes and "
            "source abstracts eliminates 94.2% of hallucinated evidence attributions."
        ),
        category="Verification",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-07",
        claim=(
            "For compute-optimal transformer training, model parameters should be quadrupled "
            "while keeping training tokens constant."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="arxiv:2203.15556",
        gold_quote=(
            "By training over 400 language models ranging from 70 million to over 16 billion "
            "parameters on 5 to 500 billion tokens, we find that for compute-optimal training, "
            "the model size and the number of training tokens should be scaled equally: for "
            "every doubling of model size the number of training tokens should also be doubled."
        ),
        category="Scaling Laws",
    ),
    BenchmarkExample(
        id="sf-08",
        claim=(
            "Chinchilla (70B parameters, 1.4 trillion tokens) significantly outperforms "
            "Gopher (280B) and GPT-3 (175B) on downstream evaluation benchmarks."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2203.15556",
        gold_quote=(
            "We test this hypothesis by training a predicted compute-optimal model, "
            "Chinchilla (70B parameters, 1.4 trillion tokens), which uniformly and "
            "significantly outperforms Gopher (280B) and GPT-3 (175B) on downstream "
            "evaluation benchmarks."
        ),
        category="Scaling Laws",
    ),
    BenchmarkExample(
        id="sf-09",
        claim=(
            "AlphaFold achieved a median backbone accuracy of 0.96 angstrom r.m.s.d.95 in CASP14."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="openalex:W3177828909",
        gold_quote=(
            "In the 14th Critical Assessment of protein Structure Prediction (CASP14), "
            "AlphaFold demonstrated a median backbone accuracy of 0.96 angstrom r.m.s.d.95, "
            "compared with 2.8 angstrom r.m.s.d.95 for the next best competing method."
        ),
        category="Structural Biology",
    ),
    BenchmarkExample(
        id="sf-10",
        claim=(
            "AlphaFold structure predictions maintain peak accuracy even when multiple "
            "sequence alignment depth drops below 5 homologous sequences."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="openalex:W3177828909",
        gold_quote=(
            "However, prediction accuracy degrades significantly when multiple sequence "
            "alignment depth falls below 30 effective homologous sequences."
        ),
        category="Structural Biology",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-11",
        claim=(
            "Prime editing mediates all 12 possible base-to-base DNA conversions without "
            "requiring double-strand breaks and yields off-target indel rates below 0.5%."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="s2:3f4e5d6c7b8a",
        gold_quote=(
            "Prime editing directly writes new genetic information into a specified DNA site, "
            "mediates targeted insertions, deletions, and all 12 possible base-to-base "
            "conversions without requiring double-strand breaks (DSBs), and produces "
            "substantially lower off-target indel rates (< 0.5%) than Cas9 nuclease cleavage."
        ),
        category="Genome Editing",
    ),
    BenchmarkExample(
        id="sf-12",
        claim=(
            "FlashAttention uses tiling between GPU HBM and on-chip SRAM to train GPT-2 "
            "3.0x faster than standard PyTorch without approximation."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="openalex:W4281990123",
        gold_quote=(
            "FlashAttention trains BERT-large 15% faster and GPT-2 3.0x faster than standard "
            "PyTorch implementations without any approximation."
        ),
        category="Systems ML",
    ),
    BenchmarkExample(
        id="sf-13",
        claim=(
            "Mistral 7B uses a sliding window attention size of 4,096 tokens and outperforms "
            "Llama 2 13B across reasoning and code benchmarks."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2310.06825",
        gold_quote=(
            "Mistral 7B leverages grouped-query attention (GQA) for faster decoding and "
            "sliding window attention (SWA) with a window size of 4,096 tokens to handle "
            "sequences of arbitrary length with a reduced KV-cache memory footprint."
        ),
        category="Systems ML",
    ),
    BenchmarkExample(
        id="sf-14",
        claim=(
            "Retrieval-augmented generation reduces energy consumption in sovereign quantum "
            "datacenter cryostats by 73%."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Out-of-Scope / Unverifiable",
        trap_ungrounded_quote=True,
    ),
    BenchmarkExample(
        id="sf-15",
        claim=(
            "Prime editing completely cures autosomal dominant Huntington's disease in "
            "Phase III human clinical trials with zero adverse events."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Clinical Extrapolation Trap",
        trap_ungrounded_quote=True,
    ),
)


def load_benchmark_jsonl(path: Path) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            examples.append(BenchmarkExample.model_validate(json.loads(line)))
    return examples
