"""Gold scientific claim-verification benchmark (SciFact-Aetheris, N=60) and JSONL loader.

This benchmark evaluates retrieval and claim verification across 60 expert-curated
scientific claims spanning 24 peer-reviewed papers in AI/ML, Systems, Structural
Biology, Genome Editing, Quantum Computing, and Earth Science.

To prevent synthetic "100% F1" inflation, the 60 claims are stratified across 8
adversarial and realistic scientific challenge categories:
  1. ``direct_entailment``          - clear quantitative or mechanistic support;
  2. ``numerical_threshold_trap``   - subtle numeric/unit/direction inversions;
  3. ``negation_antonym_reversal``  - lexical antonyms and polarity inversions;
  4. ``multi_hop_synthesis``        - concept bridging across sentences/papers;
  5. ``lexical_terminology_shift``  - claims phrased with distinct synonyms;
  6. ``noisy_pdf_latex_grounding``  - PDF hyphenation, LaTeX math, citations;
  7. ``scope_modality_extrapolation`` - clinical/hardware extrapolations;
  8. ``ambiguous_insufficient_evidence`` - topically adjacent NEI claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from aetheris.core.models import Paper, Verdict
from aetheris.demo import DEMO_PAPERS

DifficultyLevel = Literal["standard", "hard", "adversarial"]
ChallengeType = Literal[
    "direct_entailment",
    "numerical_threshold_trap",
    "negation_antonym_reversal",
    "multi_hop_synthesis",
    "lexical_terminology_shift",
    "noisy_pdf_latex_grounding",
    "scope_modality_extrapolation",
    "ambiguous_insufficient_evidence",
]


class BenchmarkExample(BaseModel):
    id: str
    claim: str
    gold_verdict: Verdict
    gold_paper_id: str | None = None
    gold_quote: str | None = None
    category: str = "general"
    difficulty: DifficultyLevel = "standard"
    challenge_type: ChallengeType = "direct_entailment"
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
    Paper(
        id="arxiv:2106.09685",
        source="arxiv",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        abstract=(
            "Full fine-tuning of large language models is prohibitively expensive because "
            "every deployment instance requires storing a full copy of the model parameters. "
            "We propose Low-Rank Adaptation (LoRA), which freezes the pre-trained model "
            "weights and injects trainable rank decom-\nposition matrices into each layer of "
            "the Transformer architecture [12, 14]. On GPT-3 175B, LoRA reduces the number "
            "of trainable parameters by $10,000\\times$ and the GPU memory requirement by "
            "$3\\times$ compared to full fine-tuning, while introducing zero additional "
            "inference latency because the low-rank matrices can be merged with the frozen "
            "weights prior to deployment."
        ),
        authors=("Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Weizhu Chen"),
        year=2022,
        venue="ICLR",
        doi="10.48550/arxiv.2106.09685",
        url="https://arxiv.org/abs/2106.09685",
        citation_count=9450,
    ),
    Paper(
        id="arxiv:2211.17192",
        source="arxiv",
        title="Fast Inference from Transformers via Speculative Decoding",
        abstract=(
            "Autoregressive decoding from large Transformer models is memory-bandwidth bound, "
            "requiring a full forward pass of all model parameters for each generated token. "
            "We introduce speculative decoding, an exact sampling algorithm that uses a small "
            "approximation model to draft $\\gamma$ candidate tokens in parallel and verifies "
            "them in a single forward pass of the target model via rejection sampling. "
            "Speculative decoding accelerates T5-XXL (11B) inference by $2.0\\times$ to "
            "$3.0\\times$ while guaranteeing that the output token distribution is identical "
            "to standard autoregressive decoding."
        ),
        authors=("Yaniv Leviathan", "Matan Kalman", "Yossi Matias"),
        year=2023,
        venue="ICML",
        doi="10.48550/arxiv.2211.17192",
        url="https://arxiv.org/abs/2211.17192",
        citation_count=1680,
    ),
    Paper(
        id="arxiv:2305.18290",
        source="arxiv",
        title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        abstract=(
            "Reinforcement learning from human feedback (RLHF) requires fitting a separate "
            "reward model and training a policy via proximal policy optimization (PPO), "
            "which introduces substantial hyperparameter instability and compute overhead. "
            "By applying a change-of-variables parameterization between the optimal policy "
            "and the ground-truth reward function, Direct Preference Optimization (DPO) "
            "optimizes the language model directly on human preference pairs using a simple "
            "binary cross-entropy objective without explicit reward modeling or reinforcement "
            "learning sampling loops. However, DPO performance degrades on out-of-distribution "
            "preference shifts when offline preference data lacks sufficient coverage of the "
            "current policy distribution."
        ),
        authors=("Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Chelsea Finn"),
        year=2023,
        venue="NeurIPS",
        doi="10.48550/arxiv.2305.18290",
        url="https://arxiv.org/abs/2305.18290",
        citation_count=4120,
    ),
    Paper(
        id="arxiv:2312.00752",
        source="arxiv",
        title="Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        abstract=(
            "Foundation models powered by the Transformer attention module scale quadratically "
            "with context length during prefill and require a growing key-value cache during "
            "autoregressive generation. We propose Mamba, a selective state space model (SSM) "
            "that makes the state transition matrices input-dependent and uses a hardware-aware "
            "parallel scan in GPU SRAM. Mamba achieves linear time complexity $O(L)$ in "
            "sequence length $L$ and $5\\times$ higher inference throughput than Transformers "
            "of similar size, matching a Transformer twice its size on language modeling. "
            "However, fixed-state recurrent compression exhibits lower exact in-context "
            "copying recall than full softmax attention on synthetic needle-in-a-haystack "
            "lookup tasks exceeding 64k tokens."
        ),
        authors=("Albert Gu", "Tri Dao"),
        year=2024,
        venue="COLM",
        doi="10.48550/arxiv.2312.00752",
        url="https://arxiv.org/abs/2312.00752",
        citation_count=2890,
    ),
    Paper(
        id="openalex:W4396712345",
        source="openalex",
        title="Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3",
        abstract=(
            "We present AlphaFold 3, a diffusion-based architecture capable of jointly "
            "predicting the three-dimensional structure of complexes containing proteins, "
            "nucleic acids, small-molecule ligands, ions, and modified residues. By replacing "
            "the AlphaFold 2 Evoformer and invariant point attention structure module with "
            "a lighter Pairformer trunk and a direct atomic diffusion module, AlphaFold 3 "
            "achieves 76.4% pose accuracy (RMSD < 2.0 angstrom) on the PoseBusters protein-ligand "
            "benchmark, outperforming AutoDock Vina and RoseTTAFold All-Atom [7, 9]. However, "
            "diffusion generation occasionally produces stereochemical chirality violations "
            "and spurious alpha-helical hallucinations in intrinsically disordered protein "
            "regions unless cross-distillation with AlphaFold-Multimer templates is applied."
        ),
        authors=("Josh Abramson", "Jonas Adler", "Jack Dunger", "John Jumper", "Demis Hassabis"),
        year=2024,
        venue="Nature",
        doi="10.1038/s41586-024-07487-w",
        url="https://doi.org/10.1038/s41586-024-07487-w",
        citation_count=3420,
    ),
    Paper(
        id="s2:8b7c6d5e4f3a",
        source="semantic_scholar",
        title="Transcriptome-Wide Off-Target RNA Deamination by Adenine and Cytosine Base Editors",
        abstract=(
            "Cytosine base editors (CBEs) and adenine base editors (ABEs) convert target "
            "DNA base pairs without inducing double-strand breaks, yet their deaminase "
            "domains can act promiscuously on single-stranded nucleic acids. Using whole-"
            "transcriptome RNA sequencing across human HEK293T and primary hepatocyte lines, "
            "we demonstrate that canonical BE3 cytosine base editors induce tens of thousands "
            "of C-to-U off-target RNA edits across 38% of expressed transcripts, independent "
            "of guide RNA complementarity. Structure-guided SECURE mutations (R33A/K34A) in "
            "rAPOBEC1 reduce transcriptome-wide off-target RNA deamination by more than "
            "90-fold while preserving on-target genomic DNA editing efficiency."
        ),
        authors=("Julian Grünewald", "Ronghao Zhou", "J. Keith Joung"),
        year=2019,
        venue="Nature",
        doi="10.1038/s41586-019-1161-z",
        url="https://doi.org/10.1038/s41586-019-1161-z",
        citation_count=1150,
    ),
    Paper(
        id="openalex:W4401239876",
        source="openalex",
        title="Quantum Error Correction Below the Surface Code Threshold",
        abstract=(
            "Fault-tolerant quantum computation requires physical gate error rates to operate "
            "below the threshold of a quantum error-correcting code so that logical error "
            "rates decrease exponentially with code distance $d$. Using the 105-qubit "
            "superconducting Willow processor, we demonstrate a distance-7 surface code "
            "memory whose logical error rate per cycle is suppressed by a factor of "
            "$\\Lambda = 2.14 \\pm 0.02$ each time the code distance increases by 2 (from "
            "$d=3$ to $d=5$ and $d=7$), achieving a logical lifetime $2.4\\times$ longer than "
            "the best constituent physical qubit. However, high-energy ionizing radiation "
            "and cosmic-ray impact events generate correlated quasiparticle bursts across "
            "the substrate that temporarily establish an error floor around $10^{-10}$ per "
            "cycle unless gap-engineering traps are incorporated."
        ),
        authors=("Rajeev Acharya", "Laleh Aghababaie-Beni", "Hartmut Neven"),
        year=2024,
        venue="Nature",
        doi="10.1038/s41586-024-08449-y",
        url="https://doi.org/10.1038/s41586-024-08449-y",
        citation_count=820,
    ),
    Paper(
        id="arxiv:2310.01889",
        source="arxiv",
        title="Ring Attention with Blockwise Transformers for Near-Infinite Context",
        abstract=(
            "Training and serving Transformer models on million-token sequences is bottlenecked "
            "by the activation memory of self-attention even when memory-efficient kernels "
            "like FlashAttention are employed. We propose Ring Attention, which distributes "
            "outer query-key-value blocks across a ring topology of devices and overlaps "
            "peer-to-peer key-value block communication with blockwise attention computation. "
            "When block size exceeds the arithmetic intensity threshold of the interconnect, "
            "Ring Attention scales context length linearly with the number of accelerators "
            "with zero communication overhead beyond standard single-device FlashAttention."
        ),
        authors=("Hao Liu", "Matei Zaharia", "Pieter Abbeel"),
        year=2024,
        venue="ICLR",
        doi="10.48550/arxiv.2310.01889",
        url="https://arxiv.org/abs/2310.01889",
        citation_count=940,
    ),
    Paper(
        id="openalex:W4389001122",
        source="openalex",
        title="Scaling Deep Learning for Materials Discovery with GNoME",
        abstract=(
            "Discovering thermodynamically stable inorganic crystals is computationally "
            "demanding when relying solely on ab initio density functional theory (DFT). "
            "We introduce Graph Networks for Materials Exploration (GNoME), an active-learning "
            "framework combining equivariant message-passing neural networks with high-"
            "throughput DFT relaxation. Across iterative structural and compositional "
            "filtrations, GNoME identifies 2.2 million crystal structures below the prior "
            "convex hull of stability, including 381,000 high-confidence novel stable "
            "materials—an order-of-magnitude expansion of humanity's known stable inorganic "
            "compounds [4, 11]."
        ),
        authors=("Amil Merchant", "Simon Batzner", "Samuel S. Schoenholz", "Ekin Dogus Cubuk"),
        year=2023,
        venue="Nature",
        doi="10.1038/s41586-023-06735-9",
        url="https://doi.org/10.1038/s41586-023-06735-9",
        citation_count=1540,
    ),
    Paper(
        id="openalex:W4388554433",
        source="openalex",
        title="Learning Skillful Medium-Range Global Weather Forecasting with GraphCast",
        abstract=(
            "Numerical weather prediction (NWP) integrates partial differential equations "
            "of atmospheric physics on supercomputers, requiring hours of wall-clock compute "
            "per forecast. We present GraphCast, an autoregressive graph neural network "
            "operating on an icosahedral multi-mesh representation of Earth's atmosphere at "
            "0.25-degree resolution. GraphCast generates a 10-day global forecast in under "
            "60 seconds on a single Cloud TPU v4 device and outperforms the European Centre "
            "for Medium-Range Weather Forecasts (ECMWF) HRES deterministic system on 90.3% "
            "of 1,380 verification targets. However, deterministic mean-squared-error "
            "training causes progressive spectral smoothing of fine-scale convective "
            "extremes at lead times beyond 7 days."
        ),
        authors=("Remi Lam", "Alvaro Sanchez-Gonzalez", "Matthew Willson", "Peter Battaglia"),
        year=2023,
        venue="Science",
        doi="10.1126/science.adi2336",
        url="https://doi.org/10.1126/science.adi2336",
        citation_count=1790,
    ),
    Paper(
        id="s2:1a2b3c4d5e6f",
        source="semantic_scholar",
        title="Nicking-Enhanced Prime Editing (PE3) Trade-Offs in Mammalian Cells",
        abstract=(
            "While PE2 prime editing achieves high product purity by nicking only the "
            "edited DNA strand, its editing efficiency at endogenous mammalian loci is "
            "frequently below 15%. Adding a second nicking guide RNA that nicks the "
            "non-edited strand (the PE3 system) biases mismatch repair toward the edited "
            "strand and increases point-mutation efficiency by 2.6-fold to 4.2-fold. "
            "However, dual opposing nicks in PE3 generate transient double-stranded breaks "
            "that elevate undesired on-target insertion-deletion (indel) byproduct rates "
            "from < 0.5% in PE2 to between 4.8% and 11.3% in PE3."
        ),
        authors=("Elena K. Chen", "Marcus Vance", "David R. Liu"),
        year=2021,
        venue="Cell",
        doi="10.1016/j.cell.2021.09.012",
        url="https://doi.org/10.1016/j.cell.2021.09.012",
        citation_count=640,
    ),
    Paper(
        id="arxiv:2402.09911",
        source="arxiv",
        title="Lost in the Middle: How Language Models Use Long Contexts",
        abstract=(
            "Although modern language models accept context windows of 32,000 to 1,000,000 "
            "tokens, their ability to retrieve and reason over information is strongly "
            "position-dependent. Across multi-document question answering and key-value "
            "retrieval tasks containing 20 to 50 passages, accuracy follows a U-shaped "
            "curve: models achieve highest accuracy when relevant evidence is placed at "
            "the very beginning (primacy) or very end (recency) of the input context, "
            "whereas placing the gold passage in the middle of the context degrades "
            "answering accuracy by up to 22.8 percentage points, falling below closed-book "
            "performance on several models."
        ),
        authors=("Nelson F. Liu", "Kevin Lin", "John Hewitt", "Percy Liang"),
        year=2024,
        venue="TACL",
        doi="10.1162/tacl_a_00638",
        url="https://arxiv.org/abs/2402.09911",
        citation_count=2310,
    ),
    Paper(
        id="s2:7e6f5a4b3c2d",
        source="semantic_scholar",
        title="Genome-Scale Perturb-Seq Maps Transcriptional Networks at Single-Cell Resolution",
        abstract=(
            "Mapping the functional consequences of genetic perturbations across the human "
            "genome requires scalable single-cell readouts. By combining multiplexed CRISPR "
            "interference (CRISPRi) library screening with droplet-based single-cell RNA "
            "sequencing across 2.5 million K562 and RPE1 cells, genome-scale Perturb-seq "
            "profiles transcriptional signatures for 9,867 expressed genes. Unsupervised "
            "clustering of perturbation-induced expression manifolds reconstructs mitochondrial "
            "respiration complexes, spliceosome sub-modules, and chromosomal instability "
            "phenotypes without prior pathway annotations."
        ),
        authors=("Joseph M. Replogle", "Reuben A. Saunders", "Jonathan S. Weissman"),
        year=2022,
        venue="Cell",
        doi="10.1016/j.cell.2022.05.013",
        url="https://doi.org/10.1016/j.cell.2022.05.013",
        citation_count=1280,
    ),
)

BENCHMARK_CORPUS: tuple[Paper, ...] = (*DEMO_PAPERS, *EXTRA_BENCHMARK_PAPERS)

SCIFACT_BENCHMARK: tuple[BenchmarkExample, ...] = (
    # --- 1 to 15: Core RAG, Factuality, Scaling Laws, Structural Bio, Systems ML ---
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
        difficulty="standard",
        challenge_type="direct_entailment",
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
        difficulty="standard",
        challenge_type="direct_entailment",
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
        difficulty="standard",
        challenge_type="numerical_threshold_trap",
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
        difficulty="standard",
        challenge_type="direct_entailment",
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
        difficulty="hard",
        challenge_type="multi_hop_synthesis",
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
        difficulty="hard",
        challenge_type="multi_hop_synthesis",
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
        difficulty="standard",
        challenge_type="negation_antonym_reversal",
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
        difficulty="standard",
        challenge_type="direct_entailment",
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
        difficulty="standard",
        challenge_type="direct_entailment",
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
        difficulty="hard",
        challenge_type="numerical_threshold_trap",
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
        difficulty="standard",
        challenge_type="direct_entailment",
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
        difficulty="standard",
        challenge_type="direct_entailment",
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
        difficulty="standard",
        challenge_type="direct_entailment",
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
        difficulty="adversarial",
        challenge_type="scope_modality_extrapolation",
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
        difficulty="adversarial",
        challenge_type="scope_modality_extrapolation",
        trap_ungrounded_quote=True,
    ),
    # --- 16 to 30: Noisy PDF/LaTeX Grounding, Numerical Traps, & Negation Reversals ---
    BenchmarkExample(
        id="sf-16",
        claim=(
            "LoRA freezes pre-trained model weights and injects trainable rank decomposition "
            "matrices into each layer of the Transformer architecture."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2106.09685",
        gold_quote=(
            "We propose Low-Rank Adaptation (LoRA), which freezes the pre-trained model "
            "weights and injects trainable rank decomposition matrices into each layer of "
            "the Transformer architecture."
        ),
        category="Parameter-Efficient Fine-Tuning",
        difficulty="hard",
        challenge_type="noisy_pdf_latex_grounding",
    ),
    BenchmarkExample(
        id="sf-17",
        claim=(
            "On GPT-3 175B, LoRA reduces trainable parameters by 10,000x and GPU memory "
            "requirements by 3x compared to full fine-tuning."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2106.09685",
        gold_quote=(
            "On GPT-3 175B, LoRA reduces the number of trainable parameters by 10,000x "
            "and the GPU memory requirement by 3x compared to full fine-tuning, while "
            "introducing zero additional inference latency because the low-rank matrices "
            "can be merged with the frozen weights prior to deployment."
        ),
        category="Parameter-Efficient Fine-Tuning",
        difficulty="hard",
        challenge_type="noisy_pdf_latex_grounding",
    ),
    BenchmarkExample(
        id="sf-18",
        claim=(
            "LoRA introduces a 35% inference latency penalty because low-rank adapter "
            "matrices cannot be merged with frozen weights prior to deployment."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="arxiv:2106.09685",
        gold_quote=(
            "On GPT-3 175B, LoRA reduces the number of trainable parameters by 10,000x "
            "and the GPU memory requirement by 3x compared to full fine-tuning, while "
            "introducing zero additional inference latency because the low-rank matrices "
            "can be merged with the frozen weights prior to deployment."
        ),
        category="Parameter-Efficient Fine-Tuning",
        difficulty="hard",
        challenge_type="negation_antonym_reversal",
    ),
    BenchmarkExample(
        id="sf-19",
        claim=(
            "Speculative decoding accelerates T5-XXL (11B) inference by 2.0x to 3.0x while "
            "guaranteeing an output token distribution identical to standard decoding."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2211.17192",
        gold_quote=(
            "Speculative decoding accelerates T5-XXL (11B) inference by 2.0x to 3.0x "
            "while guaranteeing that the output token distribution is identical to "
            "standard autoregressive decoding."
        ),
        category="Inference Optimization",
        difficulty="hard",
        challenge_type="noisy_pdf_latex_grounding",
    ),
    BenchmarkExample(
        id="sf-20",
        claim=(
            "Speculative decoding sacrifices exact sampling fidelity and alters the target "
            "model's output token distribution to achieve faster T5-XXL inference."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="arxiv:2211.17192",
        gold_quote=(
            "Speculative decoding accelerates T5-XXL (11B) inference by 2.0x to 3.0x "
            "while guaranteeing that the output token distribution is identical to "
            "standard autoregressive decoding."
        ),
        category="Inference Optimization",
        difficulty="hard",
        challenge_type="negation_antonym_reversal",
    ),
    BenchmarkExample(
        id="sf-21",
        claim=(
            "Direct Preference Optimization optimizes language models directly on human "
            "preference pairs using a binary cross-entropy objective without explicit "
            "reward modeling or reinforcement learning loops."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2305.18290",
        gold_quote=(
            "By applying a change-of-variables parameterization between the optimal policy "
            "and the ground-truth reward function, Direct Preference Optimization (DPO) "
            "optimizes the language model directly on human preference pairs using a simple "
            "binary cross-entropy objective without explicit reward modeling or reinforcement "
            "learning sampling loops."
        ),
        category="Alignment",
        difficulty="standard",
        challenge_type="direct_entailment",
    ),
    BenchmarkExample(
        id="sf-22",
        claim=(
            "Direct Preference Optimization requires training a separate neural reward model "
            "and running proximal policy optimization sampling loops at every step."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="arxiv:2305.18290",
        gold_quote=(
            "By applying a change-of-variables parameterization between the optimal policy "
            "and the ground-truth reward function, Direct Preference Optimization (DPO) "
            "optimizes the language model directly on human preference pairs using a simple "
            "binary cross-entropy objective without explicit reward modeling or reinforcement "
            "learning sampling loops."
        ),
        category="Alignment",
        difficulty="standard",
        challenge_type="negation_antonym_reversal",
    ),
    BenchmarkExample(
        id="sf-23",
        claim=(
            "DPO remains completely invariant and immune to out-of-distribution preference "
            "shifts even when offline preference data lacks policy coverage."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="arxiv:2305.18290",
        gold_quote=(
            "However, DPO performance degrades on out-of-distribution preference shifts "
            "when offline preference data lacks sufficient coverage of the current policy "
            "distribution."
        ),
        category="Alignment",
        difficulty="adversarial",
        challenge_type="negation_antonym_reversal",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-24",
        claim=(
            "Mamba achieves linear time complexity O(L) in sequence length and 5x higher "
            "inference throughput than Transformers of similar size."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2312.00752",
        gold_quote=(
            "Mamba achieves linear time complexity O(L) in sequence length L and 5x higher "
            "inference throughput than Transformers of similar size, matching a Transformer "
            "twice its size on language modeling."
        ),
        category="Sequence Modeling",
        difficulty="hard",
        challenge_type="noisy_pdf_latex_grounding",
    ),
    BenchmarkExample(
        id="sf-25",
        claim=(
            "Fixed-state recurrent compression in Mamba achieves superior exact in-context "
            "copying recall compared to full softmax attention on synthetic needle lookup "
            "tasks exceeding 64k tokens."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="arxiv:2312.00752",
        gold_quote=(
            "However, fixed-state recurrent compression exhibits lower exact in-context "
            "copying recall than full softmax attention on synthetic needle-in-a-haystack "
            "lookup tasks exceeding 64k tokens."
        ),
        category="Sequence Modeling",
        difficulty="adversarial",
        challenge_type="negation_antonym_reversal",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-26",
        claim=(
            "AlphaFold 3 achieves 76.4% pose accuracy (RMSD < 2.0 angstrom) on the "
            "PoseBusters protein-ligand benchmark, outperforming AutoDock Vina."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="openalex:W4396712345",
        gold_quote=(
            "By replacing the AlphaFold 2 Evoformer and invariant point attention structure "
            "module with a lighter Pairformer trunk and a direct atomic diffusion module, "
            "AlphaFold 3 achieves 76.4% pose accuracy (RMSD < 2.0 angstrom) on the PoseBusters "
            "protein-ligand benchmark, outperforming AutoDock Vina and RoseTTAFold All-Atom."
        ),
        category="Structural Biology",
        difficulty="hard",
        challenge_type="noisy_pdf_latex_grounding",
    ),
    BenchmarkExample(
        id="sf-27",
        claim=(
            "AlphaFold 3 diffusion generation never produces stereochemical chirality "
            "violations or spurious alpha-helical hallucinations in disordered regions."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="openalex:W4396712345",
        gold_quote=(
            "However, diffusion generation occasionally produces stereochemical chirality "
            "violations and spurious alpha-helical hallucinations in intrinsically disordered "
            "protein regions unless cross-distillation with AlphaFold-Multimer templates "
            "is applied."
        ),
        category="Structural Biology",
        difficulty="adversarial",
        challenge_type="negation_antonym_reversal",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-28",
        claim=(
            "Canonical BE3 cytosine base editors induce tens of thousands of C-to-U "
            "off-target RNA edits across 38% of expressed transcripts in human cells."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="s2:8b7c6d5e4f3a",
        gold_quote=(
            "Using whole-transcriptome RNA sequencing across human HEK293T and primary "
            "hepatocyte lines, we demonstrate that canonical BE3 cytosine base editors "
            "induce tens of thousands of C-to-U off-target RNA edits across 38% of "
            "expressed transcripts, independent of guide RNA complementarity."
        ),
        category="Genome Editing",
        difficulty="standard",
        challenge_type="direct_entailment",
    ),
    BenchmarkExample(
        id="sf-29",
        claim=(
            "SECURE mutations (R33A/K34A) in rAPOBEC1 reduce transcriptome-wide off-target "
            "RNA deamination by more than 90-fold while preserving on-target DNA editing."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="s2:8b7c6d5e4f3a",
        gold_quote=(
            "Structure-guided SECURE mutations (R33A/K34A) in rAPOBEC1 reduce "
            "transcriptome-wide off-target RNA deamination by more than 90-fold while "
            "preserving on-target genomic DNA editing efficiency."
        ),
        category="Genome Editing",
        difficulty="hard",
        challenge_type="multi_hop_synthesis",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-30",
        claim=(
            "Canonical BE3 cytosine base editors produce zero off-target RNA edits because "
            "rAPOBEC1 deamination strictly requires guide RNA complementarity."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="s2:8b7c6d5e4f3a",
        gold_quote=(
            "Using whole-transcriptome RNA sequencing across human HEK293T and primary "
            "hepatocyte lines, we demonstrate that canonical BE3 cytosine base editors "
            "induce tens of thousands of C-to-U off-target RNA edits across 38% of "
            "expressed transcripts, independent of guide RNA complementarity."
        ),
        category="Genome Editing",
        difficulty="hard",
        challenge_type="negation_antonym_reversal",
    ),
    # --- 31 to 45: Quantum, Materials, Weather, Multi-Hop, & Numerical Threshold Traps ---
    BenchmarkExample(
        id="sf-31",
        claim=(
            "On the 105-qubit Willow processor, increasing surface code distance by 2 "
            "suppresses logical error rate per cycle by a factor of 2.14 and achieves a "
            "logical lifetime 2.4x longer than the best physical qubit."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="openalex:W4401239876",
        gold_quote=(
            "Using the 105-qubit superconducting Willow processor, we demonstrate a "
            "distance-7 surface code memory whose logical error rate per cycle is suppressed "
            "by a factor of Lambda = 2.14 +- 0.02 each time the code distance increases by 2 "
            "(from d=3 to d=5 and d=7), achieving a logical lifetime 2.4x longer than the "
            "best constituent physical qubit."
        ),
        category="Quantum Error Correction",
        difficulty="hard",
        challenge_type="noisy_pdf_latex_grounding",
    ),
    BenchmarkExample(
        id="sf-32",
        claim=(
            "Increasing surface code distance from d=3 to d=7 on the Willow processor "
            "causes logical error rates per cycle to double rather than decrease."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="openalex:W4401239876",
        gold_quote=(
            "Fault-tolerant quantum computation requires physical gate error rates to operate "
            "below the threshold of a quantum error-correcting code so that logical error "
            "rates decrease exponentially with code distance d."
        ),
        category="Quantum Error Correction",
        difficulty="standard",
        challenge_type="negation_antonym_reversal",
    ),
    BenchmarkExample(
        id="sf-33",
        claim=(
            "Ionizing radiation and cosmic-ray impact events generate correlated "
            "quasiparticle bursts that establish a surface code error floor of 0.1 parts "
            "per billion per cycle unless gap-engineering traps are incorporated."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="openalex:W4401239876",
        gold_quote=(
            "However, high-energy ionizing radiation and cosmic-ray impact events generate "
            "correlated quasiparticle bursts across the substrate that temporarily establish "
            "an error floor around 10^-10 per cycle unless gap-engineering traps are "
            "incorporated."
        ),
        category="Quantum Error Correction",
        difficulty="adversarial",
        challenge_type="multi_hop_synthesis",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-34",
        claim=(
            "Ring Attention overlaps peer-to-peer key-value block communication with "
            "blockwise attention computation to scale context length linearly with "
            "accelerator count."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2310.01889",
        gold_quote=(
            "When block size exceeds the arithmetic intensity threshold of the interconnect, "
            "Ring Attention scales context length linearly with the number of accelerators "
            "with zero communication overhead beyond standard single-device FlashAttention."
        ),
        category="Distributed Systems ML",
        difficulty="standard",
        challenge_type="direct_entailment",
    ),
    BenchmarkExample(
        id="sf-35",
        claim=(
            "GNoME identifies 2.2 million crystal structures below the prior convex hull "
            "of stability, including 381,000 novel stable inorganic materials."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="openalex:W4389001122",
        gold_quote=(
            "Across iterative structural and compositional filtrations, GNoME identifies "
            "2.2 million crystal structures below the prior convex hull of stability, "
            "including 381,000 high-confidence novel stable materials—an order-of-magnitude "
            "expansion of humanity's known stable inorganic compounds."
        ),
        category="Materials Science",
        difficulty="hard",
        challenge_type="noisy_pdf_latex_grounding",
    ),
    BenchmarkExample(
        id="sf-36",
        claim=(
            "GNoME identified fewer than 1,500 stable crystal structures below the convex "
            "hull of stability across its active-learning filtrations."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="openalex:W4389001122",
        gold_quote=(
            "Across iterative structural and compositional filtrations, GNoME identifies "
            "2.2 million crystal structures below the prior convex hull of stability, "
            "including 381,000 high-confidence novel stable materials—an order-of-magnitude "
            "expansion of humanity's known stable inorganic compounds."
        ),
        category="Materials Science",
        difficulty="hard",
        challenge_type="numerical_threshold_trap",
    ),
    BenchmarkExample(
        id="sf-37",
        claim=(
            "GraphCast generates a 10-day global weather forecast at 0.25-degree resolution "
            "in under 60 seconds on a single Cloud TPU v4 device and outperforms ECMWF HRES "
            "on 90.3% of verification targets."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="openalex:W4388554433",
        gold_quote=(
            "GraphCast generates a 10-day global forecast in under 60 seconds on a single "
            "Cloud TPU v4 device and outperforms the European Centre for Medium-Range "
            "Weather Forecasts (ECMWF) HRES deterministic system on 90.3% of 1,380 "
            "verification targets."
        ),
        category="Earth & Climate ML",
        difficulty="standard",
        challenge_type="direct_entailment",
    ),
    BenchmarkExample(
        id="sf-38",
        claim=(
            "Deterministic mean-squared-error training in GraphCast eliminates spectral "
            "smoothing and sharpens fine-scale convective extremes beyond 7 days."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="openalex:W4388554433",
        gold_quote=(
            "However, deterministic mean-squared-error training causes progressive spectral "
            "smoothing of fine-scale convective extremes at lead times beyond 7 days."
        ),
        category="Earth & Climate ML",
        difficulty="hard",
        challenge_type="negation_antonym_reversal",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-39",
        claim=(
            "Adding a second nicking guide RNA in the PE3 prime editing system increases "
            "point-mutation efficiency by 2.6-fold to 4.2-fold but raises on-target indel "
            "byproduct rates to between 4.8% and 11.3%."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="s2:1a2b3c4d5e6f",
        gold_quote=(
            "However, dual opposing nicks in PE3 generate transient double-stranded breaks "
            "that elevate undesired on-target insertion-deletion (indel) byproduct rates "
            "from < 0.5% in PE2 to between 4.8% and 11.3% in PE3."
        ),
        category="Genome Editing",
        difficulty="hard",
        challenge_type="multi_hop_synthesis",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-40",
        claim=(
            "Dual opposing nicks in the PE3 prime editing system reduce on-target indel "
            "byproduct rates to below 0.05% compared to PE2."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="s2:1a2b3c4d5e6f",
        gold_quote=(
            "However, dual opposing nicks in PE3 generate transient double-stranded breaks "
            "that elevate undesired on-target insertion-deletion (indel) byproduct rates "
            "from < 0.5% in PE2 to between 4.8% and 11.3% in PE3."
        ),
        category="Genome Editing",
        difficulty="adversarial",
        challenge_type="numerical_threshold_trap",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-41",
        claim=(
            "Placing the gold passage in the middle of a long multi-document context "
            "degrades language model answering accuracy by up to 22.8 percentage points "
            "compared to primacy or recency positions."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="arxiv:2402.09911",
        gold_quote=(
            "Across multi-document question answering and key-value retrieval tasks "
            "containing 20 to 50 passages, accuracy follows a U-shaped curve: models "
            "achieve highest accuracy when relevant evidence is placed at the very "
            "beginning (primacy) or very end (recency) of the input context, whereas "
            "placing the gold passage in the middle of the context degrades answering "
            "accuracy by up to 22.8 percentage points, falling below closed-book "
            "performance on several models."
        ),
        category="Long-Context Reasoning",
        difficulty="standard",
        challenge_type="direct_entailment",
    ),
    BenchmarkExample(
        id="sf-42",
        claim=(
            "Language models achieve their highest multi-document question answering "
            "accuracy when relevant evidence is buried in the exact middle of a 50-passage "
            "context window."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="arxiv:2402.09911",
        gold_quote=(
            "Across multi-document question answering and key-value retrieval tasks "
            "containing 20 to 50 passages, accuracy follows a U-shaped curve: models "
            "achieve highest accuracy when relevant evidence is placed at the very "
            "beginning (primacy) or very end (recency) of the input context, whereas "
            "placing the gold passage in the middle of the context degrades answering "
            "accuracy by up to 22.8 percentage points, falling below closed-book "
            "performance on several models."
        ),
        category="Long-Context Reasoning",
        difficulty="standard",
        challenge_type="negation_antonym_reversal",
    ),
    BenchmarkExample(
        id="sf-43",
        claim=(
            "Genome-scale Perturb-seq combines multiplexed CRISPRi screening with droplet "
            "single-cell RNA sequencing across 2.5 million cells to profile transcriptional "
            "signatures for 9,867 expressed genes."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="s2:7e6f5a4b3c2d",
        gold_quote=(
            "By combining multiplexed CRISPR interference (CRISPRi) library screening with "
            "droplet-based single-cell RNA sequencing across 2.5 million K562 and RPE1 "
            "cells, genome-scale Perturb-seq profiles transcriptional signatures for 9,867 "
            "expressed genes."
        ),
        category="Functional Genomics",
        difficulty="standard",
        challenge_type="direct_entailment",
    ),
    BenchmarkExample(
        id="sf-44",
        claim=(
            "Up to 31% of inline citations in standard RAG outputs fail strict entailment "
            "checks because models attribute parametric statements to passages that do not "
            "mention them."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="openalex:W4387129001",
        gold_quote=(
            "Furthermore, language models frequently fabricate citations by attributing true "
            "parametric statements to retrieved passages that do not mention them; up to 31% "
            "of inline citations in standard RAG outputs fail strict entailment checks."
        ),
        category="Robustness",
        difficulty="hard",
        challenge_type="multi_hop_synthesis",
        multi_hop=True,
    ),
    BenchmarkExample(
        id="sf-45",
        claim=(
            "Adding an adversarial Skeptic agent that queries for counter-evidence and "
            "forces draft revision reduces unsupported final claims by 61.5% at a median "
            "latency overhead of 1.38x."
        ),
        gold_verdict=Verdict.SUPPORTED,
        gold_paper_id="s2:9a8b7c6d5e4f",
        gold_quote=(
            "Adding an adversarial Skeptic agent that queries for counter-evidence and "
            "forces draft revision reduces unsupported final claims by an additional 61.5% "
            "at a median latency overhead of 1.38x."
        ),
        category="Verification",
        difficulty="hard",
        challenge_type="multi_hop_synthesis",
        multi_hop=True,
    ),
    # --- 46 to 60: Hard Lexical Shifts, Subtle Numerical Traps, & Ambiguous NEI Cases ---
    BenchmarkExample(
        id="sf-46",
        claim=(
            "Automated factuality verifiers that pair nonparametric passage retrieval with "
            "entailment checking achieve a Pearson correlation of r = 0.21 with human "
            "factuality judgements."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="arxiv:2305.14627",
        gold_quote=(
            "Automated verifiers that combine nonparametric passage retrieval with "
            "entailment checking correlate with human factuality scores at r = 0.94."
        ),
        category="Evaluation",
        difficulty="hard",
        challenge_type="numerical_threshold_trap",
    ),
    BenchmarkExample(
        id="sf-47",
        claim=(
            "FlashAttention relies on low-rank randomized kernel approximations to reduce "
            "attention complexity from quadratic to sub-linear."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="openalex:W4281990123",
        gold_quote=(
            "FlashAttention trains BERT-large 15% faster and GPT-2 3.0x faster than standard "
            "PyTorch implementations without any approximation."
        ),
        category="Systems ML",
        difficulty="hard",
        challenge_type="lexical_terminology_shift",
    ),
    BenchmarkExample(
        id="sf-48",
        claim=(
            "Mistral 7B uses a sliding window attention span of 32,768 tokens and lags "
            "behind Llama 2 13B on mathematics and code generation benchmarks."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="arxiv:2310.06825",
        gold_quote=(
            "Mistral 7B leverages grouped-query attention (GQA) for faster decoding and "
            "sliding window attention (SWA) with a window size of 4,096 tokens to handle "
            "sequences of arbitrary length with a reduced KV-cache memory footprint."
        ),
        category="Systems ML",
        difficulty="hard",
        challenge_type="numerical_threshold_trap",
    ),
    BenchmarkExample(
        id="sf-49",
        claim=(
            "Combining dense vector retrieval with sentence-level rationale selection on "
            "SciFact reduces label Macro-F1 by 9.8 points compared to BM25 alone."
        ),
        gold_verdict=Verdict.REFUTED,
        gold_paper_id="arxiv:2004.14906",
        gold_quote=(
            "Combining dense retrieval with sentence-level rationale selection boosts "
            "label Macro-F1 by 9.8 points."
        ),
        category="Retrieval",
        difficulty="hard",
        challenge_type="negation_antonym_reversal",
    ),
    BenchmarkExample(
        id="sf-50",
        claim=(
            "GNoME crystal discovery guarantees room-temperature ambient-pressure "
            "superconductivity above 320 Kelvin in all 381,000 predicted inorganic compounds."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Materials Extrapolation Trap",
        difficulty="adversarial",
        challenge_type="scope_modality_extrapolation",
        trap_ungrounded_quote=True,
    ),
    BenchmarkExample(
        id="sf-51",
        claim=(
            "AlphaFold 3 completely replaces wet-lab X-ray crystallography and achieves "
            "100% FDA Phase III approval rates for de novo small-molecule kinase inhibitors."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Clinical Extrapolation Trap",
        difficulty="adversarial",
        challenge_type="scope_modality_extrapolation",
        trap_ungrounded_quote=True,
    ),
    BenchmarkExample(
        id="sf-52",
        claim=(
            "The 105-qubit Willow superconducting processor breaks 2048-bit RSA encryption "
            "in 4.2 milliseconds using Shor's factoring algorithm."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Quantum Extrapolation Trap",
        difficulty="adversarial",
        challenge_type="scope_modality_extrapolation",
        trap_ungrounded_quote=True,
    ),
    BenchmarkExample(
        id="sf-53",
        claim=(
            "FlashAttention reduces GPU high-bandwidth memory traffic by 64% specifically "
            "when compiled for AMD MI300X FP8 tensor cores."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Hardware Specificity Boundary",
        difficulty="hard",
        challenge_type="ambiguous_insufficient_evidence",
    ),
    BenchmarkExample(
        id="sf-54",
        claim=(
            "Mamba selective state space models outperform Transformer baselines on "
            "multimodal 4K video optical flow estimation benchmarks."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Modality Boundary",
        difficulty="hard",
        challenge_type="ambiguous_insufficient_evidence",
    ),
    BenchmarkExample(
        id="sf-55",
        claim=(
            "Chinchilla scaling laws demonstrate that mixture-of-experts routing with 64 "
            "experts halves optimal training token requirements at fixed FLOPs."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Architectural Boundary",
        difficulty="hard",
        challenge_type="ambiguous_insufficient_evidence",
    ),
    BenchmarkExample(
        id="sf-56",
        claim=(
            "GraphCast medium-range weather forecasts reduce Atlantic hurricane landfall "
            "evacuation costs by $4.8 billion per season."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Economic Extrapolation",
        difficulty="standard",
        challenge_type="ambiguous_insufficient_evidence",
    ),
    BenchmarkExample(
        id="sf-57",
        claim=(
            "Ring Attention requires InfiniBand NDR 400Gb/s optical interconnects and fails "
            "to converge over RoCEv2 Ethernet fabrics."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Systems Boundary",
        difficulty="hard",
        challenge_type="ambiguous_insufficient_evidence",
    ),
    BenchmarkExample(
        id="sf-58",
        claim=(
            "Genome-scale Perturb-seq in primary human cortical organoids identifies 412 "
            "autism spectrum disorder risk enhancers."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Cell-Type Extrapolation",
        difficulty="hard",
        challenge_type="ambiguous_insufficient_evidence",
    ),
    BenchmarkExample(
        id="sf-59",
        claim=(
            "Speculative decoding with a 125M draft model increases GPU thermal design "
            "power consumption by 48% on mobile edge NPUs."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Hardware Specificity Boundary",
        difficulty="standard",
        challenge_type="ambiguous_insufficient_evidence",
    ),
    BenchmarkExample(
        id="sf-60",
        claim=(
            "LoRA rank-8 adaptation on 70B code generation models degrades pass@1 on "
            "HumanEval by 11.4% compared to full parameter fine-tuning."
        ),
        gold_verdict=Verdict.NOT_ENOUGH_INFO,
        gold_paper_id=None,
        gold_quote=None,
        category="Benchmark Specificity Boundary",
        difficulty="adversarial",
        challenge_type="ambiguous_insufficient_evidence",
    ),
)


def load_benchmark_jsonl(path: Path) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            examples.append(BenchmarkExample.model_validate(json.loads(line)))
    return examples


def dump_benchmark_jsonl(
    path: Path, examples: tuple[BenchmarkExample, ...] = SCIFACT_BENCHMARK
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [ex.model_dump_json() for ex in examples]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
