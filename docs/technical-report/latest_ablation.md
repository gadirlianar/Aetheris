# Aetheris Scientific Claim Verification & Retrieval Ablation Study (N=60 Claims)

| Configuration | Retrieval | Guard | Verdict Acc (95% CI) | Verdict Macro-F1 (95% CI) | Recall@K | MRR | Mean Span IoU | ECE (↓) | Brier (↓) | Hallucinations Caught |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **dense_only** (Dense Subword/N-Gram Vector Cosine Only (k=1)) | `dense` | Yes | 71.7% `[60.0%, 83.3%]` | **67.9%** `[53.7%, 79.7%]` | 83.0% | 0.830 | 63.6% | 0.114 | 0.177 | 5 |
| **bm25_only** (Sparse Lexical Okapi BM25+ Only (k=1)) | `bm25` | Yes | 75.0% `[63.3%, 85.0%]` | **71.5%** `[57.8%, 82.4%]` | 85.1% | 0.851 | 67.8% | 0.080 | 0.166 | 5 |
| **hybrid_2way_rrf** (2-Way Hybrid RRF (Dense + BM25, k=2)) | `hybrid` | Yes | 81.7% `[71.7%, 90.0%]` | **80.5%** `[69.6%, 89.1%]` | 87.2% | 0.872 | 74.2% | 0.070 | 0.132 | 5 |
| **no_grounding_guard** (3-Way GraphRAG RRF without Quote Grounding Guard (k=4)) | `graph_hybrid` | No | 80.0% `[70.0%, 88.3%]` | **77.8%** `[65.7%, 87.2%]` | 97.9% | 0.915 | 89.1% | 0.072 | 0.160 | 0 |
| **full_aetheris** (Full Aetheris (3-Way GraphRAG RRF + Mechanical Quote Grounding, k=4)) | `graph_hybrid` | Yes | 88.3% `[80.0%, 95.0%]` | **88.0%** `[79.0%, 95.6%]` | 97.9% | 0.915 | 89.1% | 0.041 | 0.108 | 5 |

## Confusion Matrix (`full_aetheris`, N=60)

| Gold \ Predicted | `supported` | `refuted` | `not_enough_info` |
|---|---:|---:|---:|
| **`supported`** | 26 | 2 | 0 |
| **`refuted`** | 3 | 14 | 2 |
| **`not_enough_info`** | 0 | 0 | 13 |

## Error Taxonomy & Failure Mode Breakdown Across Ablations

| Failure Mode Category | `dense_only` | `bm25_only` | `hybrid_2way_rrf` | `no_grounding_guard` | `full_aetheris` |
|---|---:|---:|---:|---:|---:|
| `unretrieved_gold_evidence` | 7 | 6 | 4 | 1 | 1 |
| `ungrounded_hallucination_accepted` | 0 | 0 | 0 | 5 | 0 |
| `numerical_threshold_mismatch` | 3 | 3 | 1 | 1 | 1 |
| `negation_polarity_reversal` | 4 | 4 | 3 | 3 | 3 |
| `multi_hop_synthesis_gap` | 2 | 1 | 1 | 1 | 1 |
| `scope_or_hedging_ambiguity` | 1 | 1 | 2 | 1 | 1 |
