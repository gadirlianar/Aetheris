# Aetheris Scientific Claim Verification & Retrieval Ablation Study

| Configuration | Retrieval | Grounding Guard | Verdict Acc | Verdict Macro-F1 | Recall@K | MRR | Mean Span IoU | ECE (↓) | Brier (↓) | Hallucinations Caught |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **dense_only** (Dense Vector Cosine Only (k=2)) | `dense` | Yes | 73.3% | **63.0%** | 69.2% | 0.692 | 69.2% | 0.203 | 0.139 | 2 |
| **bm25_only** (Sparse Lexical BM25+ Only (k=2)) | `bm25` | Yes | 80.0% | **75.3%** | 76.9% | 0.769 | 76.9% | 0.159 | 0.102 | 2 |
| **hybrid_2way_rrf** (2-Way Hybrid RRF (Dense + BM25)) | `hybrid` | Yes | 93.3% | **86.7%** | 92.3% | 0.923 | 92.3% | 0.084 | 0.041 | 2 |
| **no_grounding_guard** (3-Way GraphRAG RRF without Quote Grounding Guard) | `graph_hybrid` | No | 86.7% | **63.6%** | 100.0% | 1.000 | 100.0% | 0.028 | 0.102 | 0 |
| **full_aetheris** (Full Aetheris (3-Way GraphRAG RRF + Mechanical Quote Grounding)) | `graph_hybrid` | Yes | 100.0% | **100.0%** | 100.0% | 1.000 | 100.0% | 0.117 | 0.017 | 2 |
