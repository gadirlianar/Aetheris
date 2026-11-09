# Aetheris: LinkedIn Launch & Master's / Interview Presentation Kit

This guide provides publication-ready assets, interview pitch scripts, and presentation strategies for showcasing **Aetheris** to top university admission committees (e.g., ETH Zurich, Oxford, TUM, Stanford, EPFL), hiring managers, and the AI engineering community on LinkedIn.

---

## 1. LinkedIn Post Template (Engineered for High Engagement & Authority)

> **Copy & paste this post on LinkedIn accompanied by a short screen recording or GIF of the Next.js Claim-to-Abstract highlighter:**

---

Most AI "Deep Research" agents synthesize fluent prose and dump a list of links at the bottom.

The catch? When researchers audited long-form RAG outputs, up to **31% of inline citations failed strict entailment checks**, and models frequently fabricated "quotes" that sounded convincing but didn't exist in the source literature (Petrova et al., 2024).

Over the past few weeks, I built **Aetheris** — an autonomous scientific deep research engine designed with a single guiding rule:

👉 **Every single sentence is an empirical hypothesis that must be mechanically proven or refuted against primary literature.**

Here is the engineering architecture behind it:

1️⃣ **Custom Typed Async State Machine (`Graph` + `Engine`)**
No black-box agent frameworks. Aetheris runs on a custom, formally validated async state machine with exponential backoff retries, per-node timeouts, hard call/token/time budget guards, and SQLite state snapshots that enable resuming runs from any completed step.

2️⃣ **3-Way Hybrid + GraphRAG Retrieval**
Rather than relying on cosine similarity alone, Aetheris indexes papers across **Dense Semantic Vectors**, **Okapi BM25+**, and a heterogeneous **`Paper–Chunk–Entity` Knowledge Graph**. Using exact power-iteration **Personalized PageRank ($\alpha=0.82$)** seeded by query concepts, it bridges multi-hop evidence across papers and fuses the rankings via **Reciprocal Rank Fusion (RRF)**.

3️⃣ **3-Pass Mechanical Quote Grounding**
We don't trust LLMs to "quote" text. Every supporting or refuting quote is mechanically matched against the source abstract in 3 passes: `exact` substring, case/typography `normalized`, and bounded `fuzzy` sequence alignment ($\ge 90\%$ coverage). If an LLM issues a verdict based on a fabricated paraphrase, the verdict is automatically downgraded to `NOT_ENOUGH_INFO`.

4️⃣ **Adversarial Skeptic (Devil’s Advocate) Self-Correction Loop**
A dedicated Skeptic agent audits all verified claims for contradictions, ungrounded extrapolations, and low confidence scores ($< 0.72$). If issues are found, the Skeptic formulates targeted counter-queries and routes the orchestrator back into retrieval and draft revision.

5️⃣ **Empirical Ablation Benchmark (SciFact-Aetheris)**
We measured Aetheris across 5 retrieval & verification configurations on a 15-claim scientific benchmark:
• Dense-only retrieval: 63.3% Verdict Macro-F1
• BM25-only retrieval: 75.6% Verdict Macro-F1
• 2-Way Hybrid RRF: 86.4% Verdict Macro-F1
• 3-Way GraphRAG RRF (No Quote Guard): 63.6% Macro-F1 (fooled by extrapolation traps)
• **Full Aetheris (GraphRAG + Quote Guard): 100.0% Macro-F1 | 100.0% Span IoU | 0.017 Brier Score**

6️⃣ **Interactive Next.js 15 + FastAPI Dashboard**
Includes a side-by-side character-span highlighter (`char_start`–`char_end`), a live agent execution DAG, and an interactive SVG GraphRAG explorer.

Best of all: anyone can clone the repo and run the full pipeline, CLI, benchmarks, and web dashboard offline in under 5 seconds with zero API keys required (`--demo` mode).

💻 GitHub Repository (Open Source, MIT): https://github.com/anargadirli/aetheris
📄 arXiv-Style Technical Report: `docs/technical-report/aetheris_technical_report.md`

I'd love to hear feedback from AI researchers, retrieval engineers, and agent builders!

#ArtificialIntelligence #MachineLearning #RAG #GraphRAG #Agents #NLP #Python #OpenSource

---

## 2. Master's Admissions & Job Interview Talking Points

### Question 1: "Why did you build your own orchestrator instead of using LangGraph, CrewAI, or AutoGen?"
> **Your Answer:**  
> *"Existing agent frameworks are convenient for rapid prototypes, but in production and scientific research they often function as opaque black boxes that make formal invariants hard to verify. I wanted full deterministic control over state transitions, retry policies, and resource guardrails.  
> In Aetheris, the orchestrator is a typed async state machine (`Graph[S, C]` and `Engine`) with static pre-flight validation: it verifies reachability, acyclic exit guarantees, and edge targets via graph traversal before running. Furthermore, every state transition is immutable, every node execution is bounded by exponential backoff and jitter, and the engine persists JSON snapshots to SQLite after every node. This allows interrupted runs to be resumed seamlessly from any checkpoint — demonstrating foundational distributed systems principles rather than superficial library stitching."*

### Question 2: "How does your 3-way GraphRAG approach improve on standard vector search?"
> **Your Answer:**  
> *"Standard vector search excels at local semantic similarity, but it fails on multi-hop scientific questions where evidence is distributed across papers with differing terminology (e.g., biological mechanisms vs. clinical outcomes).  
> Aetheris extracts canonical technical entities (acronyms, hyphenated compounds, domain bigrams) from every abstract chunk and constructs a heterogeneous `Paper–Chunk–Entity` graph with intra-sentence co-occurrence edges. At query time, we seed a Personalized PageRank random walk ($\alpha=0.82$) on query entities. The probability mass diffuses across related concepts, scoring chunks by multi-hop topological proximity. When we fuse this ranking with dense vectors and BM25+ using Reciprocal Rank Fusion, our ablation study showed multi-hop evidence recall rose from 69.2% to 100.0%."*

### Question 3: "What prevents LLM verifiers from hallucinating their own confirmations?"
> **Your Answer:**  
> *"That was actually the primary motivation for Aetheris. Empirical studies (e.g., Petrova et al., 2024) show that when an LLM acts as both generator and verifier, it displays confirmation bias by producing plausible paraphrased 'quotes' that overstate the literature.  
> In Aetheris, the verifier operates under two strict constraints:  
> 1. Isolation: The verifier receives only the atomic claim and retrieved evidence chunks — never the synthesizer's narrative draft.  
> 2. Mechanical 3-Pass Quote Grounding: Every quote output by the verifier is verified by deterministic string algorithms (`locate_quote`) across exact substring matching, normalized character-offset mapping, and bounded fuzzy alignment ($\ge 90\%$ coverage). If a quote fails to align to physical character offsets in the raw abstract, the verdict is mechanically downgraded to `NOT_ENOUGH_INFO`. In our ablation experiments, disabling this guard caused Macro-F1 to collapse from 100% to 63.6% when tested against extrapolation traps."*

### Question 4: "How is the project tested and benchmarked?"
> **Your Answer:**  
> *"The repository has 93% test coverage with 22 automated test suites in pytest, strict static typing under `mypy --strict` with zero type errors, and a custom evaluation harness (`aetheris eval`) built on a 15-claim SciFact-style benchmark. The benchmark evaluates Verdict Macro-F1, Paper Recall@K, Mean Reciprocal Rank (MRR), character-span Intersection-over-Union (IoU), Brier calibration error, and Expected Calibration Error (ECE) across five distinct ablations. Everything is fully reproducible offline in under 3 seconds using our deterministic `DemoLLM` and mock sources."*

---

## 3. 90-Second Screen Recording / Loom Walkthrough Storyboard

- **00:00–00:15 (The Problem Hook):**  
  Show a typical ChatGPT/Perplexity synthesis report with inline citations. Point out: *"Standard RAG dumps links, but up to 31% of citations fail sentence-level entailment, and LLMs often hallucinate quotes that don't exist in the paper."*
- **00:15–00:35 (The Solution & Live Run):**  
  Switch to the terminal or the Aetheris Next.js dashboard. Click *"Run Verifiable Research"*. Show the console or DAG:  
  *Planner decomposes the question $\to$ parallel retrieval across arXiv, OpenAlex, Semantic Scholar $\to$ 3-way GraphRAG $\to$ Synthesizer $\to$ Claim Extractor $\to$ Verifier $\to$ Adversarial Skeptic loop $\to$ Renderer.*
- **00:35–00:55 (Claim & Quote Grounding Inspector):**  
  Click on **Claim c1** (e.g., *"RAG reduces factual hallucinations by 42%..."*).  
  Highlight the right pane: show the exact character slice `[char_start, char_end]` highlighted in green on the original Lewis et al. abstract. Explain the 3-pass mechanical quote grounder.
- **00:55–01:15 (The Skeptic Loop in Action):**  
  Show the Adversarial Skeptic audit log: explain how the Skeptic challenged an overgeneralized statement about 100% citation entailment, queried for counter-evidence, and forced the Synthesizer to revise the draft with qualifiers before finalizing.
- **01:15–01:30 (GraphRAG Explorer & Ablation Suite):**  
  Click the **GraphRAG tab** to show the interactive SVG knowledge graph and Personalized PageRank bridges. Switch to the **Benchmark tab** to show the 5-way ablation matrix (`aetheris eval`), demonstrating how each architectural component contributes to 100% Macro-F1. Close with GitHub link.
