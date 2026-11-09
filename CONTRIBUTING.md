# Contributing to Aetheris

Thank you for your interest in contributing to **Aetheris**! We welcome bug reports, new literature retrieval sources, evaluation datasets, and architectural improvements.

---

## Development Setup

### Prerequisites
- **Python 3.12+** (managed via [`uv`](https://docs.astral.sh/uv/))
- **Node.js 20+** (for the Next.js dashboard)
- Git

### Installation
```bash
git clone https://github.com/anargadirli/aetheris.git
cd aetheris
make install
```

---

## Quality Standards

All pull requests must satisfy our automated verification pipeline:

```bash
make check
```

`make check` executes:
1. `uv run ruff check .` and `uv run ruff format --check .`
2. `uv run mypy backend/src` (100% strict typechecking with zero `Any` escapes)
3. `uv run pytest -v --cov=aetheris` (minimum 90% test coverage)
4. `cd frontend && npm run typecheck && npm run build`

---

## Running Benchmarks & Offline Demos

To verify that your changes do not degrade scientific claim verification accuracy:

```bash
uv run aetheris eval
uv run aetheris research "Do retrieval-augmented generation models reduce factual hallucinations compared to closed-book LLMs?" --demo
```

---

## Coding Guidelines

1. **Deterministic by Default**: Any core agent or retrieval logic must be testable offline without external API keys using `FakeLLM`, `DemoLLM`, or recorded `respx` HTTP mocks.
2. **Lean JSON Schemas**: Pydantic models used for LLM constrained decoding should be flat and free of deep recursion or character/array length constraints to keep grammar compilation overhead low.
3. **Immutability**: Domain models (`Paper`, `Chunk`, `EvidenceSpan`, `GraphNode`) are frozen. State transitions inside the orchestrator return fresh state objects via `model_copy(update=...)`.
4. **Mechanical Grounding**: Never trust an LLM to quote evidence verbatim. Supporting and refuting evidence spans must always pass `locate_quote`.
