.PHONY: install lint typecheck test eval demo serve web check

install:
	uv sync
	cd frontend && npm install

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy backend/src
	cd frontend && npm run typecheck

test:
	uv run pytest -v --cov=aetheris

eval:
	uv run aetheris eval -o docs/technical-report/latest_ablation.md --json docs/technical-report/latest_ablation.json

demo:
	uv run aetheris research "Do retrieval-augmented generation models reduce factual hallucinations compared to closed-book LLMs?" --demo

serve:
	uv run aetheris serve --host 127.0.0.1 --port 8000

web:
	cd frontend && npm run dev

check: lint typecheck test
	cd frontend && npm run build

