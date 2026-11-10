"""Runtime configuration (environment variables and an optional ``.env`` file)."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from aetheris.core.models import RetrievalMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AETHERIS_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- models & tiered routing
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "AETHERIS_GEMINI_API_KEY")
    )
    llm_model: str = "gemini-3.8-flash"
    fast_model: str = "gemini-3.5-flash-lite"
    reasoning_model: str = "gemini-3.8-flash"
    thinking_level: str | None = "low"
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 768

    # --- sources
    openalex_email: str | None = Field(
        default=None, description="Sent as `mailto` to join OpenAlex's polite pool."
    )
    openalex_api_key: SecretStr | None = None
    semantic_scholar_api_key: SecretStr | None = None
    enable_semantic_scholar: bool = True
    papers_per_query: int = 8
    source_timeout_s: float = 30.0

    # --- research, verification & concurrency shape
    retrieval_mode: RetrievalMode = "graph_hybrid"
    enable_grounding_guard: bool = True
    max_skeptic_rounds: int = 1
    skeptic_min_confidence: float = 0.72
    max_sub_questions: int = 4
    max_queries_per_sub_question: int = 3
    evidence_per_sub_question: int = 8
    evidence_per_claim: int = 6
    llm_concurrency: int = 8
    max_concurrent_claims: int = 5
    rate_limit_rps: float = 25.0
    rate_limit_burst: int = 10

    # --- budget & cost limits
    max_llm_calls: int = 200
    max_total_tokens: int = 2_000_000
    max_seconds: float = 900.0
    max_cost_usd: float = 5.0

    # --- storage
    data_dir: Path = Path(".aetheris")

    @property
    def trace_db(self) -> Path:
        return self.data_dir / "traces.sqlite3"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"
