"""Runtime configuration, read from environment variables and an optional ``.env`` file."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env may also hold variables meant for other tools.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Trace store. The default SQLite file is created on first start, relative to the working directory.
    database_url: str = "sqlite:///data/observability.db"

    # Which model the sandbox calls: "local" runs an open model on this machine
    # (pip install -e ".[local]"), "claude" calls the Anthropic API (pip install -e ".[claude]").
    llm_provider: Literal["local", "claude"] = "local"

    # Local models (Hugging Face Hub ids; downloaded once on first use)
    local_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    local_device: str = "auto"  # auto (CUDA, then Apple MPS, then CPU), cuda, mps or cpu
    local_max_new_tokens: int = 512

    # Claude (LLM_PROVIDER=claude). The judge grades coherence and relevance of traced responses.
    anthropic_api_key: str = ""
    chat_model: str = "claude-opus-5-5"
    judge_model: str = "claude-haiku-5-5"


settings = Settings()
