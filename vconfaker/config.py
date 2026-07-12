"""Typed configuration for the vcon_faker engine.

Plain dataclass (stdlib only) read from the environment, so the engine has no
hard dependency on Streamlit's ``st.secrets`` or on ``pydantic-settings``. The
Streamlit app can still build a ``Settings`` from ``st.secrets`` by passing
overrides to ``from_env``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Engine configuration. Construct via :meth:`from_env`."""

    llm_provider: str = "openai"          # openai | anthropic | scripted
    storage_backend: str = "local"        # local | s3
    output_dir: str = "./out"
    concurrency: int = 1

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    openai_tts_model: str = "tts-1"
    whisper_model: str = "whisper-1"
    whisper_enabled: bool = False

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-20250514"

    aws_access_key: str | None = None
    aws_secret_key: str | None = None
    s3_bucket: str | None = None

    def chat_model(self) -> str:
        """Return the active chat model name for the configured LLM provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        return self.openai_model

    @classmethod
    def from_env(cls, **overrides: object) -> Settings:
        """Build settings from environment variables, then apply non-None overrides.

        Recognized env vars: ``VCONFAKER_LLM_PROVIDER``, ``VCONFAKER_STORAGE``,
        ``VCONFAKER_OUTPUT_DIR``, ``VCONFAKER_CONCURRENCY``, ``VCONFAKER_WHISPER``,
        ``OPENAI_API_KEY``, ``OPENAI_MODEL``, ``OPENAI_TTS_MODEL``,
        ``WHISPER_MODEL``, ``ANTHROPIC_API_KEY``, ``ANTHROPIC_MODEL``,
        ``AWS_ACCESS_KEY``, ``AWS_SECRET_KEY``, ``S3_BUCKET``.
        """
        env = os.environ
        base = cls(
            llm_provider=env.get("VCONFAKER_LLM_PROVIDER", "openai"),
            storage_backend=env.get("VCONFAKER_STORAGE", "local"),
            output_dir=env.get("VCONFAKER_OUTPUT_DIR", "./out"),
            concurrency=int(env.get("VCONFAKER_CONCURRENCY", "1")),
            openai_api_key=env.get("OPENAI_API_KEY"),
            openai_model=env.get("OPENAI_MODEL", "gpt-4o"),
            openai_tts_model=env.get("OPENAI_TTS_MODEL", "tts-1"),
            whisper_model=env.get("WHISPER_MODEL", "whisper-1"),
            whisper_enabled=_env_bool("VCONFAKER_WHISPER"),
            anthropic_api_key=env.get("ANTHROPIC_API_KEY"),
            anthropic_model=env.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            aws_access_key=env.get("AWS_ACCESS_KEY"),
            aws_secret_key=env.get("AWS_SECRET_KEY"),
            s3_bucket=env.get("S3_BUCKET"),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(base, key):
                setattr(base, key, value)
        return base
