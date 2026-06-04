"""Provider-agnostic LLM chat for conversation generation.

A small interface with two implementations:

* :class:`ScriptedProvider` - deterministic, offline, dependency-free; used by
  tests, demos, and ``--provider scripted``.
* :class:`OpenAIProvider` - lazily imports the ``openai`` SDK and requests a JSON
  object response, then validates the turns.

Add Claude / local providers by subclassing :class:`LLMProvider` and extending
:func:`get_provider`.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from vconfaker.config import Settings
from vconfaker.models import ConversationSpec, Turn

logger = logging.getLogger(__name__)

VALID_SPEAKERS = {"Agent", "Customer"}


def validate_turns(raw: object) -> list[Turn]:
    """Coerce arbitrary provider output into clean, validated turns."""
    turns: list[Turn] = []
    if not isinstance(raw, list):
        return turns
    for item in raw:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker")
        message = item.get("message")
        if speaker in VALID_SPEAKERS and isinstance(message, str) and message.strip():
            turns.append({"speaker": speaker, "message": message.strip()})
    return turns


class LLMProvider(ABC):
    """Turns a built prompt into a list of conversation turns."""

    @abstractmethod
    def generate(self, spec: ConversationSpec, prompt: str) -> list[Turn]:
        ...


class ScriptedProvider(LLMProvider):
    """Deterministic, offline provider. Returns canned turns unless given some."""

    def __init__(self, turns: list[Turn] | None = None) -> None:
        self._turns = turns

    def generate(self, spec: ConversationSpec, prompt: str) -> list[Turn]:
        if self._turns is not None:
            return validate_turns(self._turns)
        agent, customer = spec.agent_name, spec.customer_name
        return [
            {"speaker": "Agent", "message": f"Hello, this is {agent} at {spec.business_name}. How can I help you today?"},
            {"speaker": "Customer", "message": f"Hi {agent}, this is {customer}. I am calling about {spec.problem}."},
            {"speaker": "Agent", "message": f"I understand. Let me take a look at {spec.problem} for you."},
            {"speaker": "Customer", "message": "Thank you, that resolves my issue."},
            {"speaker": "Agent", "message": f"Glad I could help. Thank you for contacting {spec.business_name}."},
        ]


class OpenAIProvider(LLMProvider):
    """OpenAI chat-completions provider (JSON response)."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def generate(self, spec: ConversationSpec, prompt: str) -> list[Turn]:
        from openai import OpenAI  # lazy import; not needed for scripted/offline use

        client = OpenAI(api_key=self._api_key)
        completion = client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You output JSON with a 'conversation' array of objects "
                        "{speaker, message}. speaker is exactly 'Agent' or 'Customer'. "
                        "Keep the agent and customer names consistent throughout."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        turns = validate_turns(data.get("conversation", []))
        logger.info("OpenAIProvider generated %d turns with model %s", len(turns), self._model)
        return turns


def get_provider(settings: Settings, name: str | None = None) -> LLMProvider:
    """Resolve an :class:`LLMProvider` from settings (or an explicit name)."""
    name = name or settings.llm_provider
    if name in ("scripted", "fake"):
        return ScriptedProvider()
    if name == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured for the 'openai' provider")
        return OpenAIProvider(settings.openai_api_key, settings.openai_model)
    raise ValueError(f"unknown llm provider: {name!r}")
