"""Provider-agnostic LLM chat for conversation generation.

Implementations:

* :class:`ScriptedProvider` - deterministic, offline; used by tests and
  ``--provider scripted``.
* :class:`OpenAIProvider` - structured JSON schema output via OpenAI chat.
* :class:`AnthropicProvider` - structured tool output via Claude.

Add local providers by subclassing :class:`LLMProvider` and extending
:func:`get_provider`.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from vconfaker.config import Settings
from vconfaker.models import ConversationSpec, Turn
from vconfaker.schemas.conversation import CONVERSATION_JSON_SCHEMA, ConversationResponse

logger = logging.getLogger(__name__)

VALID_SPEAKERS = {"Agent", "Customer", "Supervisor"}

SYSTEM_PROMPT = (
    "You generate realistic customer support conversations. "
    "Each turn has speaker exactly 'Agent', 'Customer', or 'Supervisor' "
    "(use 'Supervisor' only when the prompt describes a transfer) and a "
    "non-empty message. "
    "Use the agent and customer names given in the user prompt consistently."
)


def validate_turns(raw: object) -> list[Turn]:
    """Coerce arbitrary provider output into clean, validated turns."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    items = raw.get("conversation", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    turns: list[Turn] = []
    for item in items:
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
        if spec.conversation_type == "transfer":
            supervisor = spec.supervisor_name or "the supervisor"
            return [
                {"speaker": "Agent", "message": f"Hello, this is {agent} at {spec.business_name}. How can I help you today?"},
                {"speaker": "Customer", "message": f"Hi {agent}, this is {customer}. I am calling about {spec.problem}."},
                {"speaker": "Agent", "message": f"I need to escalate this. Let me transfer you to {supervisor}."},
                {"speaker": "Supervisor", "message": f"Hello {customer}, this is {supervisor}. I can resolve {spec.problem} for you right now."},
                {"speaker": "Customer", "message": "Thank you, that resolves my issue."},
                {"speaker": "Supervisor", "message": f"Glad I could help. Thank you for contacting {spec.business_name}."},
            ]
        return [
            {"speaker": "Agent", "message": f"Hello, this is {agent} at {spec.business_name}. How can I help you today?"},
            {"speaker": "Customer", "message": f"Hi {agent}, this is {customer}. I am calling about {spec.problem}."},
            {"speaker": "Agent", "message": f"I understand. Let me take a look at {spec.problem} for you."},
            {"speaker": "Customer", "message": "Thank you, that resolves my issue."},
            {"speaker": "Agent", "message": f"Glad I could help. Thank you for contacting {spec.business_name}."},
        ]


class OpenAIProvider(LLMProvider):
    """OpenAI chat-completions with strict JSON schema structured output."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def generate(self, spec: ConversationSpec, prompt: str) -> list[Turn]:
        from openai import OpenAI  # lazy import; not needed for scripted/offline use

        client = OpenAI(api_key=self._api_key)
        completion = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "conversation_response",
                    "strict": True,
                    "schema": CONVERSATION_JSON_SCHEMA,
                },
            },
        )
        content = completion.choices[0].message.content or "{}"
        response = ConversationResponse.model_validate(json.loads(content))
        turns = [{"speaker": t.speaker, "message": t.message} for t in response.conversation]
        logger.info("OpenAIProvider generated %d turns with model %s", len(turns), self._model)
        return turns


class AnthropicProvider(LLMProvider):
    """Anthropic Claude with tool-use structured output."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def generate(self, spec: ConversationSpec, prompt: str) -> list[Turn]:
        import anthropic  # lazy import; optional dependency

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "name": "emit_conversation",
                    "description": "Return the generated conversation turns.",
                    "input_schema": CONVERSATION_JSON_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "emit_conversation"},
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "emit_conversation":
                parsed = ConversationResponse.model_validate(block.input)
                turns = [{"speaker": t.speaker, "message": t.message} for t in parsed.conversation]
                logger.info(
                    "AnthropicProvider generated %d turns with model %s",
                    len(turns),
                    self._model,
                )
                return turns
        raise ValueError("Anthropic provider returned no conversation tool output")


def get_provider(settings: Settings, name: str | None = None) -> LLMProvider:
    """Resolve an :class:`LLMProvider` from settings (or an explicit name)."""
    name = name or settings.llm_provider
    if name in ("scripted", "fake"):
        return ScriptedProvider()
    if name == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured for the 'openai' provider")
        return OpenAIProvider(settings.openai_api_key, settings.openai_model)
    if name == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured for the 'anthropic' provider")
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    raise ValueError(f"unknown llm provider: {name!r}")
