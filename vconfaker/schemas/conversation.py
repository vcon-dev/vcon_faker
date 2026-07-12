"""Structured conversation output validated by Pydantic.

All LLM providers emit data matching :class:`ConversationResponse` so agent and
customer names stay consistent without post-hoc patching.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from vconfaker.models import Turn

Speaker = Literal["Agent", "Customer", "Supervisor"]


class ConversationTurn(BaseModel):
    speaker: Speaker
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be empty")
        return stripped


class ConversationResponse(BaseModel):
    conversation: list[ConversationTurn] = Field(min_length=1)


# JSON schema for provider APIs (OpenAI strict mode, Anthropic tools).
CONVERSATION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "conversation": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["Agent", "Customer", "Supervisor"]},
                    "message": {"type": "string", "minLength": 1},
                },
                "required": ["speaker", "message"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["conversation"],
    "additionalProperties": False,
}


def parse_conversation(raw: object) -> list[Turn]:
    """Validate provider output into clean turns (lenient item filtering)."""
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
        if speaker in ("Agent", "Customer", "Supervisor") and isinstance(message, str) and message.strip():
            turns.append({"speaker": speaker, "message": message.strip()})
    return turns
