"""Lightweight data types shared across the engine (stdlib dataclasses)."""
from __future__ import annotations

from dataclasses import dataclass

# A single conversation turn: {"speaker": "Agent"|"Customer"|"Supervisor", "message": "..."}.
Turn = dict[str, str]

# Supported conversation modes. Only "voice" can carry synthesized audio; the
# rest are pure text and never require TTS/wav synthesis.
CONVERSATION_TYPES = ("voice", "messaging", "email", "incomplete", "transfer")


@dataclass
class ConversationSpec:
    """Everything a generator needs to produce one conversation.

    Contact fields default to empty strings so a spec can be built for
    prompt/turn generation without party identities; :func:`vconfaker.generators
    .scenario.random_spec` fills them for full vCon assembly.
    """

    agent_name: str
    customer_name: str
    business: str
    business_name: str
    problem: str
    agent_phone: str = ""
    customer_phone: str = ""
    agent_email: str = ""
    customer_email: str = ""
    emotion: str | None = None
    conversation_type: str = "voice"   # one of CONVERSATION_TYPES
    supervisor_name: str = ""          # third party for "transfer" mode
    disposition: str | None = None     # failure reason for "incomplete" mode
    prompt: str | None = None          # override the default template if set
