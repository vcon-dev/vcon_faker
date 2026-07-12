"""Turn a :class:`ConversationSpec` into conversation turns via an LLM provider.

The prompt templates live here (moved out of the Streamlit app) so the CLI, the
API, and the Streamlit UI all share one source of truth.
"""
from __future__ import annotations

from vconfaker.models import ConversationSpec, Turn
from vconfaker.providers.llm import LLMProvider

DEFAULT_VOICE_PROMPT = """\
Generate a realistic phone conversation between a customer and a support agent.
The agent introduces themselves and their company and asks for the customer's
name. Over the course of the call the agent gathers two pieces of information,
helps with the issue, and politely closes the call. Spell out digits (for
example say "one zero zero zero", not "one thousand"). The conversation should
be at least 10 turns and feel complete.
"""

DEFAULT_MESSAGING_PROMPT = """\
Generate a realistic text-messaging conversation between a customer and a
support agent. The customer opens with a problem; the agent introduces
themselves and their company, gathers what they need, resolves the issue, and
offers further help. Use short, natural messaging-style turns. At least 8 turns.
"""


def build_prompt(spec: ConversationSpec) -> str:
    """Compose the full LLM prompt for a scenario."""
    base = spec.prompt or (
        DEFAULT_MESSAGING_PROMPT if spec.conversation_type == "messaging" else DEFAULT_VOICE_PROMPT
    )
    emotion = f" The customer is feeling {spec.emotion}." if spec.emotion else ""
    return (
        f"{base}\n\n"
        f"The conversation is about {spec.business_name} (a {spec.business}) and concerns "
        f"{spec.problem}.{emotion}\n\n"
        f"The agent's name MUST be {spec.agent_name} and the customer's name MUST be "
        f"{spec.customer_name}."
    )


def generate_conversation(spec: ConversationSpec, provider: LLMProvider) -> list[Turn]:
    """Generate validated conversation turns for ``spec`` using ``provider``."""
    turns = provider.generate(spec, build_prompt(spec))
    if not turns:
        raise ValueError("LLM provider returned no usable conversation turns")
    return turns
