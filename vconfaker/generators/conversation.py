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

DEFAULT_EMAIL_PROMPT = """\
Generate a realistic email exchange between a customer and a support agent.
The customer opens with a complete email describing their problem; the agent
replies with a greeting, gathers what they need, and resolves the issue over a
few replies. Each turn is one full email body (no subject lines, no headers).
At least 4 messages.
"""

DEFAULT_TRANSFER_PROMPT = """\
Generate a realistic phone conversation in which a customer calls a support
agent, the agent gathers details but needs to escalate, announces the
transfer, and a supervisor takes over and resolves the issue. Turns before the
transfer use speakers 'Agent' and 'Customer'; after the transfer the second
agent speaks as 'Supervisor'. Spell out digits. At least 10 turns and feel
complete.
"""

_DEFAULT_PROMPTS = {
    "messaging": DEFAULT_MESSAGING_PROMPT,
    "email": DEFAULT_EMAIL_PROMPT,
    "transfer": DEFAULT_TRANSFER_PROMPT,
}


def build_prompt(spec: ConversationSpec) -> str:
    """Compose the full LLM prompt for a scenario."""
    base = spec.prompt or _DEFAULT_PROMPTS.get(spec.conversation_type, DEFAULT_VOICE_PROMPT)
    emotion = f" The customer is feeling {spec.emotion}." if spec.emotion else ""
    supervisor = (
        f" The supervisor's name MUST be {spec.supervisor_name}."
        if spec.conversation_type == "transfer" and spec.supervisor_name
        else ""
    )
    return (
        f"{base}\n\n"
        f"The conversation is about {spec.business_name} (a {spec.business}) and concerns "
        f"{spec.problem}.{emotion}\n\n"
        f"The agent's name MUST be {spec.agent_name} and the customer's name MUST be "
        f"{spec.customer_name}.{supervisor}"
    )


def generate_conversation(spec: ConversationSpec, provider: LLMProvider) -> list[Turn]:
    """Generate validated conversation turns for ``spec`` using ``provider``."""
    turns = provider.generate(spec, build_prompt(spec))
    if not turns:
        raise ValueError("LLM provider returned no usable conversation turns")
    return turns
