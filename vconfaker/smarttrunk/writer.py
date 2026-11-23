"""Call scripts written by Claude, checked by ``hygiene`` before any audio is made.

One Messages API request per script, with structured output so the reply is always the
script JSON. The model is ``claude-sonnet-5-5`` with adaptive thinking (its default) at
``medium`` effort, and server-side fallbacks on, so a classifier decline is retried on the
fallback model instead of failing the call. A rejected script is written again from scratch
with the reasons, up to ``attempts`` times.

The API key is read from ``ANTHROPIC_API_KEY`` by the SDK. Nothing here prints or logs it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .hygiene import check_script
from .pack import CallPlan

MODEL = "claude-sonnet-5-5"
EFFORT = "medium"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Claude Sonnet 5.5 list prices, USD per million tokens (claude-api skill, cached 2026-11-03).
PRICE = {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.10}

SCRIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "outcome": {"type": "string"},
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["caller", "business"]},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
                "additionalProperties": False,
            },
        },
        "names_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "outcome", "turns", "names_used"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You write short phone call scripts for a synthetic test corpus. Each script is one call to a \
business, turned into audio with text-to-speech and played to speech recognition, so it must \
read aloud naturally.

How people talk:
- Spoken English as it sounds on a real business line: short turns, contractions, the odd \
"um" or "let me check", plain words. Most turns are one or two sentences; none is over 45 words.
- The business side is polite and practical, never a sales brochure. The caller has a reason \
to call and a little personality (hurried, chatty, unsure, friendly), shown in how they talk.
- The business's automated recording notice has already played before the first turn. Do not \
write it. The caller may mention it.
- Text only: no stage directions, sound effects, speaker labels, brackets or emoji inside a turn.

Names and facts (these are hard rules; a script that breaks one is thrown away):
- Use only the names listed under "Allowed names". The person answering gives only their \
first name. Never invent another person, business, town, street, school or website.
- Never name a real company: no other dealers, car retailers, parts chains, banks, lenders, \
insurers, valuation or history-report sites, rideshare or rental companies. Say "our finance \
team", "an online estimate", "your insurance" instead.
- Vehicle makes and models are fine, but only the ones in the brief.
- Phone numbers: only the caller's number given in the brief, written with digits like \
555-0152. No other numbers that look like phone numbers. No email addresses or web addresses.
- Do not state facts about real vehicles (defects, ratings, exact prices). Prices are rough \
("in the low thirties"). A VIN is only ever its last six characters, letters and digits.

Output: the script as JSON. "turns" alternate naturally between "caller" and "business". \
"names_used" lists every person or business name that appears in the turns. "summary" is one \
sentence on what happens; "outcome" is a few words (for example "test drive booked")."""

KIND_RULES = {
    "conversation": "The business speaks first, greeting with the business name, the department, and their first name.",
    "hangup": (
        "The business speaks first, greeting with the business name, the department, and their first name. "
        "The last turn is the caller's, said to themselves after being left on hold; then they hang up."
    ),
    "voicemail": (
        "Exactly two turns. Turn 1 (business) is the advisor's recorded voicemail greeting, ending by asking "
        "the caller to leave a message after the tone. Turn 2 (caller) is the whole message."
    ),
    "robocall": (
        "Exactly two turns. Turn 1 (business) is the greeting with the business name, the department, and their "
        "first name. Turn 2 (caller) is a prerecorded automated pitch, about 35 words, that names no company and "
        "breaks off mid-sentence, ending with an ellipsis."
    ),
}


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    refusals: int = 0
    fallbacks: int = 0
    seconds: float = 0.0

    def add(self, response: Any, seconds: float) -> None:
        u = response.usage
        self.requests += 1
        self.input_tokens += u.input_tokens or 0
        self.output_tokens += u.output_tokens or 0
        self.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        if any(getattr(e, "type", "") == "fallback_message" for e in getattr(u, "iterations", None) or []):
            self.fallbacks += 1
        self.seconds += seconds

    @property
    def cost_usd(self) -> float:
        return round(
            (
                self.input_tokens * PRICE["input"]
                + self.output_tokens * PRICE["output"]
                + self.cache_write_tokens * PRICE["cache_write"]
                + self.cache_read_tokens * PRICE["cache_read"]
            )
            / 1_000_000,
            4,
        )

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["estimated_cost_usd"] = self.cost_usd
        d["model"] = MODEL
        d["effort"] = EFFORT
        d["price_per_mtok_usd"] = PRICE
        return d


@dataclass
class Script:
    summary: str
    outcome: str
    turns: list[dict[str, str]]
    names_used: list[str]
    attempts: int
    rejected: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def allowed_names(plan: CallPlan) -> list[str]:
    names = [plan.business["name"], plan.agent.name or ""]
    if plan.caller.name:
        names.append(plan.caller.name)
    return [n for n in names if n]


def products(plan: CallPlan, pack: dict[str, Any]) -> list[str]:
    return [*plan.business["sells"], *plan.business["models"], plan.owned_vehicle, plan.wanted_model]


def brief(plan: CallPlan) -> str:
    s, b = plan.scenario, plan.business
    when = plan.start.strftime("%A at %-I:%M %p")
    lines = [
        f"Business: {b['name']}, a {' and '.join(b['sells'])} dealer. Hours: {b['hours']}.",
        f"This call reaches the {plan.line} line. The person answering: {plan.agent.name}, {s['agent_role']}.",
    ]
    if plan.caller.name:
        lines.append(
            f"Caller: {plan.caller.name}, calling from 555-{plan.caller_tel[-4:]}. "
            f"They own a {plan.owned_vehicle}."
        )
    else:
        lines.append("Caller: an automated dialer playing a prerecorded message.")
    lines += [
        f"Vehicles the dealer sells: {', '.join(b['models'])}. The model of interest, if any: {plan.wanted_model}.",
        f"The call is placed on {when}{' (after the dealership has closed)' if plan.after_hours else ''}. "
        "Anything it books is for a later day, not this moment.",
        f"Scenario: {s['label']}. {s['brief']}",
        f"Length: {s['turns'][0]} to {s['turns'][1]} turns.",
        KIND_RULES[plan.kind],
        f"Allowed names: {'; '.join(allowed_names(plan))}.",
    ]
    return "\n".join(lines)


class ScriptWriter:
    def __init__(self, pack: dict[str, Any], *, attempts: int = 3, client: Any = None) -> None:
        self.pack = pack
        self.attempts = attempts
        self.usage = Usage()
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=4)
        self.client = client

    def write(self, plan: CallPlan) -> Script:
        rejected: list[list[str]] = []
        base = brief(plan)
        for attempt in range(1, self.attempts + 1):
            prompt = base
            if rejected:
                prompt += "\n\nAn earlier draft of this script was thrown away because: " + "; ".join(rejected[-1]) + ". Avoid that."
            data, problem = self._request(prompt)
            if data is None:
                rejected.append([problem])
                continue
            verdict = check_script(
                data,
                allowed_names=allowed_names(plan),
                products=products(plan, self.pack),
                turn_range=tuple(plan.scenario["turns"]),
            )
            if verdict.ok:
                return Script(
                    summary=data["summary"],
                    outcome=data["outcome"],
                    turns=[{"speaker": t["speaker"], "text": t["text"].strip()} for t in data["turns"]],
                    names_used=data["names_used"],
                    attempts=attempt,
                    rejected=rejected,
                    notes=verdict.notes,
                )
            rejected.append(verdict.problems)
        raise ScriptRejected(plan.call_id, rejected)

    def _request(self, prompt: str) -> tuple[dict[str, Any] | None, str]:
        t0 = time.monotonic()
        response = self.client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCRIPT_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
        self.usage.add(response, time.monotonic() - t0)
        if response.stop_reason == "refusal":
            self.usage.refusals += 1
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            return None, f"the model declined ({category})"
        if response.stop_reason == "max_tokens":
            return None, "the reply was cut off at max_tokens"
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text), ""
        except json.JSONDecodeError:
            return None, "the reply was not valid JSON"


class ScriptRejected(Exception):
    def __init__(self, call_id: str, reasons: list[list[str]]) -> None:
        super().__init__(f"{call_id}: every draft was rejected: {reasons}")
        self.reasons = reasons
