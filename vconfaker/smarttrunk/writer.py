"""Call scripts written by Claude, checked by ``hygiene`` before any audio is made.

One Messages API request per script, with structured output so the reply is always the
script JSON. The model is ``claude-sonnet-5-5`` with adaptive thinking (its default) at
``medium`` effort, and server-side fallbacks on, so a classifier decline is retried on the
fallback model instead of failing the call. A rejected script is written again from scratch
with the reasons, up to ``attempts`` times.

Spending is capped: with ``max_cost_usd`` the writer stops before a request that would start
past the cap, counting what earlier runs recorded in ``usage_file``.

The API key is read from ``ANTHROPIC_API_KEY`` by the SDK. Nothing here prints or logs it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hygiene import check_script
from .pack import LENGTHS, OPENINGS, PEOPLE_FORMS, CallPlan

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
                    "speaker": {"type": "string", "enum": ["caller", "business", "colleague"]},
                    "text": {"type": "string"},
                    "interrupts": {"type": "boolean"},
                },
                "required": ["speaker", "text", "interrupts"],
                "additionalProperties": False,
            },
        },
        "names_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "outcome", "turns", "names_used"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You write phone call scripts for a synthetic test corpus. Each script is one call to a small \
business or office, turned into audio with text-to-speech and played to speech recognition, so \
it must read aloud naturally.

How people talk:
- Spoken English as it sounds on a real business line: contractions, the odd "um" or "let me \
check", plain words, people who do not always speak in full sentences. Turns vary in length; \
none is over 45 words.
- The business side is polite and practical. The caller has a reason to call and a little \
personality (hurried, chatty, unsure, friendly, annoyed), shown in how they talk.
- The automated recording notice has already played, and any phone menu has been dealt with, \
before the first turn. Do not write either. The caller may mention the notice.
- Text only: no stage directions, sound effects, speaker labels, brackets or emoji inside a turn.
- "interrupts" is true only on a turn that cuts in before the other person has finished; the \
turn it cuts into then ends mid-sentence with a dash. Otherwise it is false.
- "colleague" is the speaker label for the person a call is transferred to, and is used only \
when the brief says there is a transfer.

Names and facts (hard rules; a script that breaks one is thrown away):
- Use only the names listed under "Allowed names" and "Places". The person answering gives only \
their first name. Never invent another person, business, town, street, school or website.
- Never name a real company, hospital, pharmacy, bank, insurer, utility, agency, chain, \
marketplace or place. Say "your insurance", "your usual pharmacy", "an online estimate", \
"the gas company" instead.
- Phone numbers: only the caller's number given in the brief, written with digits like \
555-0152. No other numbers that look like phone numbers (911 is fine). No email or web addresses.
- Nothing is booked for the moment of the call. Anything scheduled is for a later day, or at \
least two hours after the call.

Output: the script as JSON. "names_used" lists every person, business and place name that \
appears in the turns. "summary" is one sentence on what happens; "outcome" is a few words."""

KIND_RULES = {
    "conversation": "",
    "hangup": "The last turn is the caller's, said to themselves after being left on hold; then they hang up.",
    "voicemail": (
        "Exactly two turns. Turn 1 (business) is the recorded voicemail greeting, ending by asking the caller to "
        "leave a message after the tone. Turn 2 (caller) is the whole message."
    ),
    "robocall": (
        "Exactly two turns. Turn 1 (business) answers the phone. Turn 2 (caller) is a prerecorded automated pitch, "
        "about 35 words, that names no company and breaks off mid-sentence, ending with an ellipsis."
    ),
    "wrong_number": "Keep it brief and natural; people apologise and hang up.",
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
    turns: list[dict[str, Any]]
    names_used: list[str]
    attempts: int
    rejected: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def allowed_names(plan: CallPlan) -> list[str]:
    names = [plan.business["name"], plan.agent.name or "", *plan.business.get("places", []), *plan.people.values()]
    if plan.caller.name:
        names.append(plan.caller.name)
    if plan.colleague:
        names.append(plan.colleague.name or "")
    return [n for n in names if n]


def products(plan: CallPlan, pack: dict[str, Any] | None = None) -> list[str]:
    return [*plan.business["products"], *plan.business.get("sells", []), plan.wanted, plan.caller_context]


def brief(plan: CallPlan, pack: dict[str, Any]) -> str:
    s, b = plan.scenario, plan.business
    when = plan.start.strftime("%A at %-I:%M %p")
    length = LENGTHS[plan.length]
    route = f" after the phone menu (the caller pressed {plan.ivr_key})" if plan.ivr_key else ""
    lines = [
        f"Business: {b['name']}, {b['description']}. Hours: {b['hours']}.",
        f"This call reaches the {plan.line.replace('_', ' ')} line{route}. The person answering: {plan.agent.name}, {s['agent_role']}.",
    ]
    if plan.caller.name:
        lines.append(f"Caller: {plan.caller.name}, calling from 555-{plan.caller_tel[-4:]}. {plan.caller_context}")
    else:
        lines.append("Caller: an automated dialer playing a prerecorded message.")
    lines += [
        f"{pack.get('products_label', 'What the business offers')}: {', '.join(b['products'])}. Of interest, if it fits: {plan.wanted}.",
        f"The call is placed on {when}{' (after the business has closed)' if plan.after_hours else ''}. "
        "Anything booked is for a later day or at least two hours later, never for this moment.",
        f"Scenario: {s['label']}. {s['brief']}",
    ]
    if plan.kind not in ("voicemail", "robocall"):
        lines.append(f"Opening: {OPENINGS[plan.opening]}")
    lines.append(
        f"Length: {length['words'][0]} to {length['words'][1]} words in all, {length['turns'][0]} to {length['turns'][1]} turns "
        f"({length['says']})."
    )
    if plan.transfer:
        lines.append(
            f"Transfer: partway through, {plan.agent.name} transfers the caller to {plan.colleague.name}, the "
            f"{plan.colleague_role}, after a short hold; label {plan.colleague.name}'s turns \"colleague\". "
            f"{plan.colleague.name} greets the caller by name and picks up where {plan.agent.name} left off."
        )
    if plan.interruptions:
        lines.append("Interruptions: once or twice, one person cuts in before the other finishes (mark it as the rules say).")
    if KIND_RULES[plan.kind]:
        lines.append(KIND_RULES[plan.kind])
    if pack.get("writer_rules"):
        lines.append(f"Industry rules: {pack['writer_rules']}")
    if plan.people:
        lines.append("People: " + "; ".join(f"{name}, {PEOPLE_FORMS[role][1]}" for role, name in plan.people.items()) + ".")
    if b.get("places"):
        lines.append(f"Places: {'; '.join(b['places'])}.")
    lines.append(f"Allowed names: {'; '.join(allowed_names(plan))}.")
    return "\n".join(lines)


class BudgetExceeded(Exception):
    pass


class ScriptWriter:
    def __init__(
        self,
        pack: dict[str, Any],
        *,
        attempts: int = 3,
        client: Any = None,
        max_cost_usd: float | None = None,
        usage_file: Path | None = None,
    ) -> None:
        self.pack = pack
        self.attempts = attempts
        self.usage = Usage()
        self.max_cost_usd = max_cost_usd
        self.prior_cost_usd = _recorded_cost(usage_file)
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=4)
        self.client = client

    def check(self, plan: CallPlan, data: dict[str, Any]):
        return check_script(
            data,
            allowed_names=allowed_names(plan),
            products=products(plan),
            turn_range=tuple(LENGTHS[plan.length]["turns"]),
            industry=plan.industry,
            extra_real=self.pack.get("real_businesses", ()),
            reject_places=bool(self.pack.get("reject_places")),
            call_start=plan.start,
            colleague=plan.transfer,
        )

    def write(self, plan: CallPlan) -> Script:
        rejected: list[list[str]] = []
        base = brief(plan, self.pack)
        for attempt in range(1, self.attempts + 1):
            prompt = base
            if rejected:
                prompt += "\n\nAn earlier draft of this script was thrown away because: " + "; ".join(rejected[-1]) + ". Avoid that."
            data, problem = self._request(prompt)
            if data is None:
                rejected.append([problem])
                continue
            verdict = self.check(plan, data)
            if verdict.ok:
                return Script(
                    summary=data["summary"],
                    outcome=data["outcome"],
                    turns=[
                        {"speaker": t["speaker"], "text": t["text"].strip(), "interrupts": bool(t.get("interrupts"))}
                        for t in data["turns"]
                    ],
                    names_used=data["names_used"],
                    attempts=attempt,
                    rejected=rejected,
                    notes=verdict.notes,
                )
            rejected.append(verdict.problems)
        raise ScriptRejected(plan.call_id, rejected)

    def _request(self, prompt: str) -> tuple[dict[str, Any] | None, str]:
        spent = self.prior_cost_usd + self.usage.cost_usd
        if self.max_cost_usd is not None and spent >= self.max_cost_usd:
            raise BudgetExceeded(f"script spending reached ${spent:.2f}, the cap is ${self.max_cost_usd:.2f}")
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


def _recorded_cost(usage_file: Path | None) -> float:
    if usage_file is None or not Path(usage_file).is_file():
        return 0.0
    record = json.loads(Path(usage_file).read_text())
    return float(sum(r.get("estimated_cost_usd", 0) for r in record.get("runs", [])))


class ScriptRejected(Exception):
    def __init__(self, call_id: str, reasons: list[list[str]]) -> None:
        super().__init__(f"{call_id}: every draft was rejected: {reasons}")
        self.reasons = reasons
