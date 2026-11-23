"""Scenario packs and the call plan built from one.

A pack (``packs/<industry>.json``) names invented businesses, their Smart Trunk lines, the
scenarios a caller can bring, and a mix: how many calls of each scenario a run makes. The plan
turns the mix into concrete calls: which business and line, the caller, the person who answers,
the voices, and when the call happened. The plan is deterministic for a given seed, so a run
can be repeated or extended.

People's names come from the repo's ``fake_names`` pools. Every phone number is in the
fictional 555-0100 to 555-0199 range.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import fake_names

PACKS = Path(__file__).parent / "packs"

# Kokoro voices, by the gender of the name they speak for. The first letter of a voice id is
# its Kokoro language code ("a" American English, "b" British English).
FEMALE_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_kore", "af_aoede", "af_river", "bf_emma", "bf_isabella"]
MALE_VOICES = ["am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric", "am_liam", "bm_george", "bm_fable"]
SYSTEM_VOICE = "af_sky"  # the receiver's own prompts: the notice and the screening announcement
ROBOCALL_VOICE = "am_onyx"

FIRST_DAY = "2026-11-16"  # a Monday; business-hours calls fall Monday to Saturday after it
FICTIONAL_LOW, FICTIONAL_HIGH = 100, 199  # 555-0100 to 555-0199


def load_pack(industry: str) -> dict[str, Any]:
    path = PACKS / f"{industry}.json"
    if not path.is_file():
        raise ValueError(f"no scenario pack for {industry!r} (looked for {path})")
    pack = json.loads(path.read_text())
    _check_pack(pack)
    return pack


def _check_pack(pack: dict[str, Any]) -> None:
    ids = {s["id"] for s in pack["scenarios"]}
    unknown = set(pack["pilot_mix"]) - ids
    if unknown:
        raise ValueError(f"pilot_mix names unknown scenarios: {sorted(unknown)}")
    for business in pack["businesses"]:
        for line in business["lines"].values():
            fictional_number(business["area_code"], line["number"])


def fictional_number(area_code: str, local: str) -> str:
    """E.164 for ``<area code> 555-01NN``; anything outside 555-0100..0199 is refused."""
    exchange, _, line = local.partition("-")
    if exchange != "555" or not line.isdigit() or not FICTIONAL_LOW <= int(line) <= FICTIONAL_HIGH:
        raise ValueError(f"{local!r} is outside the fictional 555-0100 to 555-0199 range")
    if not (len(area_code) == 3 and area_code.isdigit()):
        raise ValueError(f"bad area code {area_code!r}")
    return f"+1{area_code}555{int(line):04d}"


@dataclass
class Speaker:
    name: str | None  # None for a robocall
    gender: str
    voice: str
    speed: float

    @property
    def lang(self) -> str:
        return self.voice[0]


@dataclass
class CallPlan:
    index: int
    call_id: str
    industry: str
    scenario: dict[str, Any]
    business: dict[str, Any]
    line: str
    business_tel: str
    screened: bool
    caller_tel: str
    caller: Speaker
    agent: Speaker
    owned_vehicle: str
    wanted_model: str
    start: datetime
    after_hours: bool
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.scenario["kind"]

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["scenario"] = self.scenario["id"]
        d["business"] = self.business["name"]
        return d


def build_plan(pack: dict[str, Any], *, count: int | None = None, seed: int = 20261123) -> list[CallPlan]:
    """The calls a run makes: the pack's mix in a shuffled order, cut to ``count``."""
    rng = random.Random(seed)
    scenarios = {s["id"]: s for s in pack["scenarios"]}
    order = [sid for sid, n in pack["pilot_mix"].items() for _ in range(n)]
    rng.shuffle(order)
    if count is not None:
        if count > len(order):
            order = (order * (count // len(order) + 1))[:count]
        order = order[:count]

    tz = ZoneInfo(pack["timezone"])
    businesses = pack["businesses"]
    used_first: set[str] = set()
    used_last: set[str] = set()
    plan: list[CallPlan] = []
    voice_turn = {"f": 0, "m": 0}

    def next_voice(gender: str, avoid: str | None = None) -> str:
        pool = FEMALE_VOICES if gender == "f" else MALE_VOICES
        for _ in range(len(pool)):
            voice = pool[voice_turn[gender] % len(pool)]
            voice_turn[gender] += 1
            if voice != avoid:
                return voice
        return pool[0]

    def person(gender: str) -> str:
        pool = fake_names.female_names if gender == "f" else fake_names.male_names
        for _ in range(50):
            first = rng.choice(pool)
            if first not in used_first:
                used_first.add(first)
                return first
        return rng.choice(pool)

    for i, sid in enumerate(order, start=1):
        scenario = scenarios[sid]
        line = scenario["line"]
        business = businesses[(i - 1) % len(businesses)]
        if scenario["kind"] == "robocall":
            # A robocall is the case screening exists for: put it on a number that screens.
            screened = [b for b in businesses if b["lines"][line].get("screen")] or businesses
            business = screened[(i - 1) % len(screened)]
        line_cfg = business["lines"][line]
        caller_number = f"555-{FICTIONAL_LOW + 50 + i:04d}"
        caller_area = pack["caller_area_codes"][(i - 1) % len(pack["caller_area_codes"])]

        agent_gender = rng.choice("fm")
        agent = Speaker(person(agent_gender), agent_gender, next_voice(agent_gender), round(rng.uniform(0.95, 1.08), 2))
        if scenario["kind"] == "robocall":
            caller = Speaker(None, "m", ROBOCALL_VOICE, 1.05)
        else:
            caller_gender = rng.choice("fm")
            first = person(caller_gender)
            last = rng.choice([n for n in fake_names.last_names if n not in used_last] or fake_names.last_names)
            used_last.add(last)
            caller = Speaker(
                f"{first} {last}",
                caller_gender,
                next_voice(caller_gender, avoid=agent.voice),
                round(rng.uniform(0.92, 1.1), 2),
            )

        after_hours = sid == "after_hours"
        start = _call_time(rng, tz, i, after_hours)
        # Service, parts and recall callers own the make the dealer services; others may not.
        owned_pool = business["models"] if line == "service" else pack["owned_vehicles"]
        plan.append(
            CallPlan(
                index=i,
                call_id=f"{pack['industry']}-{i:02d}-{sid.replace('_', '-')}",
                industry=pack["industry"],
                scenario=scenario,
                business=business,
                line=line,
                business_tel=fictional_number(business["area_code"], line_cfg["number"]),
                screened=bool(line_cfg.get("screen")) and not after_hours,
                caller_tel=fictional_number(caller_area, caller_number),
                caller=caller,
                agent=agent,
                owned_vehicle=f"{rng.randint(2015, 2023)} {rng.choice(owned_pool)}",
                wanted_model=rng.choice(business["models"]),
                start=start,
                after_hours=after_hours,
            )
        )
    return plan


def _call_time(rng: random.Random, tz: ZoneInfo, i: int, after_hours: bool) -> datetime:
    first = datetime.fromisoformat(FIRST_DAY).replace(tzinfo=tz)
    if after_hours:
        # a weekday evening, or the Sunday the dealerships are closed
        day = first + timedelta(days=rng.choice([1, 2, 3, 6]))
        hour = 10 if day.weekday() == 6 else rng.choice([20, 21])
    else:
        day = first + timedelta(days=(i - 1) % 6)  # Monday to Saturday
        hour = rng.randint(9, 17)
    return day.replace(hour=hour, minute=rng.randint(0, 59), second=rng.randint(0, 59))
