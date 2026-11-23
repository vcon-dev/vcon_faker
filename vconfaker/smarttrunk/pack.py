"""Scenario packs and the call plan built from one.

A pack (``packs/<industry>.json``) names invented businesses, their Smart Trunk lines, the
scenarios a caller can bring and the weight of each in a mix. The plan turns a count into
concrete calls: which business and line, the caller, the person who answers, the voices, when
the call happened, and how it goes. The plan is deterministic for a given seed, so a run can be
repeated or extended.

How a call goes varies so a corpus does not sound like one call twenty times over: the opening,
the length (many calls are short), a phone menu on some lines, a transfer to a colleague, and
interruptions. Wrong numbers, hang-ups, voicemails and robocalls are scenarios of their own.

People's names come from the repo's ``fake_names`` pools. Business lines use 555-0100 to
555-0149 and callers 555-0150 to 555-0199, the fictional range, with the area code varying.
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
INDUSTRIES = ("dealership", "clinic", "municipal", "home_services")

# Kokoro voices, by the gender of the name they speak for. The first letter of a voice id is
# its Kokoro language code ("a" American English, "b" British English).
FEMALE_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_kore", "af_aoede", "af_river", "bf_emma", "bf_isabella"]
MALE_VOICES = ["am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric", "am_liam", "bm_george", "bm_fable"]
SYSTEM_VOICE = "af_sky"  # the receiver's and the business's recorded prompts
ROBOCALL_VOICE = "am_onyx"

FIRST_DAY = "2026-11-16"  # a Monday
FICTIONAL_LOW, FICTIONAL_HIGH = 100, 199  # 555-0100 to 555-0199
BUSINESS_LINES = (100, 149)
CALLER_LINES = (150, 199)

# How the person answering opens the call. The writer gets the sentence.
OPENINGS = {
    "full": "Answer with the business name, the department and your first name.",
    "name_only": "Answer with just your first name and an offer to help; no business name.",
    "department": "Answer with only the department and your first name, for example 'Front desk, this is Dana.'",
    "time_of_day": "Open with good morning or good afternoon, whichever fits the time, then the business name.",
    "busy": "Pick up busy: the business name, ask the caller to hold one second, then come straight back to them.",
    "plain": "Pick up with only the business name and 'how can I help', said quickly.",
}
# Length buckets: total words, turns, and how long that is on a phone.
LENGTHS = {
    "short": {"words": (25, 70), "turns": (2, 5), "says": "10 to 30 seconds of talk"},
    "medium": {"words": (110, 210), "turns": (6, 12), "says": "about a minute of talk"},
    "long": {"words": (230, 350), "turns": (12, 20), "says": "two minutes or so of talk"},
}
CONVERSATION_LENGTHS = {"short": 0.40, "medium": 0.38, "long": 0.22}
FIXED_LENGTH = {"voicemail": "short", "robocall": "short", "wrong_number": "short", "hangup": "short"}
TRANSFER_SHARE = 0.18  # of medium and long conversations
INTERRUPTION_SHARE = 0.35  # of medium and long conversations and transfers

# People a scenario can ask for, besides the caller and the person answering.
PEOPLE_FORMS = {
    "provider": ("Dr. {last}", "the doctor or dentist at the practice the call is about"),
    "asked_for": ("{first}", "the person the caller meant to reach, who does not work here"),
    "patient": ("{first} {last}", "the patient the call is about, in the caller's family"),
}


# Pairs of first and last names from the ``fake_names`` pools that belong to well-known people.
# A caller or patient never gets one.
FAMOUS_NAMES = {
    "Martha Stewart", "Patrick Stewart", "James Taylor", "Elizabeth Taylor", "John Williams", "Larry King",
    "Jennifer Lopez", "George Lopez", "George Washington", "John Adams", "Andrew Johnson", "Michael Johnson",
    "Brian Wilson", "Russell Wilson", "Paul Allen", "Timothy Allen", "Steven Martin", "Kevin James", "Bruce Lee",
    "Brenda Lee", "Robert Lee", "Diana Ross", "Betty White", "Jack White", "Emma Watson", "Emma Thompson",
    "Samuel Jackson", "Janet Jackson", "Michael Jackson", "Andrew Jackson", "Jesse Jackson", "Peter Jackson",
    "Joe Jackson", "Paul Walker", "Grace Kelly", "Gary Cooper", "Anthony Davis", "Willie Nelson", "Matthew Perry",
    "Tyler Perry", "Joseph Smith", "John Smith", "Roy Rogers", "Carl Lewis", "Jerry Lewis", "John Lewis",
    "Kevin Hart", "Mark Taylor", "Elizabeth Warren", "Kenneth Rogers", "Henry Ford", "Thomas Edison",
    "Jimmy Carter", "James Carter", "Nancy Reagan", "Ronald Reagan", "Mary Phillips", "Billy Ray", "Johnny Cash",
}


def load_pack(industry: str) -> dict[str, Any]:
    path = PACKS / f"{industry}.json"
    if not path.is_file():
        raise ValueError(f"no scenario pack for {industry!r} (looked for {path})")
    pack = json.loads(path.read_text())
    check_pack(pack)
    return pack


def check_pack(pack: dict[str, Any]) -> None:
    ids = {s["id"] for s in pack["scenarios"]}
    unknown = set(pack["mix"]) - ids
    if unknown:
        raise ValueError(f"mix names unknown scenarios: {sorted(unknown)}")
    lines = {s["line"] for s in pack["scenarios"]}
    known = {b["id"] for b in pack["businesses"]}
    for scenario in pack["scenarios"]:
        if set(scenario.get("businesses", [])) - known:
            raise ValueError(f"scenario {scenario['id']} names an unknown business")
    for business in pack["businesses"]:
        missing = lines - set(business["lines"])
        if missing:
            raise ValueError(f"{business['name']} has no {sorted(missing)} line")
        for line in business["lines"].values():
            fictional_number(business["area_code"], line["number"], BUSINESS_LINES)
            if line.get("ivr_key") and not business.get("ivr"):
                raise ValueError(f"{business['name']} has a menu key but no menu")


def fictional_number(area_code: str, local: str, lines: tuple[int, int] = (FICTIONAL_LOW, FICTIONAL_HIGH)) -> str:
    """E.164 for ``<area code> 555-01NN``; anything outside the fictional range is refused."""
    exchange, _, line = local.partition("-")
    if exchange != "555" or not line.isdigit() or not lines[0] <= int(line) <= lines[1]:
        raise ValueError(f"{local!r} is outside 555-{lines[0]:04d} to 555-{lines[1]:04d}")
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
    caller_context: str
    wanted: str
    start: datetime
    after_hours: bool
    opening: str
    length: str
    ivr_key: str | None = None
    colleague: Speaker | None = None
    colleague_role: str | None = None
    interruptions: bool = False
    people: dict[str, str] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.scenario["kind"]

    @property
    def transfer(self) -> bool:
        return self.colleague is not None

    def features(self) -> dict[str, str]:
        """What varies about this call, for tags and the review page."""
        f = {"opening": self.opening, "length": self.length}
        if self.ivr_key:
            f["ivr"] = self.ivr_key
        if self.transfer:
            f["transfer"] = "true"
        if self.interruptions:
            f["interruptions"] = "true"
        return f

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["scenario"] = self.scenario["id"]
        d["business"] = self.business["name"]
        return d


def allocate(mix: dict[str, float], count: int) -> list[str]:
    """Scenario ids for ``count`` calls in proportion to ``mix`` (largest remainder), each at least
    once when there is room."""
    total = sum(mix.values())
    exact = {k: count * w / total for k, w in mix.items()}
    counts = {k: int(v) for k, v in exact.items()}
    if count >= len(mix):
        for k in counts:
            counts[k] = max(counts[k], 1)
    while sum(counts.values()) < count:
        k = max(exact, key=lambda k: exact[k] - counts[k])
        counts[k] += 1
    while sum(counts.values()) > count:
        k = max((k for k in counts if counts[k] > 1), key=lambda k: counts[k] - exact[k])
        counts[k] -= 1
    return [k for k, n in counts.items() for _ in range(n)]


def build_plan(pack: dict[str, Any], *, count: int = 20, seed: int = 20261123) -> list[CallPlan]:
    """The calls a run makes: ``count`` calls in the pack's mix, in a shuffled order."""
    rng = random.Random(f"{pack['industry']}-{seed}")
    scenarios = {s["id"]: s for s in pack["scenarios"]}
    order = allocate(pack["mix"], count)
    rng.shuffle(order)

    tz = ZoneInfo(pack["timezone"])
    businesses = pack["businesses"]
    used_first: set[str] = set()
    used_last: set[str] = set()
    voice_turn = {"f": 0, "m": 0}

    def next_voice(gender: str, avoid: set[str]) -> str:
        pool = FEMALE_VOICES if gender == "f" else MALE_VOICES
        for _ in range(len(pool)):
            voice = pool[voice_turn[gender] % len(pool)]
            voice_turn[gender] += 1
            if voice not in avoid:
                return voice
        return pool[0]

    def first_name(gender: str) -> str:
        pool = fake_names.female_names if gender == "f" else fake_names.male_names
        choices = [n for n in pool if n not in used_first] or pool
        name = rng.choice(choices)
        used_first.add(name)
        return name

    def last_name() -> str:
        choices = [n for n in fake_names.last_names if n not in used_last] or fake_names.last_names
        name = rng.choice(choices)
        used_last.add(name)
        return name

    plan: list[CallPlan] = []
    for i, sid in enumerate(order, start=1):
        scenario = scenarios[sid]
        kind = scenario["kind"]
        line = scenario["line"]
        pool = [b for b in businesses if b["id"] in scenario.get("businesses", [b["id"]])]  # e.g. dental only
        if kind == "robocall":  # screening exists for these: put them on a number that screens
            pool = [b for b in pool if b["lines"][line].get("screen")] or pool
        business = pool[(i - 1) % len(pool)]
        line_cfg = business["lines"][line]
        caller_line = CALLER_LINES[0] + (i - 1) % (CALLER_LINES[1] - CALLER_LINES[0] + 1)
        caller_area = pack["caller_area_codes"][(i - 1) % len(pack["caller_area_codes"])]

        agent_gender = rng.choice("fm")
        agent = Speaker(first_name(agent_gender), agent_gender, next_voice(agent_gender, set()), round(rng.uniform(0.95, 1.08), 2))
        if kind == "robocall":
            caller = Speaker(None, "m", ROBOCALL_VOICE, 1.05)
        else:
            gender = rng.choice("fm")
            first, last = first_name(gender), last_name()
            if f"{first} {last}" in FAMOUS_NAMES:  # redrawn apart from the plan's own draws
                other = random.Random(f"{pack['industry']}-{seed}-{i}-name")
                last = other.choice([n for n in fake_names.last_names if f"{first} {n}" not in FAMOUS_NAMES and n not in used_last])
            caller = Speaker(f"{first} {last}", gender, next_voice(gender, {agent.voice}), round(rng.uniform(0.92, 1.1), 2))

        after_hours = kind == "conversation" and scenario.get("after_hours", False)
        length = FIXED_LENGTH.get(kind) or rng.choices(list(CONVERSATION_LENGTHS), list(CONVERSATION_LENGTHS.values()))[0]
        opening = "voicemail" if kind == "voicemail" else rng.choice(list(OPENINGS))
        ivr_key = line_cfg.get("ivr_key") if kind in ("conversation", "hangup", "wrong_number") and not after_hours else None

        colleague = colleague_role = None
        interruptions = False
        if kind == "conversation" and length != "short" and not after_hours and scenario.get("transfer_to"):
            if rng.random() < TRANSFER_SHARE:
                g = rng.choice("fm")
                colleague = Speaker(first_name(g), g, next_voice(g, {agent.voice, caller.voice}), round(rng.uniform(0.95, 1.08), 2))
                colleague_role = scenario["transfer_to"]
        if kind == "conversation" and length != "short":
            interruptions = rng.random() < INTERRUPTION_SHARE

        people = {}
        for role in scenario.get("people", []):
            form, _ = PEOPLE_FORMS[role]
            people[role] = form.format(first=first_name(rng.choice("fm")), last=last_name())

        context = pack["caller_context"]
        items = business["products"] if line in context.get("own_products_on", []) else context["items"]
        years = context.get("years")
        item = rng.choice(items)
        if business.get("caller_items"):  # e.g. parents at a pediatric practice; own generator, so the plan holds
            item = random.Random(f"{pack['industry']}-{seed}-{i}-caller").choice(business["caller_items"])
        caller_context = context["template"].format(item=item, year=rng.randint(*years) if years else "").replace("  ", " ")
        if caller.name and item in context.get("family_items", []):
            # The relative the call is about gets a name too, or the writer invents one. Drawn from
            # its own generator so the rest of the plan does not change.
            family = random.Random(f"{pack['industry']}-{seed}-{i}-family")
            pool = fake_names.female_names if family.random() < 0.5 else fake_names.male_names
            surname = caller.name.split()[-1]
            first = family.choice([
                n for n in pool if n not in (caller.name.split()[0], agent.name) and f"{n} {surname}" not in FAMOUS_NAMES
            ])
            people["patient"] = PEOPLE_FORMS["patient"][0].format(first=first, last=caller.name.split()[-1])

        plan.append(
            CallPlan(
                index=i,
                call_id=f"{pack['industry']}-{i:02d}-{sid.replace('_', '-')}",
                industry=pack["industry"],
                scenario=scenario,
                business=business,
                line=line,
                business_tel=fictional_number(business["area_code"], line_cfg["number"], BUSINESS_LINES),
                screened=bool(line_cfg.get("screen")) and not after_hours,
                caller_tel=fictional_number(caller_area, f"555-{caller_line:04d}", CALLER_LINES),
                caller=caller,
                agent=agent,
                caller_context=caller_context,
                wanted=rng.choice(business["products"]),
                start=_call_time(rng, tz, i, after_hours, pack.get("open_days", [0, 1, 2, 3, 4, 5])),
                after_hours=after_hours,
                opening=opening,
                length=length,
                ivr_key=ivr_key,
                colleague=colleague,
                colleague_role=colleague_role,
                interruptions=interruptions,
                people=people,
            )
        )
    return plan


def _call_time(rng: random.Random, tz: ZoneInfo, i: int, after_hours: bool, open_days: list[int]) -> datetime:
    """A time in the two weeks from FIRST_DAY: open hours, or an evening or closed day."""
    first = datetime.fromisoformat(FIRST_DAY).replace(tzinfo=tz)
    days = [first + timedelta(days=d) for d in range(14)]
    if after_hours:
        closed = [d for d in days if d.weekday() not in open_days]
        evenings = [d for d in days if d.weekday() in open_days]
        if closed and rng.random() < 0.4:
            return rng.choice(closed).replace(hour=rng.randint(9, 16), minute=rng.randint(0, 59), second=rng.randint(0, 59))
        return rng.choice(evenings).replace(hour=rng.choice([19, 20, 21]), minute=rng.randint(0, 59), second=rng.randint(0, 59))
    open_ = [d for d in days if d.weekday() in open_days]
    day = open_[(i - 1) % len(open_)]
    return day.replace(hour=rng.randint(8, 16), minute=rng.randint(0, 59), second=rng.randint(0, 59))
