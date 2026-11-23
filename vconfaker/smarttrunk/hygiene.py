"""Script hygiene: refuse a script that could name a real person, business, number or address.

The writer is told exactly which names it may use. This module checks that it did:

* every phone number, written or spoken digit by digit, is in the fictional 555-0100 to
  555-0199 range (a bare run of digits such as a stock number is not a phone number);
* no email address, web address or "dot com";
* no name from ``REAL_NAMES``: chains, banks, insurers and platforms common to every
  industry, plus the industry's own list (for dealerships: dealer groups, car retailers, service
  and parts chains, valuation sites and auto lenders);
* every person the script names (spaCy PERSON entities and the writer's own ``names_used``)
  is one of the names the call was given, and every organisation (ORG) is the business, a
  vehicle make or model, or a department.

Vehicle makes and models are allowed: they are products, not parties. Anything else is a
rejection with a reason, and the writer is asked again.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

# Real businesses a call could plausibly drop in, by industry. "common" applies to every
# industry; each pack's industry adds its own list, and a pack may add more under
# "real_businesses". Matched as whole words: names of five or more letters in any case, shorter
# names and the AMBIGUOUS words only as written.
REAL_NAMES: dict[str, tuple[str, ...]] = {
    "common": (
        # big-box retail, warehouse clubs and home improvement
        "Walmart", "Costco", "Sam's Club", "Target", "Home Depot", "Lowe's", "Best Buy", "BJ's",
        # banks, lenders and payment apps
        "Ally", "Capital One", "Chase", "Wells Fargo", "Bank of America", "Santander", "Credit Karma",
        "Venmo", "PayPal", "Zelle",
        # insurers
        "Geico", "Progressive", "State Farm", "Allstate", "Liberty Mutual", "AAA",
        # rental, rides and the big platforms
        "Enterprise", "Hertz", "Avis", "Budget", "Uber", "Lyft", "Google", "Amazon", "Facebook", "Craigslist",
    ),
    "dealership": (
        # dealer groups and car retailers, including the ones that sell direct
        "AutoNation", "CarMax", "Carvana", "Vroom", "Tesla", "Penske", "Lithia", "Hendrick",
        "Sonic Automotive", "Group 1", "Asbury",
        # service, tire, glass and collision chains
        "Jiffy Lube", "Pep Boys", "AutoZone", "O'Reilly", "Advance Auto", "NAPA", "Firestone", "Midas",
        "Meineke", "Valvoline", "Discount Tire", "Les Schwab", "Mavis", "Monro", "Take 5", "Safelite",
        "Maaco", "Caliber Collision", "Christian Brothers Automotive",
        # valuation and listing sites, auto lenders and service contracts
        "Kelley Blue Book", "KBB", "Carfax", "Edmunds", "TrueCar", "Cars.com", "Autotrader", "CarGurus",
        "Toyota Financial", "Ford Credit", "Honda Financial", "Subaru Motors Finance", "Westlake Financial",
        "Credit Acceptance", "CarShield",
    ),
    # Still to fill before those packs are written:
    "clinic": (),  # hospital systems, urgent-care and pharmacy chains, labs, insurers' plan names
    "municipal": (),  # real towns, counties and agencies, utilities, state DMV names
    "home_services": (),  # national plumbing, HVAC, pest, roofing and cleaning chains; lead marketplaces
}

# Ordinary words in lower case: matched only as written, whatever their length.
AMBIGUOUS = {"Ally", "Chase", "Budget", "Progressive", "Enterprise", "Target", "Group 1", "Take 5", "Advance Auto"}

DEPARTMENT_WORDS = {
    "sales", "service", "parts", "finance", "department", "desk", "counter", "office", "manager",
    "showroom", "center", "express", "lube", "collision", "body", "shop", "team", "the",
    "and", "of", "auto", "motors", "group", "dealership", "dealer",
}

# A phone number written the way phone numbers are written: 555-0152, 555 0152, (614) 555-0152,
# 614.555.0152, +1 614 555 0152. A bare run of digits (a stock number, 1234567) is not one.
PHONE = re.compile(
    r"(?<![\d-])(?:\+?1[\s.-])?(?:\(\d{3}\)\s?|\d{3}[\s.-])?(\d{3})[\s.-](\d{4})(?![\d-])"
)
EMAIL = re.compile(r"\S+@\S+|\bat\s+\w+\s+dot\s+(?:com|net|org)\b", re.I)
WEB = re.compile(r"\bwww\b|https?://|\b\w+\.(?:com|net|org|biz|us|auto|io)\b|\bdot\s+com\b", re.I)

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_DIGIT_RUN = re.compile(r"\b(?:(?:" + "|".join(DIGIT_WORDS) + r")\b[\s,.-]*){7,}", re.I)


def digit_words_as_digits(text: str) -> str:
    """Runs of seven or more spoken digits ("five five five, zero two zero zero") as a written
    number in phone shape (555-0200), so ``PHONE`` sees them. Shorter runs are left alone."""

    def write(m: re.Match[str]) -> str:
        d = "".join(DIGIT_WORDS[w.lower()] for w in re.findall(r"[a-z]+", m.group(0), re.I))
        if len(d) == 7:
            d = f"{d[:3]}-{d[3:]}"
        elif len(d) == 10:
            d = f"{d[:3]}-{d[3:6]}-{d[6:]}"
        elif len(d) == 11 and d[0] == "1":
            d = f"1-{d[1:4]}-{d[4:7]}-{d[7:]}"
        return d + (" " if m.group(0)[-1:] in " ,.-" else "")

    return _DIGIT_RUN.sub(write, text)


def in_fictional_range(exchange: str, line: str) -> bool:
    return exchange == "555" and 100 <= int(line) <= 199


def numbers_outside_range(text: str) -> list[str]:
    """Phone numbers, written or spoken as digits, that are outside 555-0100 to 555-0199."""
    return [
        m.group(0).strip()
        for m in PHONE.finditer(digit_words_as_digits(text))
        if not in_fictional_range(m.group(1), m.group(2))
    ]


def real_names(industry: str, extra: list[str] | tuple[str, ...] = ()) -> tuple[str, ...]:
    return (*REAL_NAMES["common"], *REAL_NAMES.get(industry, ()), *extra)


@lru_cache(maxsize=16)
def _real_patterns(names: tuple[str, ...]) -> tuple[re.Pattern[str] | None, re.Pattern[str] | None]:
    def letters(n: str) -> int:
        return sum(ch.isalpha() for ch in n)

    loose = [n for n in names if letters(n) >= 5 and n not in AMBIGUOUS]
    strict = [n for n in names if n not in loose]

    def compile_(group: list[str], flags: int) -> re.Pattern[str] | None:
        if not group:
            return None
        return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in group) + r")(?!\w)", flags)

    return compile_(loose, re.I), compile_(strict, 0)


def real_business_matches(text: str, industry: str = "dealership", extra: list[str] | tuple[str, ...] = ()) -> list[str]:
    found = []
    for pattern in _real_patterns(real_names(industry, extra)):
        if pattern is not None:
            found += [m.group(0) for m in pattern.finditer(text)]
    return found


@dataclass
class Verdict:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def allowed_words(allowed_names: list[str], products: list[str]) -> set[str]:
    words: set[str] = set(DEPARTMENT_WORDS)
    for item in [*allowed_names, *products]:
        for w in re.split(r"[\s/]+", item):
            w = w.strip(".,'\"").lower()
            if w:
                words.add(w)
                words.add(w.removesuffix("'s"))
    return words


def check_script(
    script: dict[str, Any],
    *,
    allowed_names: list[str],
    products: list[str],
    turn_range: tuple[int, int],
    industry: str = "dealership",
    extra_real: list[str] | tuple[str, ...] = (),
) -> Verdict:
    """Problems that reject the script, and notes that only get recorded.

    ``industry`` picks the list of real names to refuse (``REAL_NAMES``), and ``extra_real``
    adds a pack's own.
    """
    v = Verdict()
    turns = script.get("turns") or []
    lo, hi = turn_range
    if not lo <= len(turns) <= hi + 2:
        v.problems.append(f"{len(turns)} turns, wanted {lo} to {hi}")
    for t in turns:
        if t.get("speaker") not in ("caller", "business") or not str(t.get("text", "")).strip():
            v.problems.append("a turn has no text or an unknown speaker")
            break

    text = "\n".join(str(t.get("text", "")) for t in turns)
    for number in numbers_outside_range(text):
        v.problems.append(f"phone number outside 555-0100..0199: {number!r}")
    if EMAIL.search(text):
        v.problems.append("mentions an email address")
    if WEB.search(text):
        v.problems.append("mentions a web address")
    for name in real_business_matches(text, industry, extra_real):
        v.problems.append(f"names a real business: {name!r}")

    ok_words = allowed_words(allowed_names, products)
    person_words = allowed_words(allowed_names, [])
    for name in script.get("names_used") or []:
        if not _covered(name, ok_words):
            v.problems.append(f"names_used lists {name!r}, which the call was not given")

    turn_texts = [str(t.get("text", "")) for t in turns]
    for label, ent, opens_sentence in _entities(turn_texts):
        if label == "PERSON" and not _covered(ent, person_words | ok_words):
            if opens_sentence and len(ent.split()) == 1:
                # spaCy's small model tags sentence-opening words ("Perfect.") as people
                v.notes.append(f"possible name opening a sentence: {ent!r}")
            else:
                v.problems.append(f"names a person the call was not given: {ent!r}")
        elif label == "ORG" and not _covered(ent, ok_words):
            # one word or several: a name the call was not given is refused, at the price of
            # the occasional rewrite when spaCy's small model tags a common noun ("Front")
            v.problems.append(f"names an organisation the call was not given: {ent!r}")
        elif label in ("GPE", "LOC", "FAC") and not _covered(ent, ok_words):
            v.notes.append(f"place named: {ent!r}")
    v.problems = list(dict.fromkeys(v.problems))
    return v


# Trim levels and car words spaCy tags as names. Not names of people or businesses.
CAR_WORDS = {
    "le", "xle", "se", "xse", "limited", "premium", "sport", "touring", "wilderness", "lariat", "xlt", "xl",
    "ex", "lx", "ex-l", "platinum", "trd", "off-road", "pro", "premier", "signature", "carbon", "turbo",
    "gt", "select", "preferred", "elite", "trailsport", "edition", "hybrid", "plug-in", "base", "titanium",
    "active", "heritage", "big", "bend", "outer", "banks", "badlands", "tremor", "raptor", "st", "sv", "sl",
    "awd", "fwd", "4wd", "suv", "ev", "vin", "mpg", "eyesight", "carplay", "android", "auto",
}
# Greetings and titles spaCy glues onto a name ("Hi Dennis").
SPOKEN_WORDS = {
    "hi", "hey", "hello", "thanks", "thank", "yes", "yeah", "okay", "ok", "oh", "um", "uh", "so",
    "mr", "mrs", "ms", "dr", "sir", "ma'am", "good", "morning", "afternoon", "evening", "great",
    "perfect", "sure", "bye", "goodbye", "dear", "well", "alright", "awesome", "wonderful",
}


def _covered(name: str, ok_words: set[str]) -> bool:
    """True when every word of ``name`` is allowed, or is not a name at all.

    Words with digits (a VIN's last six, CX-5) and short all-capital acronyms (VIN, SUV) are not
    names; real businesses written as acronyms are caught by ``REAL_NAMES`` instead.
    """
    for raw in re.split(r"[\s/]+", name):
        word = raw.strip(".,'\"?!:;()")
        if not word or any(ch.isdigit() for ch in word) or (word.isupper() and len(word) <= 4):
            continue
        word = word.lower().removesuffix("'s")
        if word not in ok_words and word not in CAR_WORDS and word not in SPOKEN_WORDS:
            return False
    return True


@lru_cache(maxsize=1)
def _nlp():
    import spacy  # installed with misaki[en], which Kokoro needs anyway

    return spacy.load("en_core_web_sm", disable=["lemmatizer"])


def _entities(texts: list[str]) -> list[tuple[str, str, bool]]:
    """(label, text, whether it opens a sentence) for each named entity spaCy finds.

    Each turn is parsed on its own, so an entity never runs from one speaker into the next.
    """
    try:
        docs = list(_nlp().pipe(texts))
    except (ImportError, OSError):  # no spaCy or its model: the regex checks and names_used still apply
        return []
    return [(e.label_, e.text, e.start == e.sent.start) for doc in docs for e in doc.ents]
