"""Script hygiene: refuse a script that could name a real person, business, number or address.

The writer is told exactly which names it may use. This module checks that it did:

* every phone number is in the fictional 555-0100 to 555-0199 range;
* no email address, web address or "dot com";
* no name from a list of real dealer groups, car retailers, parts chains, valuation sites,
  lenders and insurers;
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

# Real businesses a dealership call could plausibly drop in. Matched as whole words, ignoring case.
REAL_BUSINESSES = [
    "AutoNation", "CarMax", "Carvana", "Vroom", "Penske", "Lithia", "Hendrick", "Sonic Automotive",
    "Group 1", "Asbury", "Jiffy Lube", "Pep Boys", "AutoZone", "O'Reilly", "Advance Auto",
    "NAPA", "Firestone", "Midas", "Meineke", "Valvoline", "Discount Tire", "Les Schwab",
    "Kelley Blue Book", "KBB", "Carfax", "Edmunds", "TrueCar", "Cars.com", "Autotrader", "CarGurus",
    "Ally", "Capital One", "Chase", "Wells Fargo", "Bank of America", "Credit Karma",
    "Geico", "Progressive", "State Farm", "Allstate", "Liberty Mutual", "AAA",
    "Enterprise", "Hertz", "Avis", "Budget", "Uber", "Lyft",
    "Google", "Amazon", "Facebook", "Craigslist", "Venmo", "PayPal", "Zelle",
    "Toyota Financial", "Ford Credit", "Honda Financial", "Subaru Motors Finance",
]

DEPARTMENT_WORDS = {
    "sales", "service", "parts", "finance", "department", "desk", "counter", "office", "manager",
    "showroom", "center", "express", "lube", "collision", "body", "shop", "team", "the",
    "and", "of", "auto", "motors", "group", "dealership", "dealer",
}

PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?(\d{3})[\s.-]?(\d{4})(?!\d)")
EMAIL = re.compile(r"\S+@\S+|\bat\s+\w+\s+dot\s+(?:com|net|org)\b", re.I)
WEB = re.compile(r"\bwww\b|https?://|\b\w+\.(?:com|net|org|biz|us|auto|io)\b|\bdot\s+com\b", re.I)
# Case-sensitive: "budget", "chase" and "progressive" are ordinary words in lower case.
REAL = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in REAL_BUSINESSES) + r")\b")


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
) -> Verdict:
    """Problems that reject the script, and notes that only get recorded."""
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
    for m in PHONE.finditer(text):
        exchange, line = m.group(1), m.group(2)
        if not (exchange == "555" and 100 <= int(line) <= 199):
            v.problems.append(f"phone number outside 555-0100..0199: {m.group(0).strip()!r}")
    if EMAIL.search(text):
        v.problems.append("mentions an email address")
    if WEB.search(text):
        v.problems.append("mentions a web address")
    for m in REAL.finditer(text):
        v.problems.append(f"names a real business: {m.group(0)!r}")

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
            if len(ent.split()) == 1:
                # one word ("Front", "Recall") is usually a common noun to spaCy's small model;
                # a real business named in one word is caught by REAL above
                v.notes.append(f"possible organisation: {ent!r}")
            else:
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
    names; real businesses written as acronyms are caught by ``REAL`` instead.
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
