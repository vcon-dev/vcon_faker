"""A synthetic call as a vCon in Smart Trunk's call shape (syntax 0.5.0, draft-ietf-vcon-vcon-core-05).

The shape follows what the carrier adapters in ``vcon-dev/vcon-telephony-adapters`` write for a
forwarded call through the Smart Trunk receiver:

* party 0 is the caller, party 1 the business's Smart Trunk number;
* one Session for the call, both parties on it; a ``recording`` Dialog with both parties as
  contributors, an external MP3 by ``url`` with its ``content_hash``;
* ``join`` and ``drop`` Events for each party;
* ``tags`` carrying the call's timing and the receiver's notice evidence under the
  ``annotation_*`` names the receiver sends (the notice file, its SHA-256, what it discloses,
  when it was served);
* a ``transcript`` analysis entry, and a ``voice-screen`` entry on numbers that screen calls.

What it does not copy: no ``tenant`` attachment (the loader that puts a corpus into a staging
customer adds the tenant stamp), no consent basis, consent receipt or ``consent_record``
annotation (there is no real caller to consent, so the only basis is the synthetic one from
``vconfaker.governance``), and no carrier ids. The screening result is simulated, so its
``vendor`` is ``vcon-faker``, not the screening vendor; its ``type`` and ``schema`` match the
real entry.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

from vcon import Vcon
from vcon.dialog import Dialog
from vcon.event import Event
from vcon.party import Party
from vcon.session import Session

from ..governance import GENERATOR, add_synthetic_governance, governance_problems
from .check import Check
from .pack import CallPlan

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "tests" / "appendix-c-json-schema.json"
SCREEN = {"type": "voice-screen", "vendor": GENERATOR, "product": "simulated-voice-screen", "schema": "voice-screen-result-v1"}
DATASET = "smarttrunk-synthetic"


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).isoformat(timespec="milliseconds")


def screen_result(plan: CallPlan, t0: datetime, layout: Any, rng: random.Random) -> dict[str, Any] | None:
    """The screening outcome for a screened number: clean for a person, acted for the robocall."""
    if not plan.screened:
        return None
    base = {"status": "clean", "mode": "listen", "threshold": 0.5, "windows": 2, "frames": 8, "synthetic_frames": 0}
    if plan.kind == "robocall":
        start, end = layout.turns[-1]
        decided = t0 + timedelta(seconds=max(start + 3.0, end - 1.0))
        return {
            **base, "status": "acted", "synthetic_frames": 6, "max_confidence": round(rng.uniform(0.9, 0.99), 2),
            "decided_at": _iso(decided), "action": "announce", "action_result": "accepted",
            "acted_at": _iso(decided + timedelta(seconds=0.4)), "telnyx_status": 200,
        }
    decided = t0 + timedelta(seconds=layout.answered_at + 8.0)
    return {
        **base, "max_confidence": round(rng.uniform(0.02, 0.2), 2), "decided_at": _iso(decided),
        "action": None, "action_result": None, "acted_at": None, "telnyx_status": None,
    }


def build_call_vcon(
    *,
    plan: CallPlan,
    pack: dict[str, Any],
    layout: Any,
    filename: str,
    media_url: str,
    content_hash: str,
    duration: float,
    transcript: Check,
    notice: dict[str, Any],
    screen: dict[str, Any] | None,
    run_label: str,
) -> Vcon:
    t0 = plan.start
    at = lambda seconds: _iso(t0 + timedelta(seconds=seconds))  # noqa: E731
    ended = layout.ended_at

    v = Vcon.build_new(created_at=at(ended + 5.0))
    v.add_party(Party(tel=plan.caller_tel, name=plan.caller.name) if plan.caller.name else Party(tel=plan.caller_tel))
    v.add_party(Party(tel=plan.business_tel, name=plan.business["name"]))

    session = v.add_session(Session(start=at(0), duration=round(ended, 3), parties=[0, 1]))
    v.add_dialog(
        Dialog(
            type="recording",
            start=at(0),
            duration=duration,
            parties=[0, 1],
            mediatype="audio/mpeg",
            filename=filename,
            url=media_url,
            content_hash=content_hash,
        ),
        session=session,
    )

    v.add_event(Event(type="join", time=at(0), sessions=session, party=0))
    v.add_event(Event(type="join", time=at(layout.answered_at), sessions=session, party=1))
    caller_drop = layout.caller_dropped_at if layout.caller_dropped_at is not None else ended
    v.add_event(Event(type="drop", time=at(caller_drop), sessions=session, party=0))
    v.add_event(Event(type="drop", time=at(ended), sessions=session, party=1))

    v.add_analysis(
        type="transcript",
        dialog=0,
        vendor="parakeet-mlx",
        product=transcript_model(),
        encoding="json",
        body={"text": transcript.text, "language": "en", "duration": duration, "segments": transcript.sentences},
    )
    if screen is not None:
        v.add_analysis(dialog=0, encoding="json", body=screen, **SCREEN)

    tags = {
        "generator": GENERATOR,
        "dataset": DATASET,
        "dataset_version": run_label,
        "industry": pack["industry"],
        "scenario": plan.scenario["id"],
        "call_id": plan.call_id,
        "call_initiated_at": at(0),
        "recording_started_at": at(0),
        "annotation_announcement": notice["file"],
        "annotation_announcement_sha256": notice["sha256"],
        "annotation_discloses": ",".join(notice["discloses"]),
        "annotation_announcement_served_at": at(layout.notice_at),
        "annotation_duration_s": f"{layout.notice_seconds:.1f}",
    }
    add_synthetic_governance(
        v,
        at=at(0),
        session=session,
        tags=tags,
        pipeline_note=(
            "Script written by an LLM from an invented scenario; voices synthesized locally with "
            "Kokoro-82M; transcript produced by speech recognition on the synthesized audio."
        ),
    )
    return v


def transcript_model() -> str:
    from .check import PARAKEET_MODEL

    return PARAKEET_MODEL


def validate(v: Vcon) -> list[str]:
    """Every problem found by vcon-lib, the core-05 JSON schema and the governance check."""
    d = json.loads(json.dumps(v.to_dict()))
    problems: list[str] = []
    if d.get("vcon") != "0.5.0":
        problems.append(f"vcon is {d.get('vcon')!r}, not '0.5.0'")
    ok, errors = Vcon(d).is_valid()
    if not ok:
        problems += [f"vcon-lib: {e}" for e in errors]
    problems += [f"vcon-lib warning: {w}" for w in Vcon(d).get_validation_warnings()]
    if SCHEMA_FILE.is_file():
        import jsonschema

        schema = json.loads(SCHEMA_FILE.read_text())
        validator = jsonschema.validators.validator_for(schema)(schema)  # the file says draft-07
        problems += [f"schema: {e.message}" for e in validator.iter_errors(d)]
    else:
        problems.append(f"core-05 schema not found at {SCHEMA_FILE}")
    problems += [f"governance: {p}" for p in governance_problems(d)]
    return problems
