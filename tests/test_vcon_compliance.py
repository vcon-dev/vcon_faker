"""vCon spec-compliance gate for vcon_faker output.

These tests build vCons through ``vcon_builder.build_vcon`` (no network, no
Streamlit) and assert the IETF ``draft-ietf-vcon-vcon-core-02`` / speckit
non-negotiables. They are the CI gate: any regression that reintroduces a legacy
field name or a malformed attachment/analysis fails the build.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import pytest
from vcon import Vcon

import vcon_builder as vb

CONVERSATION = [
    {"speaker": "Agent", "message": "Hello, this is Pat at Acme Insurance. May I have your name?"},
    {"speaker": "Customer", "message": "Hi Pat, this is Jamie. I have a question about my bill."},
    {"speaker": "Agent", "message": "Of course, Jamie. Can you confirm the last four of your account?"},
    {"speaker": "Customer", "message": "Sure, it is one two three four."},
    {"speaker": "Agent", "message": "Thank you. I have resolved the billing issue. Have a great day."},
]

COMMON = dict(
    agent_name="Pat Rivera",
    customer_name="Jamie Lin",
    agent_phone="+15551112222",
    customer_phone="+15553334444",
    agent_email="pat.rivera@acme.com",
    customer_email="jamie.lin@gmail.com",
    business_name="Acme Insurance",
    business="Insurance Agency",
    problem="billing question",
    emotion="frustrated",
    conversation=CONVERSATION,
    conversation_date=datetime(2026, 1, 2, 15, 30, tzinfo=UTC),
    chat_model="gpt-4o",
    tts_model="tts-1",
)


def _voice_with_audio() -> Vcon:
    return vb.build_vcon(
        **COMMON,
        conversation_type="voice",
        generate_audio=True,
        audio_url="https://example.com/2026/01/02/call.mp3",
        audio_filename="call.mp3",
        audio_content_hash=vb.sha512_base64url(b"fake-audio-bytes"),
        audio_duration=42.0,
    )


def _voice_no_audio() -> Vcon:
    return vb.build_vcon(**COMMON, conversation_type="voice", generate_audio=False)


def _messaging() -> Vcon:
    return vb.build_vcon(**COMMON, conversation_type="messaging")


def _email() -> Vcon:
    return vb.build_vcon(**COMMON, conversation_type="email")


def _incomplete() -> Vcon:
    return vb.build_vcon(
        **{**COMMON, "conversation": []},
        conversation_type="incomplete",
        disposition="busy",
    )


TRANSFER_CONVERSATION = [
    *CONVERSATION[:3],
    {"speaker": "Agent", "message": "Let me transfer you to my supervisor Sam Lee."},
    {"speaker": "Supervisor", "message": "Hello Jamie, this is Sam Lee. I have resolved your billing issue."},
    {"speaker": "Customer", "message": "Thank you Sam."},
]


def _transfer() -> Vcon:
    return vb.build_vcon(
        **{**COMMON, "conversation": TRANSFER_CONVERSATION},
        conversation_type="transfer",
        supervisor_name="Sam Lee",
    )


ALL_BUILDERS = [_voice_with_audio, _voice_no_audio, _messaging, _email, _incomplete, _transfer]
_ids = [b.__name__ for b in ALL_BUILDERS]


def _all_keys(obj) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _all_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _all_keys(item)
    return keys


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=_ids)
def test_is_valid(build):
    ok, errors = build().is_valid()
    assert ok, f"is_valid failed: {errors}"


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=_ids)
def test_syntax_version(build):
    assert build().to_dict()["vcon"] == "0.4.0"


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=_ids)
def test_no_legacy_field_names(build):
    keys = _all_keys(build().to_dict())
    for legacy in ("schema_version", "appended", "must_support", "mimetype"):
        assert legacy not in keys, f"legacy field present: {legacy}"


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=_ids)
def test_parties_marked_synthetic(build):
    parties = build().to_dict()["parties"]
    assert parties
    for p in parties:
        assert p.get("validation") == "synthetic"
        assert "did" not in p


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=_ids)
def test_dialogs_have_no_legacy_media_fields(build):
    for d in build().to_dict()["dialog"]:
        assert "alg" not in d and "signature" not in d
        assert "mimetype" not in d


def test_external_media_has_spec_content_hash():
    d0 = _voice_with_audio().to_dict()["dialog"][0]
    assert d0["mediatype"] == "audio/mpeg"
    assert d0["url"]
    assert re.match(r"^sha512-[A-Za-z0-9_-]+$", d0["content_hash"]), d0["content_hash"]


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=_ids)
def test_analysis_compliance(build):
    for a in build().to_dict().get("analysis", []):
        assert a.get("vendor"), "analysis entry missing required vendor"
        assert "schema_version" not in a, "use 'schema', never 'schema_version'"
        assert isinstance(a["body"], str), "analysis body must be a string"
        if a.get("encoding") == "json":
            json.loads(a["body"])  # JSON body must parse


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=_ids)
def test_attachments_use_purpose_not_type(build):
    attachments = build().to_dict().get("attachments", [])
    purposes = {a.get("purpose") for a in attachments}
    assert "tags" in purposes and "lawful_basis" in purposes
    for a in attachments:
        assert "type" not in a, "attachment must use 'purpose', not 'type'"
        assert a.get("party") == 0 and a.get("dialog") == 0
        assert isinstance(a["body"], str)


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=_ids)
def test_lawful_basis_and_synthetic_markers(build):
    d = build().to_dict()
    assert "lawful_basis" in d.get("extensions", [])
    lb = next(a for a in d["attachments"] if a["purpose"] == "lawful_basis")
    body = json.loads(lb["body"])
    assert body["lawful_basis"] == "legitimate_interests"
    assert body["terms_of_service"].endswith("CONSENT.md")
    tags = next(a for a in d["attachments"] if a["purpose"] == "tags")
    assert json.loads(tags["body"]).get("synthetic") == "true"


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=_ids)
def test_json_round_trip_revalidates(build):
    d = json.loads(build().to_json())
    ok, errors = Vcon(d).is_valid()
    assert ok, f"round-trip is_valid failed: {errors}"


@pytest.mark.parametrize("build", [_voice_with_audio, _voice_no_audio], ids=["with_audio", "no_audio"])
def test_voice_transcript_is_a_wtf_document(build):
    d = build().to_dict()
    transcripts = [a for a in d["analysis"] if a["type"] in ("transcript", "transcription")]
    assert len(transcripts) == 1
    t = transcripts[0]
    assert t["encoding"] == "json"
    assert "wtf" in t["schema"].lower()
    assert t.get("vendor")
    body = json.loads(t["body"])
    assert body["transcript"]["text"]
    assert body["segments"]
    assert "wtf_transcription" in d.get("extensions", [])


def test_voice_no_audio_has_per_turn_dialogs():
    dialogs = _voice_no_audio().to_dict()["dialog"]
    assert len(dialogs) == len(CONVERSATION)
    assert all(d["type"] == "text" for d in dialogs)
    starts = [d["start"] for d in dialogs]
    assert starts == sorted(starts)


def test_incomplete_has_single_disposition_dialog():
    d = _incomplete().to_dict()
    assert len(d["dialog"]) == 1
    dlg = d["dialog"][0]
    assert dlg["type"] == "incomplete"
    assert dlg["disposition"] == "busy"
    assert "body" not in dlg or not dlg["body"]
    assert not d.get("analysis") or all(a["type"] != "transcription" for a in d["analysis"])


def test_transfer_has_three_parties_and_transfer_dialog():
    d = _transfer().to_dict()
    assert len(d["parties"]) == 3
    assert d["parties"][2]["name"] == "Sam Lee"
    transfers = [x for x in d["dialog"] if x["type"] == "transfer"]
    assert len(transfers) == 1
    texts = [x for x in d["dialog"] if x["type"] == "text"]
    assert len(texts) == len(TRANSFER_CONVERSATION)
    # Transfer dialog sits immediately before the first Supervisor turn.
    idx = d["dialog"].index(transfers[0])
    assert d["dialog"][idx + 1]["body"].startswith("Hello Jamie, this is Sam Lee")


def test_email_dialogs_are_hours_apart():
    dialogs = _email().to_dict()["dialog"]
    assert len(dialogs) == len(CONVERSATION)
    first = datetime.fromisoformat(dialogs[0]["start"])
    second = datetime.fromisoformat(dialogs[1]["start"])
    assert (second - first).total_seconds() >= 3600


def test_empty_conversation_raises():
    args = {**COMMON, "conversation": []}
    with pytest.raises(ValueError):
        vb.build_vcon(**args)


def test_unknown_conversation_type_raises():
    with pytest.raises(ValueError):
        vb.build_vcon(**COMMON, conversation_type="carrier-pigeon")
