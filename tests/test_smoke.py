"""Smoke test: generate vCons and validate each against the core-05 schema."""
import json
import os
import random
from datetime import datetime
from unittest import mock

import jsonschema
import pytest

import vcon_faker
from vcon import Vcon, compute_content_hash

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "appendix-c-json-schema.json")
with open(SCHEMA_PATH) as f:
    SCHEMA = json.load(f)

N = 10
CONVERSATION = [
    {"speaker": "Agent", "message": "Hello, this is Alex. How can I help?"},
    {"speaker": "Customer", "message": "My order has not arrived."},
    {"speaker": "Agent", "message": "I will look into that now."},
]


def _build(kind, audio, i):
    random.seed(i)
    return vcon_faker.create_vcon_object(
        agent_name="Alex Agent",
        customer_name="Casey Customer",
        agent_phone="+15555550100",
        customer_phone="+15555550101",
        agent_email="agent@example.com",
        customer_email="customer@example.com",
        url="https://example.com/a.wav" if audio else None,
        filename="a.wav" if audio else None,
        signature=compute_content_hash(b"fake audio") if audio else None,
        audio_duration=12.5 if audio else 0,
        business_name="Example Co",
        business="retail",
        problem="a late order",
        emotion="frustrated",
        generation_prompt="prompt",
        conversation=CONVERSATION,
        generate_audio=audio,
        conversation_type=kind,
        conversation_date=datetime(2025, 1, 2, 10, 0, 0),
    )


CASES = [(k, a, i) for i in range(N) for k, a in
         (("voice", True), ("voice", False), ("messaging", False))]


@pytest.mark.parametrize("kind,audio,i", CASES)
def test_generated_vcon_matches_schema(kind, audio, i):
    d = _build(kind, audio, i).to_dict()
    jsonschema.validate(d, SCHEMA)
    ref = Vcon(json.loads(json.dumps(d)))
    assert ref.is_valid() == (True, [])
    assert ref.get_validation_warnings() == []
    assert d["vcon"] == "0.5.0"
    assert len(d["sessions"]) == 1
    session = d["sessions"][0]
    assert session["dialog"] == list(range(len(d["dialog"])))
    # No removed 0.4.0 fields
    for dlg in d["dialog"]:
        for gone in ("originator", "alg", "signature", "mimetype", "party_history"):
            assert gone not in dlg
    # Dialog parties are contributors; Session parties cover all of them
    for dlg in d["dialog"]:
        assert set(dlg["parties"]) <= set(session["parties"])
    # The faker has never modelled consent, so it must not invent a basis
    assert not [a for a in d.get("attachments", []) if a.get("purpose") == "lawful_basis"]


def test_conversation_generation_is_mocked():
    payload = json.dumps({"conversation": CONVERSATION})
    fake = mock.Mock()
    fake.choices = [mock.Mock(message=mock.Mock(content=payload))]
    with mock.patch.object(vcon_faker.client.chat.completions, "create", return_value=fake) as m:
        turns = vcon_faker.generate_conversation(
            "p", "Alex", "Casey", "retail", "late order", None, "Example Co"
        )
    assert m.called
    assert len(turns) == 3
