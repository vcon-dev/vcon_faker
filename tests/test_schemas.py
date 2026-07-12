"""Tests for Pydantic conversation schemas and structured parsing."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vconfaker.schemas.conversation import ConversationResponse, parse_conversation


def test_conversation_response_validates():
    payload = {
        "conversation": [
            {"speaker": "Agent", "message": "Hello"},
            {"speaker": "Customer", "message": "Hi there"},
        ]
    }
    response = ConversationResponse.model_validate(payload)
    assert len(response.conversation) == 2


def test_conversation_response_rejects_bad_speaker():
    with pytest.raises(ValidationError):
        ConversationResponse.model_validate(
            {"conversation": [{"speaker": "Bot", "message": "nope"}]}
        )


def test_parse_conversation_lenient_filter():
    raw = [
        {"speaker": "Agent", "message": "hi"},
        {"speaker": "Bot", "message": "x"},
        {"speaker": "Customer", "message": "  "},
        "nope",
    ]
    assert parse_conversation(raw) == [{"speaker": "Agent", "message": "hi"}]
