"""Tests for STT providers (offline)."""
from __future__ import annotations

from vconfaker.providers.stt import SourceTextSTTProvider
from vconfaker.providers.tts import ScriptedTTSProvider

TURNS = [
    {"speaker": "Agent", "message": "Hello"},
    {"speaker": "Customer", "message": "Hi"},
]


def test_source_text_stt_builds_wtf():
    audio = ScriptedTTSProvider().synthesize(TURNS)
    wtf = SourceTextSTTProvider().transcribe(
        audio, TURNS, created_at="2026-01-01T12:00:00+00:00", tts_model="tts-1"
    )
    assert wtf["transcript"]["confidence"] == 1.0
    assert len(wtf["segments"]) == 2
    assert wtf["metadata"]["provider"] == "vcon-faker"
