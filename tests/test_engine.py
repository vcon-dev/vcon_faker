"""Unit tests for the headless engine: config, providers, generator, storage.

All offline (the ScriptedProvider needs no network), so these run anywhere the
``vcon`` library is installed.
"""
from __future__ import annotations

import pytest

from vconfaker.config import Settings
from vconfaker.generators.conversation import build_prompt, generate_conversation
from vconfaker.models import ConversationSpec
from vconfaker.providers.llm import (
    OpenAIProvider,
    ScriptedProvider,
    get_provider,
    validate_turns,
)
from vconfaker.storage import LocalStorage, get_storage

SPEC = ConversationSpec(
    agent_name="Pat Rivera",
    customer_name="Jamie Lin",
    business="Insurance Agency",
    business_name="Acme Insurance",
    problem="a billing question",
    emotion="frustrated",
    conversation_type="voice",
)


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("VCONFAKER_STORAGE", "s3")
    s = Settings.from_env()
    assert s.openai_model == "gpt-test"
    assert s.storage_backend == "s3"


def test_settings_overrides_skip_none():
    s = Settings.from_env(llm_provider=None, output_dir="/tmp/x")
    assert s.output_dir == "/tmp/x"
    assert s.llm_provider  # not clobbered by a None override


def test_validate_turns_filters_bad_entries():
    raw = [
        {"speaker": "Agent", "message": "hi"},
        {"speaker": "Bot", "message": "x"},        # bad speaker
        {"speaker": "Customer", "message": "  "},   # empty
        "nope",                                      # not a dict
    ]
    assert validate_turns(raw) == [{"speaker": "Agent", "message": "hi"}]


def test_scripted_provider_default_turns():
    turns = ScriptedProvider().generate(SPEC, "prompt")
    assert len(turns) >= 4
    assert all(t["speaker"] in {"Agent", "Customer"} for t in turns)
    assert any(SPEC.agent_name in t["message"] for t in turns)


def test_scripted_provider_canned_turns():
    canned = [{"speaker": "Agent", "message": "hello"}]
    assert ScriptedProvider(canned).generate(SPEC, "p") == canned


def test_get_provider_scripted():
    assert isinstance(get_provider(Settings(llm_provider="scripted")), ScriptedProvider)


def test_get_provider_openai_requires_key():
    with pytest.raises(ValueError):
        get_provider(Settings(llm_provider="openai", openai_api_key=None))


def test_get_provider_openai_with_key():
    assert isinstance(get_provider(Settings(openai_api_key="sk-x")), OpenAIProvider)


def test_get_provider_unknown_name():
    with pytest.raises(ValueError):
        get_provider(Settings(), name="bogus")


def test_build_prompt_includes_names_and_scenario():
    prompt = build_prompt(SPEC)
    assert "Pat Rivera" in prompt and "Jamie Lin" in prompt
    assert "Acme Insurance" in prompt and "billing" in prompt
    assert "frustrated" in prompt


def test_generate_conversation_with_scripted():
    assert len(generate_conversation(SPEC, ScriptedProvider())) >= 4


def test_generate_conversation_empty_raises():
    class Empty(ScriptedProvider):
        def generate(self, spec, prompt):
            return []

    with pytest.raises(ValueError):
        generate_conversation(SPEC, Empty())


def test_local_storage_writes_and_returns_uri(tmp_path):
    store = LocalStorage(str(tmp_path))
    loc = store.save("a/b/c.json", b"hello", "application/json")
    assert loc.startswith("file://")
    assert (tmp_path / "a" / "b" / "c.json").read_bytes() == b"hello"


def test_get_storage_local(tmp_path):
    assert isinstance(
        get_storage(Settings(storage_backend="local", output_dir=str(tmp_path))),
        LocalStorage,
    )


def test_get_storage_s3_requires_bucket():
    with pytest.raises(ValueError):
        get_storage(Settings(storage_backend="s3", s3_bucket=None))


def test_get_storage_unknown_backend():
    with pytest.raises(ValueError):
        get_storage(Settings(storage_backend="bogus"))
