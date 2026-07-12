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
    AnthropicProvider,
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


def test_get_provider_anthropic_requires_key():
    with pytest.raises(ValueError):
        get_provider(Settings(llm_provider="anthropic", anthropic_api_key=None))


def test_get_provider_anthropic_with_key():
    assert isinstance(
        get_provider(Settings(llm_provider="anthropic", anthropic_api_key="sk-ant-x")),
        AnthropicProvider,
    )


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


def test_generate_vcon_uuid_matches_storage_key(tmp_path):
    from vconfaker.engine import generate_vcon

    result = generate_vcon(
        SPEC, provider=ScriptedProvider(), storage=LocalStorage(str(tmp_path))
    )
    assert result.vcon_dict["uuid"] == result.vcon_uuid
    assert result.vcon_locator.endswith(f"{result.vcon_uuid}.vcon.json")


@pytest.mark.parametrize("conv_type", ["voice", "messaging", "email", "incomplete", "transfer"])
def test_generate_vcon_all_types_offline(tmp_path, conv_type):
    from dataclasses import replace

    from vconfaker.engine import generate_vcon

    spec = replace(SPEC, conversation_type=conv_type, supervisor_name="Sam Lee")
    result = generate_vcon(
        spec, provider=ScriptedProvider(), storage=LocalStorage(str(tmp_path))
    )
    assert result.vcon_dict["uuid"] == result.vcon_uuid
    assert result.audio_locator is None  # none of these modes synthesize audio


def test_load_signing_key_generates_then_reuses(tmp_path):
    from vconfaker.engine import load_signing_key

    pem = load_signing_key(None, str(tmp_path))
    key_file = tmp_path / "vcon-faker-signing-key.pem"
    assert key_file.exists() and (tmp_path / "vcon-faker-signing-key.pem.pub").exists()
    assert b"PRIVATE KEY" in pem
    assert load_signing_key(str(key_file)) == pem


def test_generate_vcon_signed_and_verifiable(tmp_path):
    import json

    from vcon import Vcon

    from vconfaker.engine import generate_vcon, load_signing_key

    sign_key = load_signing_key(None, str(tmp_path))
    result = generate_vcon(
        SPEC,
        provider=ScriptedProvider(),
        storage=LocalStorage(str(tmp_path)),
        sign_key=sign_key,
    )
    stored = json.loads((tmp_path / result.vcon_locator.split(str(tmp_path) + "/")[1]).read_text())
    assert "signatures" in stored and "payload" in stored
    public_pem = (tmp_path / "vcon-faker-signing-key.pem.pub").read_bytes()
    assert Vcon(stored).verify(public_pem)


def test_conserver_storage_posts_json(monkeypatch):
    from vconfaker.storage import ConserverStorage

    calls: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

    def fake_post(url, **kwargs):  # noqa: ANN001
        calls["url"] = url
        calls.update(kwargs)
        return FakeResponse()

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    store = ConserverStorage("http://conserver:8000/", "tok", "faker")
    locator = store.save("2026/01/02/abc.vcon.json", b"{}", "application/json")
    assert calls["url"] == "http://conserver:8000/vcon"
    assert calls["params"] == {"ingress_list": "faker"}
    assert calls["headers"]["x-conserver-api-token"] == "tok"
    assert locator == "http://conserver:8000/vcon/abc"

    with pytest.raises(ValueError):
        store.save("a.mp3", b"...", "audio/mpeg")


def test_get_storage_conserver_requires_url():
    from vconfaker.storage import ConserverStorage

    with pytest.raises(ValueError):
        get_storage(Settings(storage_backend="conserver"))
    assert isinstance(
        get_storage(Settings(storage_backend="conserver", conserver_url="http://x")),
        ConserverStorage,
    )


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
