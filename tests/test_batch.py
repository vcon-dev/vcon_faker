"""Tests for concurrent batch generation."""
from __future__ import annotations

from vconfaker.config import Settings
from vconfaker.engine import generate_many
from vconfaker.generators.scenario import random_spec
from vconfaker.providers.llm import ScriptedProvider
from vconfaker.storage import LocalStorage


def test_generate_many_concurrent(tmp_path):
    specs = [
        random_spec(
            business="Insurance Agency",
            business_name="Acme Insurance",
            problem="a billing question",
            conversation_type="messaging",
        )
        for _ in range(6)
    ]
    settings = Settings(concurrency=3)
    results = generate_many(
        specs,
        settings=settings,
        provider=ScriptedProvider(),
        storage=LocalStorage(str(tmp_path)),
        concurrency=3,
    )
    assert len(results) == 6
    assert len(list(tmp_path.rglob("*.vcon.json"))) == 6
