"""FastAPI surface tests (skipped if fastapi is not installed)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from vcon import Vcon  # noqa: E402

from vconfaker.api import create_app  # noqa: E402
from vconfaker.config import Settings  # noqa: E402


def _client(tmp_path) -> TestClient:
    settings = Settings(llm_provider="scripted", storage_backend="local", output_dir=str(tmp_path))
    return TestClient(create_app(settings))


def test_healthz(tmp_path):
    assert _client(tmp_path).get("/healthz").json() == {"status": "ok"}


def test_generate_endpoint_writes_valid_vcons(tmp_path):
    resp = _client(tmp_path).post(
        "/generate", json={"count": 2, "conversation_type": "messaging", "provider": "scripted"}
    )
    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 2

    files = list(Path(tmp_path).rglob("*.vcon.json"))
    assert len(files) == 2
    assert Vcon(json.loads(files[0].read_text())).is_valid()[0]


def test_generate_unknown_provider_returns_400(tmp_path):
    resp = _client(tmp_path).post("/generate", json={"provider": "bogus"})
    assert resp.status_code == 400
