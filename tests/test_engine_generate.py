"""Tests for the unified generate_vcon orchestration (offline)."""
from __future__ import annotations

import pytest
from vcon import Vcon

from vconfaker.engine import generate_vcon
from vconfaker.generators.scenario import random_spec
from vconfaker.providers.llm import ScriptedProvider
from vconfaker.providers.tts import ScriptedTTSProvider
from vconfaker.storage import LocalStorage


def _spec(conv_type: str = "voice"):
    return random_spec(
        business="Insurance Agency",
        business_name="Acme Insurance",
        problem="a billing question",
        conversation_type=conv_type,
    )


@pytest.mark.parametrize("conv_type", ["voice", "messaging"])
def test_generate_vcon_no_audio(tmp_path, conv_type):
    result = generate_vcon(
        _spec(conv_type), provider=ScriptedProvider(), storage=LocalStorage(str(tmp_path))
    )
    assert result.vcon_locator.startswith("file://")
    assert result.audio_locator is None
    assert result.vcon_dict["vcon"] == "0.4.0"
    assert Vcon(result.vcon_dict).is_valid()[0]
    assert len(list(tmp_path.rglob("*.vcon.json"))) == 1


def test_generate_vcon_with_audio(tmp_path):
    result = generate_vcon(
        _spec("voice"),
        provider=ScriptedProvider(),
        storage=LocalStorage(str(tmp_path)),
        generate_audio=True,
        tts=ScriptedTTSProvider(),
        tts_model="tts-1",
    )
    assert result.audio_locator and result.audio_locator.endswith(".mp3")
    dialog = result.vcon_dict["dialog"][0]
    assert dialog["type"] == "recording"
    assert dialog["mediatype"] == "audio/mpeg"
    assert dialog["content_hash"].startswith("sha512-")
    assert Vcon(result.vcon_dict).is_valid()[0]
    assert len(list(tmp_path.rglob("*.mp3"))) == 1
    assert len(list(tmp_path.rglob("*.vcon.json"))) == 1


def test_generate_audio_requires_tts(tmp_path):
    with pytest.raises(ValueError):
        generate_vcon(
            _spec("voice"),
            provider=ScriptedProvider(),
            storage=LocalStorage(str(tmp_path)),
            generate_audio=True,
            tts=None,
        )
