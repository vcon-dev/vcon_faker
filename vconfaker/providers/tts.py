"""Text-to-speech providers for voice conversations.

* :class:`ScriptedTTSProvider` - offline stub returning fixed bytes; used by
  tests and ``--provider scripted`` so the audio path is exercisable without a
  network call or audio toolchain.
* :class:`OpenAITTSProvider` - lazily imports ``openai`` and ``pydub`` to
  synthesize per-turn speech and concatenate it to a single MP3.

Add ElevenLabs / other engines by subclassing :class:`TTSProvider`.
"""
from __future__ import annotations

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vconfaker.config import Settings
from vconfaker.models import Turn

logger = logging.getLogger(__name__)

# Per-speaker default voices (OpenAI voice names).
DEFAULT_VOICES = {"Agent": "alloy", "Customer": "nova"}


@dataclass
class AudioResult:
    data: bytes
    duration: float
    mediatype: str = "audio/mpeg"


class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, turns: list[Turn], *, voices: dict[str, str] | None = None) -> AudioResult:
        ...


class ScriptedTTSProvider(TTSProvider):
    """Offline stub: fixed bytes, duration estimated from the turn count."""

    def __init__(self, data: bytes = b"\x00", seconds_per_turn: float = 3.0) -> None:
        self._data = data or b"\x00"
        self._seconds_per_turn = seconds_per_turn

    def synthesize(self, turns: list[Turn], *, voices: dict[str, str] | None = None) -> AudioResult:
        return AudioResult(data=self._data, duration=round(len(turns) * self._seconds_per_turn, 3))


class OpenAITTSProvider(TTSProvider):
    """Synthesize each turn via OpenAI TTS and concatenate to one MP3."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def synthesize(self, turns: list[Turn], *, voices: dict[str, str] | None = None) -> AudioResult:
        from openai import OpenAI  # lazy
        from pydub import AudioSegment  # lazy

        voices = voices or DEFAULT_VOICES
        client = OpenAI(api_key=self._api_key)
        combined = AudioSegment.silent(duration=0)
        for turn in turns:
            voice = voices.get(turn["speaker"], "alloy")
            response = client.audio.speech.create(
                model=self._model, voice=voice, input=turn["message"], response_format="mp3"
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
                tmp = handle.name
            try:
                response.stream_to_file(tmp)
                combined += AudioSegment.from_file(tmp, format="mp3")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            out = handle.name
        try:
            combined.export(out, format="mp3")
            with open(out, "rb") as fh:
                data = fh.read()
        finally:
            if os.path.exists(out):
                os.remove(out)

        return AudioResult(data=data, duration=round(len(combined) / 1000.0, 3))


def get_tts_provider(settings: Settings, name: str | None = None) -> TTSProvider:
    """Resolve a :class:`TTSProvider` from settings (or an explicit name)."""
    name = name or "openai"
    if name in ("scripted", "fake", "silent"):
        return ScriptedTTSProvider()
    if name == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured for the OpenAI TTS provider")
        return OpenAITTSProvider(settings.openai_api_key, settings.openai_tts_model)
    raise ValueError(f"unknown tts provider: {name!r}")
