"""Speech-to-text providers for optional Whisper round-trip on synthesized audio.

When ``VCONFAKER_WHISPER=1``, generated audio is transcribed through OpenAI
Whisper so WTF segments carry recognizer confidence instead of the default
source-text transcript (confidence 1.0).
"""
from __future__ import annotations

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any

from vconfaker.config import Settings
from vconfaker.models import Turn
from vconfaker.providers.tts import AudioResult
from vconfaker.vconbuild.builder import SYNTHETIC_VENDOR, build_wtf_from_turns

logger = logging.getLogger(__name__)


class STTProvider(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio: AudioResult,
        turns: list[Turn],
        *,
        created_at: str,
        tts_model: str | None = None,
    ) -> dict[str, Any]:
        """Return a WTF document dict (transcript, segments, metadata)."""
        ...


class SourceTextSTTProvider(STTProvider):
    """Use the authored conversation text (honest synthetic provenance)."""

    def transcribe(
        self,
        audio: AudioResult,
        turns: list[Turn],
        *,
        created_at: str,
        tts_model: str | None = None,
    ) -> dict[str, Any]:
        return build_wtf_from_turns(
            turns,
            audio.duration,
            provider=SYNTHETIC_VENDOR,
            model=tts_model or "unknown",
            created_at=created_at,
        )


class WhisperSTTProvider(STTProvider):
    """Transcribe synthesized audio via OpenAI Whisper (verbose JSON segments)."""

    def __init__(self, api_key: str, model: str = "whisper-1") -> None:
        self._api_key = api_key
        self._model = model

    def transcribe(
        self,
        audio: AudioResult,
        turns: list[Turn],
        *,
        created_at: str,
        tts_model: str | None = None,
    ) -> dict[str, Any]:
        from openai import OpenAI  # lazy

        client = OpenAI(api_key=self._api_key)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            tmp = handle.name
            handle.write(audio.data)
        try:
            with open(tmp, "rb") as fh:
                response = client.audio.transcriptions.create(
                    model=self._model,
                    file=fh,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        segments_raw = getattr(response, "segments", None) or []
        segments: list[dict[str, Any]] = []
        confidences: list[float] = []
        for idx, seg in enumerate(segments_raw):
            if isinstance(seg, dict):
                text = seg.get("text", "")
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                confidence = float(seg.get("avg_logprob", 0.0))
            else:
                text = getattr(seg, "text", "")
                start = float(getattr(seg, "start", 0.0))
                end = float(getattr(seg, "end", 0.0))
                confidence = float(getattr(seg, "avg_logprob", 0.0))
            # avg_logprob is log probability; map to a 0-1-ish confidence for display.
            display_conf = round(min(1.0, max(0.0, 1.0 + confidence)), 3)
            confidences.append(display_conf)
            segments.append(
                {
                    "id": idx,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text.strip(),
                    "confidence": display_conf,
                    "speaker": "",
                }
            )

        full_text = getattr(response, "text", "") or " ".join(s["text"] for s in segments)
        avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        return {
            "transcript": {
                "text": full_text.strip(),
                "language": getattr(response, "language", None) or "en",
                "duration": round(audio.duration, 3),
                "confidence": avg_conf,
            },
            "segments": segments,
            "metadata": {
                "created_at": created_at,
                "processed_at": created_at,
                "provider": "openai",
                "model": self._model,
                "note": (
                    "Whisper round-trip on TTS-synthesized audio; "
                    "confidence reflects recognizer output, not human speech."
                ),
            },
        }


def get_stt_provider(settings: Settings) -> STTProvider:
    """Resolve STT provider from settings."""
    if settings.whisper_enabled:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when VCONFAKER_WHISPER=1")
        return WhisperSTTProvider(settings.openai_api_key, settings.whisper_model)
    return SourceTextSTTProvider()
