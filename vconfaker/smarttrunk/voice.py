"""Local text-to-speech with Kokoro-82M through mlx-audio (Apple silicon).

The model is loaded once per run and kept: about 330 MB of weights. Each turn is spoken with
its speaker's voice at 24 kHz; the telephone chain brings it down to 8 kHz afterwards.
"""
from __future__ import annotations

import re
import time

import numpy as np

from .hygiene import PHONE

KOKORO_REPO = "prince-canuma/Kokoro-82M"
SAMPLE_RATE = 24_000


_DIGIT_WORDS = "zero one two three four five six seven eight nine".split()


def speakable(text: str) -> str:
    """Phone numbers spelled out digit by digit, as people say them.

    Kokoro reads a written number such as "555-0151" as a different number, so the audio would
    carry a number outside the fictional range. Spelled digits come back right.
    """

    def spell(m: re.Match[str]) -> str:
        groups = re.findall(r"\d+", m.group(0))
        return ", ".join(" ".join(_DIGIT_WORDS[int(d)] for d in g) for g in groups)

    return PHONE.sub(spell, text)


class Voices:
    def __init__(self, repo: str = KOKORO_REPO) -> None:
        self.repo = repo
        self.model = None
        self.load_seconds = 0.0
        self.spoken_seconds = 0.0  # audio produced
        self.busy_seconds = 0.0  # wall time spent producing it

    def _load(self) -> None:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        mx.set_cache_limit(256 * 1024 * 1024)  # keep freed buffers from piling up between turns
        t0 = time.monotonic()
        self.model = load_model(self.repo)
        self.load_seconds += time.monotonic() - t0

    def release(self) -> None:
        """Drop the model and MLX's buffer cache, so the recognizer has the memory."""
        import gc

        self.model = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001 - best effort
            pass

    def say(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        if self.model is None:
            self._load()
        text = speakable(text)
        t0 = time.monotonic()
        chunks = [
            np.asarray(r.audio, dtype=np.float32).reshape(-1)
            for r in self.model.generate(text, voice=voice, speed=speed, lang_code=voice[0])
        ]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        self.busy_seconds += time.monotonic() - t0
        self.spoken_seconds += len(audio) / SAMPLE_RATE
        return audio

    @property
    def realtime_factor(self) -> float:
        """Seconds of speech per second of work (higher is faster)."""
        return round(self.spoken_seconds / self.busy_seconds, 1) if self.busy_seconds else 0.0

    @staticmethod
    def mlx_peak_bytes() -> int | None:
        try:
            import mlx.core as mx

            return int(mx.get_peak_memory())
        except Exception:  # noqa: BLE001 - the figure is informational
            return None
