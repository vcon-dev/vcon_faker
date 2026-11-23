"""Turn spoken turns into a phone call recording.

The call is laid out at 24 kHz: ringback, the receiver's recording notice (Smart Trunk plays it
as ringback, before anyone answers), more ringback, then the conversation with short gaps.
ffmpeg makes the tones and does the telephone part: light compression, a little pink line
noise, a 300 to 3400 Hz band, G.711 mu-law at 8 kHz mono, then MP3.

Real Smart Trunk recordings start at answer, so the notice is not in them; here the notice is
kept at the start of the recording so a listener hears what the caller heard. The vCon's
``recording_started_at`` and ``annotation_announcement_served_at`` tags say where it sits.
"""
from __future__ import annotations

import hashlib
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .voice import SAMPLE_RATE

FFMPEG = "ffmpeg"
MP3_BITRATE = "32k"
TELEPHONE_BAND = "if(lt(f,300),-50,if(gt(f,3400),-50,0))"  # dB by frequency


def _ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _tone(path: Path, expr: str, seconds: float) -> np.ndarray:
    if not path.exists():
        _ffmpeg("-f", "lavfi", "-i", f"aevalsrc={expr}:s={SAMPLE_RATE}:d={seconds}", "-ac", "1", str(path))
    audio, _ = sf.read(path, dtype="float32")
    return audio


@dataclass
class Tones:
    """US ringback (440 + 480 Hz), the voicemail tone (1000 Hz) and a hang-up click."""

    workdir: Path
    ring: np.ndarray = field(init=False)
    beep: np.ndarray = field(init=False)
    click: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.ring = _tone(self.workdir / "ringback.wav", "0.08*sin(2*PI*440*t)+0.08*sin(2*PI*480*t)", 2.0)
        self.beep = _tone(self.workdir / "beep.wav", "0.25*sin(2*PI*1000*t)", 0.45)
        self.click = _tone(self.workdir / "click.wav", "0.3*exp(-60*t)*sin(2*PI*180*t)", 0.08)


@dataclass
class Layout:
    """Where things sit in the recording, in seconds from its start (the call start)."""

    notice_at: float = 0.0
    notice_seconds: float = 0.0
    answered_at: float = 0.0
    ended_at: float = 0.0
    caller_dropped_at: float | None = None
    turns: list[tuple[float, float]] = field(default_factory=list)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def _gain(audio: np.ndarray, db: float) -> np.ndarray:
    return audio * (10 ** (db / 20))


def lay_out(
    *,
    kind: str,
    notice: np.ndarray,
    turns: list[tuple[str, np.ndarray]],
    tones: Tones,
    rng: random.Random,
    system_audio: np.ndarray | None = None,
) -> tuple[np.ndarray, Layout]:
    """The call at 24 kHz and its layout. ``turns`` are (speaker, audio) in order."""
    parts: list[np.ndarray] = []
    layout = Layout()
    clock = 0.0

    def put(audio: np.ndarray) -> float:
        nonlocal clock
        at = clock
        parts.append(audio)
        clock += len(audio) / SAMPLE_RATE
        return at

    caller_db = rng.uniform(-3.0, 1.0)  # each caller's line is a little different
    put(_silence(0.3))
    put(tones.ring)
    put(_silence(0.5))
    layout.notice_at = put(notice)
    layout.notice_seconds = len(notice) / SAMPLE_RATE
    put(_silence(0.4))
    put(tones.ring)
    if kind == "voicemail":
        put(_silence(2.0))
        put(tones.ring)
    put(_silence(rng.uniform(0.3, 0.8)))
    layout.answered_at = clock

    for i, (speaker, audio) in enumerate(turns):
        if i:
            if kind == "hangup" and i == len(turns) - 1:
                put(_silence(rng.uniform(5.0, 7.0)))  # left on hold
            else:
                put(_silence(rng.uniform(0.25, 0.7)))
        start = put(_gain(audio, caller_db) if speaker == "caller" else audio)
        layout.turns.append((start, clock))
        if kind == "voicemail" and i == 0:
            put(_silence(0.3))
            put(tones.beep)
            put(_silence(0.2))

    if kind == "robocall" and system_audio is not None:
        put(_silence(0.2))
        put(system_audio)  # the screening announcement cuts in
        layout.caller_dropped_at = clock
    put(_silence(0.3))
    if kind in ("hangup", "voicemail", "robocall"):
        layout.caller_dropped_at = clock
        put(tones.click)
    put(_silence(0.5))
    layout.ended_at = clock
    return np.concatenate(parts), layout


def to_phone_mp3(audio: np.ndarray, workdir: Path, name: str, rng: random.Random) -> Path:
    """24 kHz float audio to a telephone-quality 8 kHz mono MP3 in ``workdir``."""
    raw = workdir / f"{name}.raw.wav"
    ulaw = workdir / f"{name}.ulaw.wav"
    mp3 = workdir / f"{name}.mp3"
    peak = float(np.max(np.abs(audio))) or 1.0
    sf.write(raw, audio / peak * 0.8, SAMPLE_RATE)
    noise = rng.uniform(0.0025, 0.0045)
    # The telephone band (300 to 3400 Hz) as one FIR equalizer over voice and line noise together.
    chain = (
        "[0:a]acompressor=threshold=-20dB:ratio=3:attack=5:release=80[v];"
        "[v][1:a]amix=inputs=2:duration=first:normalize=0,"
        f"firequalizer=gain='{TELEPHONE_BAND}',alimiter=limit=0.95,aresample=8000"
    )
    _ffmpeg(
        "-i", str(raw),
        "-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude={noise}:sample_rate={SAMPLE_RATE}:seed={rng.randint(1, 10**6)}",
        "-filter_complex", chain,
        "-ac", "1", "-ar", "8000", "-c:a", "pcm_mulaw", str(ulaw),
    )
    _ffmpeg("-i", str(ulaw), "-ac", "1", "-ar", "8000", "-c:a", "libmp3lame", "-b:a", MP3_BITRATE, str(mp3))
    raw.unlink(missing_ok=True)
    ulaw.unlink(missing_ok=True)
    return mp3


def notice_file(audio: np.ndarray, path: Path) -> tuple[Path, str]:
    """Write the notice as the receiver would serve it (8 kHz mu-law WAV) and return its SHA-256."""
    tmp = path.with_suffix(".24k.wav")
    sf.write(tmp, audio, SAMPLE_RATE)
    _ffmpeg("-i", str(tmp), "-ac", "1", "-ar", "8000", "-c:a", "pcm_mulaw", str(path))
    tmp.unlink(missing_ok=True)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def probe_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return round(float(out.stdout.strip()), 3)
