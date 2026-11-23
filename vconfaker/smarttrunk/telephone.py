"""Turn spoken turns into a two-channel phone call recording.

The recording starts at answer, as Smart Trunk's do: the ringback before it is not recorded,
only its length is kept for the Session timing. The recording opens with the receiver's notice,
then any phone menu (with the key the caller presses), then the conversation.

Channel 0 is the caller, channel 1 the business side: the people who answer, and everything the
caller hears from the system (the notice, the menu, hold music, the screening announcement).
Holds play hold music, not silence. An interrupting turn starts before the turn it cuts into has
finished; on separate channels the two simply overlap.

ffmpeg makes the tones and the hold music and does the telephone part on each channel: light
compression, a little pink line noise, a 300 to 3400 Hz band, G.711 mu-law at 8 kHz, then a
stereo MP3.
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
MP3_BITRATE = "48k"  # stereo, 8 kHz
TELEPHONE_BAND = "if(lt(f,300),-50,if(gt(f,3400),-50,0))"  # dB by frequency
CALLER, BUSINESS = 0, 1
CHANNEL_OF = {"caller": CALLER, "business": BUSINESS, "colleague": BUSINESS, "system": BUSINESS}
# DTMF: (row, column) frequencies in Hz for each key.
DTMF = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "4": (770, 1209), "5": (770, 1336),
    "6": (770, 1477), "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "0": (941, 1336),
}
# A slow, quiet pentatonic arpeggio: twelve notes a phrase, each with a soft decay.
HOLD_MUSIC = (
    "0.09*sin(2*PI*t*220*pow(2,(0+2*eq(mod(floor(t*2.5),6),1)+4*eq(mod(floor(t*2.5),6),2)"
    "+7*eq(mod(floor(t*2.5),6),3)+9*eq(mod(floor(t*2.5),6),4)+12*eq(mod(floor(t*2.5),6),5))/12))"
    "*exp(-2.2*mod(t*2.5,1))"
)


def _ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _tone(path: Path, expr: str, seconds: float) -> np.ndarray:
    if not path.exists():
        _ffmpeg("-f", "lavfi", "-i", f"aevalsrc='{expr}':s={SAMPLE_RATE}:d={seconds}", "-ac", "1", str(path))
    audio, _ = sf.read(path, dtype="float32")
    return audio


@dataclass
class Tones:
    """Ringback, the voicemail tone, a hang-up click, hold music and the DTMF keys."""

    workdir: Path
    ring: np.ndarray = field(init=False)
    beep: np.ndarray = field(init=False)
    click: np.ndarray = field(init=False)
    hold: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.ring = _tone(self.workdir / "ringback.wav", "0.08*sin(2*PI*440*t)+0.08*sin(2*PI*480*t)", 2.0)
        self.beep = _tone(self.workdir / "beep.wav", "0.25*sin(2*PI*1000*t)", 0.45)
        self.click = _tone(self.workdir / "click.wav", "0.3*exp(-60*t)*sin(2*PI*180*t)", 0.08)
        self.hold = _tone(self.workdir / "hold.wav", HOLD_MUSIC, 30.0)

    def key(self, digit: str) -> np.ndarray:
        low, high = DTMF[digit]
        return _tone(self.workdir / f"dtmf-{digit}.wav", f"0.2*sin(2*PI*{low}*t)+0.2*sin(2*PI*{high}*t)", 0.18)

    def hold_music(self, seconds: float, rng: random.Random) -> np.ndarray:
        n = int(seconds * SAMPLE_RATE)
        start = rng.randint(0, max(len(self.hold) - n, 0))
        return self.hold[start : start + n]


@dataclass
class Layout:
    """Where things sit, in seconds from the start of the recording (the answer)."""

    ring_seconds: float = 0.0  # before answer, not recorded
    notice_at: float = 0.0
    notice_seconds: float = 0.0
    key_at: float | None = None
    key: str | None = None
    ivr: tuple[float, float] | None = None
    announce: tuple[float, float] | None = None
    holds: list[tuple[float, float]] = field(default_factory=list)
    turns: list[tuple[str, float, float]] = field(default_factory=list)  # speaker, start, end
    ended_at: float = 0.0
    caller_dropped_at: float | None = None


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def _gain(audio: np.ndarray, db: float) -> np.ndarray:
    return audio * (10 ** (db / 20))


def lay_out(
    *,
    kind: str,
    notice: np.ndarray,
    turns: list[tuple[str, np.ndarray, bool]],
    tones: Tones,
    rng: random.Random,
    ivr: np.ndarray | None = None,
    key: str | None = None,
    announce: np.ndarray | None = None,
) -> tuple[np.ndarray, Layout]:
    """The call as a (samples, 2) array at 24 kHz, and its layout.

    ``turns`` are (speaker, audio, interrupts) in order; speaker is caller, business or colleague.
    """
    placed: list[tuple[int, float, np.ndarray]] = []  # channel, start seconds, audio
    layout = Layout(ring_seconds=round(rng.uniform(20.0, 26.0) if kind == "voicemail" else rng.uniform(5.0, 14.0), 2))

    def put(channel: int, audio: np.ndarray, at: float) -> float:
        placed.append((channel, at, audio))
        return at + len(audio) / SAMPLE_RATE

    clock = put(BUSINESS, notice, 0.0)
    layout.notice_seconds = round(clock, 3)
    clock += 0.3
    if ivr is not None and key:
        ivr_end = put(BUSINESS, ivr, clock)
        layout.ivr = (round(clock, 3), round(ivr_end, 3))
        clock = ivr_end + rng.uniform(0.6, 1.4)
        layout.key_at, layout.key = round(clock, 3), key
        clock = put(CALLER, tones.key(key), clock) + 0.3
        clock = put(BUSINESS, tones.ring, clock) + rng.uniform(0.2, 0.6)

    caller_db = rng.uniform(-3.0, 1.0)  # each caller's line is a little different
    last_end = clock
    for i, (speaker, audio, interrupts) in enumerate(turns):
        start = last_end + rng.uniform(0.25, 0.7)
        if i and interrupts and layout.turns:
            _, prev_start, prev_end = layout.turns[-1]
            start = max(prev_start + 0.5, prev_end - rng.uniform(0.35, 0.8))
        first_colleague = speaker == "colleague" and all(s != "colleague" for s, _, _ in layout.turns)
        on_hold = kind == "hangup" and i == len(turns) - 1 and i > 0
        if first_colleague or on_hold:
            seconds = rng.uniform(6.0, 9.0) if on_hold else rng.uniform(4.0, 7.0)
            hold_start = last_end + 0.4
            put(BUSINESS, tones.hold_music(seconds, rng), hold_start)
            layout.holds.append((round(hold_start, 3), round(hold_start + seconds, 3)))
            start = hold_start + seconds + rng.uniform(0.3, 0.7)
        audio = _gain(audio, caller_db) if speaker == "caller" else audio
        end = put(CHANNEL_OF[speaker], audio, start)
        layout.turns.append((speaker, round(start, 3), round(end, 3)))
        last_end = max(last_end, end)
        if kind == "voicemail" and i == 0:
            last_end = put(BUSINESS, tones.beep, last_end + 0.3) + 0.2

    if kind == "robocall" and announce is not None:
        start = last_end + 0.2
        last_end = put(BUSINESS, announce, start)
        layout.announce = (round(start, 3), round(last_end, 3))
    last_end += 0.3
    if kind in ("hangup", "voicemail", "robocall", "wrong_number"):
        layout.caller_dropped_at = round(last_end, 3)
        last_end = put(CALLER, tones.click, last_end)
    layout.ended_at = round(last_end + 0.5, 3)

    out = np.zeros((int(layout.ended_at * SAMPLE_RATE) + 1, 2), dtype=np.float32)
    for channel, at, audio in placed:
        i0 = int(at * SAMPLE_RATE)
        out[i0 : i0 + len(audio), channel] += audio[: len(out) - i0]
    return out, layout


def to_phone_mp3(audio: np.ndarray, workdir: Path, name: str, rng: random.Random) -> Path:
    """A (samples, 2) 24 kHz call to a telephone-quality 8 kHz stereo MP3 in ``workdir``."""
    raw = workdir / f"{name}.raw.wav"
    ulaw = workdir / f"{name}.ulaw.wav"
    mp3 = workdir / f"{name}.mp3"
    peak = float(np.max(np.abs(audio))) or 1.0
    sf.write(raw, audio / peak * 0.8, SAMPLE_RATE)
    noise = [rng.uniform(0.0025, 0.0045) for _ in range(2)]
    chain = (
        "[1:a][2:a]join=inputs=2:channel_layout=stereo[n];"
        "[0:a]acompressor=threshold=-20dB:ratio=3:attack=5:release=80:link=maximum[v];"
        "[v][n]amix=inputs=2:duration=first:normalize=0,"
        f"firequalizer=gain='{TELEPHONE_BAND}',alimiter=limit=0.95,aresample=8000"
    )
    noise_inputs = []
    for amplitude in noise:
        noise_inputs += ["-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude={amplitude}:sample_rate={SAMPLE_RATE}:seed={rng.randint(1, 10**6)}"]
    _ffmpeg("-i", str(raw), *noise_inputs, "-filter_complex", chain, "-ac", "2", "-ar", "8000", "-c:a", "pcm_mulaw", str(ulaw))
    _ffmpeg("-i", str(ulaw), "-ac", "2", "-ar", "8000", "-c:a", "libmp3lame", "-b:a", MP3_BITRATE, str(mp3))
    raw.unlink(missing_ok=True)
    ulaw.unlink(missing_ok=True)
    return mp3


def split_channels(mp3: Path, workdir: Path) -> tuple[Path, Path]:
    """The caller and business channels of a call as 16 kHz mono WAVs."""
    out = []
    for channel in (CALLER, BUSINESS):
        path = workdir / f"{mp3.stem}.ch{channel}.wav"
        _ffmpeg("-i", str(mp3), "-af", f"pan=mono|c0=c{channel}", "-ar", "16000", str(path))
        out.append(path)
    return out[0], out[1]


def speech_regions(layout: Layout, channel: int) -> list[tuple[float, float]]:
    """Where speech sits on a channel: the lines, plus the notice, menu and announcement on the
    business side. Hold music and key tones are not speech."""
    regions = [(start, end) for speaker, start, end in layout.turns if CHANNEL_OF[speaker] == channel]
    if channel == BUSINESS:
        regions.append((layout.notice_at, layout.notice_at + layout.notice_seconds))
        regions += [r for r in (layout.ivr, layout.announce) if r]
    return sorted(regions)


TimeMap = list[tuple[float, float, float]]  # (start in the cut file, end in the cut file, start in the call)
GAP, MARGIN = 0.6, 0.15


def speech_only(mp3: Path, layout: Layout, workdir: Path) -> list[tuple[Path, TimeMap]]:
    """Each channel cut down to its speech, joined with short gaps, as 16 kHz mono WAVs.

    The recognizer drops speech that follows a long silence, and on its own channel each speaker
    is silent while the other talks. This is the voice activity step a real pipeline runs before
    recognition; here the layout says where the speech is. The time map turns times in a cut
    file back into times in the call.
    """
    rate = 16000
    raw = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(mp3), "-f", "f32le", "-ar", str(rate), "-ac", "2", "-"],
        check=True, capture_output=True,
    ).stdout
    both = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)
    out = []
    for channel in (CALLER, BUSINESS):
        pieces: list[np.ndarray] = []
        time_map: TimeMap = []
        cursor = 0.0
        for start, end in speech_regions(layout, channel):
            a, b = max(start - MARGIN, 0.0), min(end + MARGIN, len(both) / rate)
            piece = both[int(a * rate) : int(b * rate), channel]
            if not len(piece):
                continue
            pieces += [piece, np.zeros(int(GAP * rate), dtype=np.float32)]
            time_map.append((round(cursor, 3), round(cursor + len(piece) / rate, 3), round(a, 3)))
            cursor += (len(piece) + int(GAP * rate)) / rate
        audio = np.concatenate(pieces) if pieces else np.zeros(rate // 2, dtype=np.float32)
        path = workdir / f"{mp3.stem}.speech{channel}.wav"
        sf.write(path, audio, rate)
        out.append((path, time_map))
    return out


def to_call_time(t: float, time_map: TimeMap) -> float:
    """A time in a cut file as a time in the call."""
    for cut_start, cut_end, call_start in time_map:
        if t <= cut_end + GAP / 2:
            return round(call_start + max(t - cut_start, 0.0), 3)
    if time_map:
        cut_start, _, call_start = time_map[-1]
        return round(call_start + t - cut_start, 3)
    return t


def notice_file(audio: np.ndarray, path: Path) -> tuple[Path, str]:
    """Write the notice as the receiver would serve it (8 kHz mu-law WAV) and return its SHA-256."""
    tmp = path.with_suffix(".24k.wav")
    sf.write(tmp, audio, SAMPLE_RATE)
    _ffmpeg("-i", str(tmp), "-ac", "1", "-ar", "8000", "-c:a", "pcm_mulaw", str(path))
    tmp.unlink(missing_ok=True)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path: Path) -> tuple[float, int]:
    """Duration in seconds and channel count."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=channels", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    import json

    info = json.loads(out.stdout)
    return round(float(info["format"]["duration"]), 3), int(info["streams"][0]["channels"])
