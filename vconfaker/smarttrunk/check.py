"""Transcribe each call back with parakeet-mlx and compare it with the script, line by line.

The recognizer hears each channel of the finished 8 kHz recording on its own: the caller, and the
business side with the notice and any menu. A call passes when the word error rate over both
channels is within ``MAX_WER``, every line of three words or more was heard (at least
``MIN_TURN_COVERAGE`` of its words, in order), and no phone number outside the fictional range
came out of the recognizer. Otherwise it is voiced again with other voices. The recognizer's
output, not the script, becomes the vCon's transcript, with each segment's channel and party.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hygiene import DIGIT_WORDS as _DIGITS
from .hygiene import in_fictional_range, numbers_outside_range

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
MAX_WER = 0.20
MIN_TURN_COVERAGE = 0.5
MIN_TURN_WORDS = 3
CLI_CHUNK = 5  # files per plain-CLI process, whose memory grows about 1.5 GB per file
# What huggingface_hub says when HF_HUB_OFFLINE=1 and the model is not in the local cache.
_CACHE_MISS = re.compile(
    r"LocalEntryNotFoundError|Cannot find the requested files in the (?:local|disk) cache|"
    r"outgoing traffic has been disabled",
    re.I,
)
_SAME = {"okay": "ok", "alright": "all right", "a.m.": "am", "p.m.": "pm", "%": " percent"}


_VALUES = {
    **{w: int(d) for w, d in _DIGITS.items()},
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": "00", "thousand": "000"}


def _spoken_numbers(words: list[str]) -> list[str]:
    """Number words as digit strings: "nineteen eighty five" -> 19, 85; "ninth" -> 9; "thirty two
    thousand" -> 32, 000. A scale word counts only at the end of a number ("two hundred fifty" is
    2, 50), which is close enough once everything is compared digit by digit."""
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in _TENS:
            value = _TENS[w]
            if i + 1 < len(words) and words[i + 1] in _VALUES and _VALUES[words[i + 1]] < 10:
                value += _VALUES[words[i + 1]]
                i += 1
            out.append(str(value))
        elif w in _VALUES and w not in ("oh", "second") or (w in ("oh", "second") and out and out[-1].isdigit()):
            out.append(str(_VALUES[w]))
        elif w in _SCALES and out and out[-1].isdigit():
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if nxt not in _VALUES and nxt not in _TENS:
                out.append(_SCALES[w])
        else:
            out.append(w)
        i += 1
    return out


def normalise(text: str) -> list[str]:
    text = text.lower()
    for a, b in _SAME.items():
        text = text.replace(a, b)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 32,000 -> 32000
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    tokens = [re.sub(r"^(\d+)(?:st|nd|rd|th)$", r"\1", w.strip("'")) for w in text.split()]  # 9th -> 9
    words: list[str] = []
    for w in _spoken_numbers([t for t in tokens if t]):
        if w.isdigit():
            words.extend(w)  # compared digit by digit: "555-0152" and "five five five zero one five two"
        else:
            words.append(w)
    return words


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return round(prev[-1] / len(ref), 4)


@dataclass
class Check:
    passed: bool
    wer: float
    text: str
    sentences: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    stray_numbers: list[str] = field(default_factory=list)
    channel_wer: list[float] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed, "wer": self.wer, "channel_wer": self.channel_wer, "seconds": round(self.seconds, 2),
            "max_wer": MAX_WER, "uncovered_lines": self.uncovered, "stray_numbers": self.stray_numbers,
        }


def turn_coverage(items: list[str], heard: str) -> list[float | None]:
    """For each line meant to be said, the share of its words the recognizer heard, in order.

    The lines are aligned against the transcript as one sequence, so a word heard for one line
    is not credited to another. Lines under ``MIN_TURN_WORDS`` words get None (not judged).
    """
    expected: list[str] = []
    owner: list[int] = []
    for k, item in enumerate(items):
        words = normalise(item)
        expected += words
        owner += [k] * len(words)
    matched = [0] * len(items)
    matcher = difflib.SequenceMatcher(None, expected, normalise(heard), autojunk=False)
    for a, _, size in matcher.get_matching_blocks():
        for j in range(a, a + size):
            matched[owner[j]] += 1
    out: list[float | None] = []
    for k, item in enumerate(items):
        n = len(normalise(item))
        out.append(None if n < MIN_TURN_WORDS else round(matched[k] / n, 3))
    return out


def check_call(
    heard: list[dict[str, Any]], expected: list[list[str]], seconds: float = 0.0, time_maps: list[Any] | None = None
) -> Check:
    """One call from the recognizer's output per channel and the lines meant for each channel.

    With ``time_maps`` (from ``telephone.speech_only``) segment times are moved back to call time.
    """
    from .telephone import to_call_time

    edits = words = 0.0
    channel_wer: list[float] = []
    uncovered: list[str] = []
    stray: list[str] = []
    segments: list[dict[str, Any]] = []
    for channel, (result, items) in enumerate(zip(heard, expected, strict=True)):
        text = (result.get("text") or "").strip()
        reference = " ".join(items)
        wer = word_error_rate(reference, text)
        n = len(normalise(reference))
        edits += wer * n
        words += n
        channel_wer.append(wer)
        for item, cover in zip(items, turn_coverage(items, text), strict=True):
            if cover is not None and cover < MIN_TURN_COVERAGE:
                uncovered.append(f"channel {channel}: {item[:70]}")
        stray += stray_numbers(text, reference)
        for sentence in result.get("sentences") or []:
            seg = {k: sentence.get(k) for k in ("start", "end", "text", "confidence") if k in sentence}
            seg["text"] = (seg.get("text") or "").strip()
            if time_maps:
                for key in ("start", "end"):
                    if seg.get(key) is not None:
                        seg[key] = to_call_time(seg[key], time_maps[channel])
            seg.update(channel=channel, party=channel)
            segments.append(seg)
    segments.sort(key=lambda seg: (seg.get("start") or 0.0, seg["channel"]))
    total_wer = round(edits / words, 4) if words else 0.0
    passed = total_wer <= MAX_WER and not uncovered and not stray
    text = " ".join(seg["text"] for seg in segments if seg["text"])
    return Check(passed, total_wer, text, segments, seconds, stray, channel_wer, uncovered)


class Transcriber:
    """Runs the parakeet-mlx command line over a batch of calls in one child process.

    One process per batch loads the 0.6B model once (its weights are about 2.4 GB, and on a
    machine short of memory they are read from disk again on every load). The run calls it after
    Kokoro is released, so the two models are never in memory together. On macOS the child runs
    under ``/usr/bin/time -l``, which reports its peak memory footprint: MLX memory is mostly
    outside the resident set size, so the footprint is the honest figure.
    """

    def __init__(self, binary: str, model: str = PARAKEET_MODEL) -> None:
        self.binary = binary
        self.model = model
        self.busy_seconds = 0.0
        self.audio_seconds = 0.0
        self.batches = 0
        self.peak_footprint_bytes = 0
        self.peak_rss_bytes = 0

    def _commands(self, audio: list[Path], out: Path) -> list[tuple[list[str], list[Path]]]:
        """The processes to run: (command, the files it transcribes).

        Normally one: the installed package run by ``RUNNER`` under its own interpreter, which
        loads the model once and clears MLX's buffer cache after every file. Without it, the plain
        CLI keeps that cache between files and grows by about 1.5 GB per file, so it runs at most
        ``CLI_CHUNK`` files per process, and says so.
        """
        interpreter = _interpreter_of(self.binary)
        if interpreter:
            runner = out / "_runner.py"
            runner.write_text(RUNNER)
            return [([interpreter, "-I", str(runner), self.model, str(out), *map(str, audio)], audio)]
        print(
            f"note: parakeet_mlx is not importable from {self.binary}'s interpreter; running the plain CLI "
            f"{CLI_CHUNK} files at a time, one model load each",
            file=sys.stderr,
        )
        chunks = [audio[i : i + CLI_CHUNK] for i in range(0, len(audio), CLI_CHUNK)]
        return [
            ([self.binary, *map(str, chunk), "--model", self.model, "--output-format", "json", "--output-dir", str(out)], chunk)
            for chunk in chunks
        ]

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        """Run with the model hub offline; go online only when the model is not in the cache."""
        timer = ["/usr/bin/time", "-l"] if Path("/usr/bin/time").exists() else []
        done = subprocess.run([*timer, *cmd], capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
        if done.returncode != 0 and _CACHE_MISS.search(done.stderr or ""):
            done = subprocess.run([*timer, *cmd], capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "0"})
        if done.returncode != 0:
            raise RuntimeError(f"parakeet-mlx failed: {(done.stderr or '')[-800:]}")
        return done

    def transcribe_many(self, audio: list[Path]) -> dict[Path, dict[str, Any]]:
        out = Path(tempfile.mkdtemp(prefix="parakeet-"))
        try:
            results = {}
            for cmd, files in self._commands(audio, out):
                t0 = time.monotonic()
                done = self._run(cmd)
                self.busy_seconds += time.monotonic() - t0
                self.batches += 1
                self._read_time(done.stderr)
                for path in files:
                    found = out / f"{path.stem}.json"
                    if not found.is_file():
                        raise RuntimeError(f"parakeet-mlx wrote no JSON for {path.name}")
                    results[path] = json.loads(found.read_text())
            return results
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def _read_time(self, stderr: str) -> None:
        for line in stderr.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                if line.strip().endswith("peak memory footprint"):
                    self.peak_footprint_bytes = max(self.peak_footprint_bytes, int(parts[0]))
                elif line.strip().endswith("maximum resident set size"):
                    self.peak_rss_bytes = max(self.peak_rss_bytes, int(parts[0]))


RUNNER = """\
import sys
from pathlib import Path

import mlx.core as mx
from parakeet_mlx import from_pretrained
from parakeet_mlx.cli import drop_control_tokens, to_json

model_name, out = sys.argv[1], Path(sys.argv[2])
mx.set_cache_limit(256 * 1024 * 1024)
model = from_pretrained(model_name, dtype=mx.bfloat16)
for name in sys.argv[3:]:
    path = Path(name)
    result = model.transcribe(path, dtype=mx.bfloat16, chunk_duration=120.0, overlap_duration=15.0)
    (out / f"{path.stem}.json").write_text(to_json(drop_control_tokens(result)))
    mx.clear_cache()
"""


def _interpreter_of(binary: str) -> str | None:
    """The Python behind a console script (its ``#!`` line), when it has parakeet_mlx installed."""
    try:
        first = Path(shutil.which(binary) or binary).read_text(errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return None
    if not first.startswith("#!"):
        return None
    interpreter = first[2:].strip().split()[0]
    probe = subprocess.run([interpreter, "-I", "-c", "import parakeet_mlx.cli"], capture_output=True)
    return interpreter if probe.returncode == 0 else None


_BARE_RUN = re.compile(r"(?<![\d-])(?:\d{7}|\d{10}|1\d{10})(?![\d-])")


def stray_numbers(text: str, expected: str = "") -> list[str]:
    """Numbers in what the recognizer heard that could be a phone number outside 555-0100..0199.

    Written or spelled-out phone numbers are checked as in the script hygiene. The recognizer also
    runs a spoken number together ("5550200"), so a bare run of 7, 10 or 11 digits counts too,
    unless it is in the fictional range or the script itself said exactly that (a stock number).
    """
    stray = numbers_outside_range(text)
    said = set(_BARE_RUN.findall(expected))
    for run in _BARE_RUN.findall(text):
        local = run[-7:]
        if not in_fictional_range(local[:3], local[3:]) and run not in said:
            stray.append(run)
    return stray
