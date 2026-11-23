"""Transcribe each call back with parakeet-mlx and compare it with the script.

The recognizer hears the finished 8 kHz MP3, as a real pipeline would. The word error rate
against what was meant to be said (the notice, the turns, any screening announcement) decides
whether the audio is usable: above ``MAX_WER`` the call counts as garbled and is made again with
other voices. The recognizer's output, not the script, becomes the vCon's transcript.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
MAX_WER = 0.20

_DIGITS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_SAME = {"okay": "ok", "alright": "all right", "a.m.": "am", "p.m.": "pm", "%": " percent"}


def normalise(text: str) -> list[str]:
    text = text.lower()
    for a, b in _SAME.items():
        text = text.replace(a, b)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 32,000 -> 32000
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    words: list[str] = []
    for w in text.split():
        w = w.strip("'")
        if not w:
            continue
        if w.isdigit():
            words.extend(w)  # "555-0152" and "five five five, zero one five two" compare equal
        elif w in _DIGITS:
            words.append(_DIGITS[w])
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed, "wer": self.wer, "seconds": round(self.seconds, 2), "max_wer": MAX_WER,
            "stray_numbers": self.stray_numbers,
        }


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

    def _command(self, audio: list[Path], out: Path) -> list[str]:
        """The installed package run by ``RUNNER`` under its own interpreter, else the plain CLI.

        The CLI keeps MLX's buffer cache between files, so its footprint grows by about 1.5 GB
        per call; ``RUNNER`` loads the model once and clears the cache after every file, using
        the CLI's own JSON formatter.
        """
        interpreter = _interpreter_of(self.binary)
        if interpreter:
            runner = out / "_runner.py"
            runner.write_text(RUNNER)
            return [interpreter, "-I", str(runner), self.model, str(out), *map(str, audio)]
        return [self.binary, *map(str, audio), "--model", self.model, "--output-format", "json", "--output-dir", str(out)]

    def transcribe_many(self, audio: list[Path]) -> dict[Path, dict[str, Any]]:
        out = Path(tempfile.mkdtemp(prefix="parakeet-"))
        try:
            cmd = self._command(audio, out)
            timer = ["/usr/bin/time", "-l"] if Path("/usr/bin/time").exists() else []
            t0 = time.monotonic()
            done = None
            for offline in ("1", "0"):  # the model is normally cached: skip the hub unless it is not
                env = {**os.environ, "HF_HUB_OFFLINE": offline}
                done = subprocess.run([*timer, *cmd], capture_output=True, text=True, env=env)
                if done.returncode == 0:
                    break
            if done is None or done.returncode != 0:
                raise RuntimeError(f"parakeet-mlx failed: {(done.stderr if done else '')[-800:]}")
            self.busy_seconds += time.monotonic() - t0
            self.batches += 1
            self._read_time(done.stderr)
            results = {}
            for path in audio:
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

    def check_many(self, items: list[tuple[Path, str, float]]) -> dict[Path, Check]:
        """``items`` are (audio, expected text, seconds of audio). One Check per audio file."""
        t0 = time.monotonic()
        results = self.transcribe_many([p for p, _, _ in items])
        share = (time.monotonic() - t0) / max(len(items), 1)
        checks = {}
        for path, expected, seconds in items:
            self.audio_seconds += seconds
            checks[path] = to_check(results[path], expected, share)
        return checks


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


def stray_numbers(text: str) -> list[str]:
    """Phone-like numbers in what the recognizer heard that are outside 555-0100..0199."""
    from .hygiene import PHONE

    return [
        m.group(0).strip() for m in PHONE.finditer(text)
        if not (m.group(1) == "555" and 100 <= int(m.group(2)) <= 199)
    ]


def to_check(result: dict[str, Any], expected: str, seconds: float = 0.0) -> Check:
    """Pass when the word error rate is within ``MAX_WER`` and the transcript carries no number
    outside the fictional range (it goes into the vCon as is)."""
    text = (result.get("text") or "").strip()
    wer = word_error_rate(expected, text)
    stray = stray_numbers(text)
    sentences = [
        {k: s.get(k) for k in ("start", "end", "text", "confidence") if k in s}
        for s in result.get("sentences") or []
    ]
    for s in sentences:
        s["text"] = (s.get("text") or "").strip()
    return Check(wer <= MAX_WER and not stray, wer, text, sentences, seconds, stray)
