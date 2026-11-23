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
CLI_CHUNK = 5  # files per plain-CLI process, whose memory grows about 1.5 GB per file
# What huggingface_hub says when HF_HUB_OFFLINE=1 and the model is not in the local cache.
_CACHE_MISS = re.compile(
    r"LocalEntryNotFoundError|Cannot find the requested files in the (?:local|disk) cache|"
    r"outgoing traffic has been disabled",
    re.I,
)
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


def to_check(result: dict[str, Any], expected: str, seconds: float = 0.0) -> Check:
    """Pass when the word error rate is within ``MAX_WER`` and the transcript carries no number
    outside the fictional range (it goes into the vCon as is)."""
    text = (result.get("text") or "").strip()
    wer = word_error_rate(expected, text)
    stray = stray_numbers(text, expected)
    sentences = [
        {k: s.get(k) for k in ("start", "end", "text", "confidence") if k in s}
        for s in result.get("sentences") or []
    ]
    for s in sentences:
        s["text"] = (s.get("text") or "").strip()
    return Check(wer <= MAX_WER and not stray, wer, text, sentences, seconds, stray)
