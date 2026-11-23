"""Generate a run of Smart Trunk shaped synthetic calls: script, voices, phone audio, check, vCon.

One process, in three phases, so memory stays bounded on a shared machine:

1. For each call: Claude writes the script, Kokoro speaks it, ffmpeg makes the phone audio.
2. Kokoro is released, then parakeet-mlx transcribes every call in one child process. A call
   whose audio is garbled is spoken again with other voices and checked again, up to
   ``AUDIO_ATTEMPTS`` times. The two models are never loaded at the same time.
3. Each call that passed becomes a vCon, validated against syntax 0.5.0.

Every stage is timed, memory is sampled, and API tokens are counted. Scripts are cached in
``scripts/``, so a rerun after a failure does not pay for them again.
"""
from __future__ import annotations

import ctypes
import json
import os
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np
from vcon import compute_content_hash

from .build import build_call_vcon, screen_result, validate
from .check import MAX_WER, Check, Transcriber
from .pack import FEMALE_VOICES, MALE_VOICES, ROBOCALL_VOICE, SYSTEM_VOICE, CallPlan, build_plan, load_pack
from .review import write_index
from .telephone import MP3_BITRATE, Layout, Tones, lay_out, notice_file, probe_seconds, to_phone_mp3
from .voice import Voices
from .writer import Script, ScriptRejected, ScriptWriter

AUDIO_ATTEMPTS = 3
MB = 1024 * 1024


def peak_footprint_bytes() -> int | None:
    """This process's lifetime peak memory footprint on macOS (``proc_pid_rusage``, flavor 4).

    The footprint counts MLX's GPU buffers, which the resident set size mostly does not.
    """
    if sys.platform != "darwin":
        return None
    u = ctypes.c_uint64
    names = (
        "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size "
        "phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time "
        "child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread "
        "diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background "
        "cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated cpu_time_qos_user_interactive "
        "billed_system_time serviced_system_time logical_writes lifetime_max_phys_footprint instructions "
        "cycles billed_energy serviced_energy interval_max_phys_footprint runnable_time flags user_ptime "
        "system_ptime pinstructions pcycles energy_nj penergy_nj"
    ).split()

    class RUsageInfoV4(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), *[(n, u) for n in names], ("reserved", u * 14)]

    info = RUsageInfoV4()
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
            return None
    except OSError:
        return None
    return int(info.lifetime_max_phys_footprint)


class MemorySampler(threading.Thread):
    """Peak resident memory of this process and its children, and the lowest free system memory."""

    def __init__(self, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        import psutil

        self.psutil = psutil
        self.me = psutil.Process()
        self.interval = interval
        self.peak_self = 0
        self.peak_children = 0
        self.min_available = psutil.virtual_memory().available
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                kids = 0
                for child in self.me.children(recursive=True):
                    try:
                        kids += child.memory_info().rss
                    except self.psutil.Error:
                        pass
                self.peak_self = max(self.peak_self, self.me.memory_info().rss)
                self.peak_children = max(self.peak_children, kids)
                self.min_available = min(self.min_available, self.psutil.virtual_memory().available)
            except self.psutil.Error:
                pass
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2)


@dataclass
class RunConfig:
    industry: str
    out: Path
    count: int | None
    seed: int
    media_base_url: str
    parakeet: str
    run_label: str


@dataclass
class Call:
    """One call as it moves through the phases."""

    plan: CallPlan
    script: Script
    script_cached: bool
    script_seconds: float
    rng: random.Random
    mp3: Path | None = None
    layout: Layout | None = None
    expected: str = ""
    duration: float = 0.0
    voices: dict[str, str] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    check: Check | None = None


def _voices_for(plan: CallPlan, attempt: int) -> dict[str, tuple[str, float]]:
    """Speaker voices for an attempt. A retry moves both speakers to other voices at normal speed."""
    if attempt == 1:
        return {"caller": (plan.caller.voice, plan.caller.speed), "business": (plan.agent.voice, plan.agent.speed)}

    def shift(voice: str, gender: str, avoid: str) -> str:
        if voice == ROBOCALL_VOICE:
            return voice
        pool = FEMALE_VOICES if gender == "f" else MALE_VOICES
        i = pool.index(voice) if voice in pool else 0
        for k in range(1, len(pool) + 1):
            candidate = pool[(i + attempt * 3 + k) % len(pool)]
            if candidate != avoid:
                return candidate
        return pool[0]

    business = shift(plan.agent.voice, plan.agent.gender, avoid="")
    caller = shift(plan.caller.voice, plan.caller.gender, avoid=business)
    return {"caller": (caller, 1.0), "business": (business, 1.0)}


def _script_for(writer: ScriptWriter, plan: CallPlan, scripts_dir: Path) -> tuple[Script, bool]:
    path = scripts_dir / f"{plan.call_id}.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        same = ("call_id", "scenario", "business", "caller", "agent", "owned_vehicle", "wanted_model", "start")
        current = json.loads(json.dumps(plan.summary()))
        if all(saved.get("plan", {}).get(k) == current.get(k) for k in same):
            return Script(**saved["script"]), True
    script = writer.write(plan)
    path.write_text(json.dumps({"plan": plan.summary(), "script": script.to_dict()}, indent=2))
    return script, False


def _accumulate_usage(path: Path, this_run: dict[str, Any]) -> dict[str, Any]:
    """Add this run's API use to the folder's running total, so a rerun that reuses cached
    scripts still reports what they cost."""
    record = json.loads(path.read_text()) if path.is_file() else {"runs": []}
    record["runs"].append({"at": datetime.now(UTC).isoformat(timespec="seconds"), **this_run})
    keys = ("requests", "input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens", "refusals", "fallbacks")
    total: dict[str, Any] = {k: sum(r.get(k, 0) for r in record["runs"]) for k in keys}
    total["estimated_cost_usd"] = round(sum(r.get("estimated_cost_usd", 0) for r in record["runs"]), 4)
    total.update({"model": this_run["model"], "effort": this_run["effort"], "runs": len(record["runs"]),
                  "this_run_requests": this_run["requests"], "price_per_mtok_usd": this_run["price_per_mtok_usd"]})
    record["total"] = total
    path.write_text(json.dumps(record, indent=2))
    return total


class Renderer:
    def __init__(self, voices: Voices, tones: Tones, work: Path, notice: np.ndarray, notice_text: str,
                 announce: np.ndarray | None, announce_text: str) -> None:
        self.voices, self.tones, self.work = voices, tones, work
        self.notice, self.notice_text = notice, notice_text
        self.announce, self.announce_text = announce, announce_text

    def render(self, call: Call, attempt: int) -> None:
        p = call.plan
        chosen = _voices_for(p, attempt)
        tts_before, spoken_before = self.voices.busy_seconds, self.voices.spoken_seconds
        spoken = [(t["speaker"], self.voices.say(t["text"], *chosen[t["speaker"]])) for t in call.script.turns]
        tts_seconds = self.voices.busy_seconds - tts_before
        t_mix = time.monotonic()
        robocall = p.kind == "robocall"
        audio, call.layout = lay_out(
            kind=p.kind, notice=self.notice, turns=spoken, tones=self.tones, rng=call.rng,
            system_audio=self.announce if robocall else None,
        )
        call.mp3 = to_phone_mp3(audio, self.work, p.call_id, call.rng)
        call.duration = probe_seconds(call.mp3)
        call.expected = " ".join(
            [self.notice_text, *(t["text"] for t in call.script.turns)]
            + ([self.announce_text] if robocall and self.announce_text else [])
        )
        call.voices = {k: v[0] for k, v in chosen.items()}
        call.attempts.append({
            "attempt": attempt, "voices": dict(call.voices), "tts_seconds": round(tts_seconds, 2),
            "spoken_seconds": round(self.voices.spoken_seconds - spoken_before, 2),
            "mix_seconds": round(time.monotonic() - t_mix, 2),
        })


def run(cfg: RunConfig) -> dict[str, Any]:
    t_run = time.monotonic()
    pack = load_pack(cfg.industry)
    plan = build_plan(pack, count=cfg.count, seed=cfg.seed)
    out = cfg.out
    dirs = {name: out / name for name in ("audio", "vcons", "scripts", "transcripts", "rejected")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    work = out / ".work"
    work.mkdir(exist_ok=True)

    sampler = MemorySampler()
    sampler.start()
    writer = ScriptWriter(pack)
    voices = Voices()
    transcriber = Transcriber(cfg.parakeet)

    notice_cfg = pack["notice"]
    notice_audio = voices.say(notice_cfg["text"], SYSTEM_VOICE)
    served, notice_sha = notice_file(notice_audio, dirs["audio"] / f"{notice_cfg['file']}.wav")
    notice = {"file": served.name, "sha256": notice_sha, "discloses": notice_cfg["discloses"]}
    announce_text = pack.get("screen", {}).get("announce", "")
    announce_audio = voices.say(announce_text, SYSTEM_VOICE) if announce_text else None
    renderer = Renderer(voices, Tones(work), work, notice_audio, notice_cfg["text"], announce_audio, announce_text)

    # Phase 1: scripts and audio.
    calls: list[Call] = []
    skipped: list[dict[str, Any]] = []
    t_phase1 = time.monotonic()
    for p in plan:
        t0 = time.monotonic()
        try:
            script, cached = _script_for(writer, p, dirs["scripts"])
        except ScriptRejected as rejected:
            print(f"[{p.index:02d}] {p.call_id}: no usable script: {rejected.reasons}", flush=True)
            skipped.append({"call_id": p.call_id, "reasons": rejected.reasons})
            continue
        if script.rejected and not cached:
            print(f"[{p.index:02d}] {p.call_id}: drafts thrown away first: {script.rejected}", flush=True)
        call = Call(p, script, cached, round(time.monotonic() - t0, 2), random.Random(cfg.seed * 1000 + p.index))
        renderer.render(call, 1)
        calls.append(call)
        print(f"[{p.index:02d}] {p.call_id}: script {call.script_seconds:.0f} s, {call.duration:.0f} s of audio", flush=True)
    phase1_seconds = time.monotonic() - t_phase1

    # Phase 2: transcript check, with re-rendering of garbled calls.
    t_phase2 = time.monotonic()
    pending = list(calls)
    for attempt in range(1, AUDIO_ATTEMPTS + 1):
        if not pending:
            break
        voices.release()
        checks = transcriber.check_many([(c.mp3, c.expected, c.duration) for c in pending])
        retry = []
        for c in pending:
            c.check = checks[c.mp3]
            c.attempts[-1].update({"wer": c.check.wer, "passed": c.check.passed, "stt_seconds": round(c.check.seconds, 2)})
            print(f"[{c.plan.index:02d}] {c.plan.call_id}: WER {c.check.wer:.1%} ({'pass' if c.check.passed else 'FAIL'}, attempt {attempt})", flush=True)
            if not c.check.passed and attempt < AUDIO_ATTEMPTS:
                renderer.render(c, attempt + 1)
                retry.append(c)
        pending = retry
    voices.release()
    phase2_seconds = time.monotonic() - t_phase2

    # Phase 3: vCons for the calls that passed.
    records: list[dict[str, Any]] = []
    for c in calls:
        p = c.plan
        passed = bool(c.check and c.check.passed)
        final = (dirs["audio"] if passed else dirs["rejected"]) / f"{p.call_id}.mp3"
        shutil.move(str(c.mp3), final)
        data = final.read_bytes()
        (dirs["transcripts"] / f"{p.call_id}.json").write_text(json.dumps(
            {"text": c.check.text, "sentences": c.check.sentences, "wer": c.check.wer, "expected": c.expected}, indent=2))
        record: dict[str, Any] = {
            "index": p.index, "call_id": p.call_id, "scenario": p.scenario["id"],
            "scenario_label": p.scenario["label"], "business": p.business["name"], "line": p.line,
            "screened": p.screened, "start_local": p.start.strftime("%a %Y-%m-%d %H:%M %Z"),
            "speakers": {"caller": p.caller.name or "automated dialer", "business": f"{p.agent.name}, {p.business['name']}"},
            "voices": c.voices, "audio_file": final.name, "duration": c.duration, "mp3_bytes": len(data),
            "check": c.check.to_dict(), "transcript_text": c.check.text, "audio_attempts": len(c.attempts),
            "attempts": c.attempts, "script": c.script.to_dict(), "script_cached": c.script_cached,
            "script_seconds": c.script_seconds,
        }
        if passed:
            screen = screen_result(p, p.start, c.layout, c.rng)
            v = build_call_vcon(
                plan=p, pack=pack, layout=c.layout, filename=final.name,
                media_url=cfg.media_base_url.rstrip("/") + "/" + final.name, content_hash=compute_content_hash(data),
                duration=c.duration, transcript=c.check, notice=notice, screen=screen, run_label=cfg.run_label,
            )
            vcon_file = dirs["vcons"] / f"{p.call_id}.vcon.json"
            vcon_file.write_text(json.dumps(v.to_dict(), indent=2))
            record.update({"uuid": v.uuid, "vcon_file": vcon_file.name, "screen": screen, "validation_problems": validate(v)})
        else:
            record.update({"uuid": None, "vcon_file": None, "screen": None, "validation_problems": ["rejected: garbled audio"]})
        records.append(record)

    sampler.stop()
    shutil.rmtree(work, ignore_errors=True)
    if not any(dirs["rejected"].iterdir()):
        dirs["rejected"].rmdir()
    wall = time.monotonic() - t_run

    passed = [r for r in records if r["check"]["passed"]]
    first_pass = sum(r["audio_attempts"] == 1 for r in passed)
    total_audio = sum(r["duration"] for r in passed)
    usage = _accumulate_usage(out / "api_usage.json", writer.usage.to_dict())
    footprint = peak_footprint_bytes()
    memory = {
        "generator_peak_footprint_mb": round(footprint / MB) if footprint else None,
        "generator_peak_rss_mb": round(sampler.peak_self / MB),
        "mlx_peak_mb": round((Voices.mlx_peak_bytes() or 0) / MB),
        "parakeet_peak_footprint_mb": round(transcriber.peak_footprint_bytes / MB),
        "parakeet_peak_rss_mb": round(transcriber.peak_rss_bytes / MB),
        "lowest_free_system_memory_mb": round(sampler.min_available / MB),
    }
    n = max(len(records), 1)
    stats = {
        "calls": f"{len(passed)} delivered" + (f", {len(skipped)} with no usable script" if skipped else "")
        + (f", {len(records) - len(passed)} rejected as garbled" if len(records) > len(passed) else ""),
        "passed transcript check": f"{len(passed)} of {len(records)} ({first_pass} on the first voices; limit WER {MAX_WER:.0%})",
        "word error rate": f"mean {sum(r['check']['wer'] for r in passed) / max(len(passed), 1):.1%}, worst {max((r['check']['wer'] for r in passed), default=0):.1%}",
        "vCons valid (vcon-lib, core-05 schema, governance)": f"{sum(not r['validation_problems'] for r in passed)} of {len(passed)}",
        "audio": f"{total_audio / 60:.1f} min in all, {min((r['duration'] for r in passed), default=0):.0f} to {max((r['duration'] for r in passed), default=0):.0f} s per call, mean {total_audio / max(len(passed), 1):.0f} s",
        "MP3 size": f"{sum(r['mp3_bytes'] for r in passed) / 1e6:.2f} MB in all, mean {sum(r['mp3_bytes'] for r in passed) / max(len(passed), 1) / 1e3:.0f} KB (8 kHz mono, {MP3_BITRATE}bps)",
        "wall time": f"{wall / 60:.1f} min, {wall / n:.0f} s per call (scripts and audio {phase1_seconds / 60:.1f} min, checks {phase2_seconds / 60:.1f} min)",
        "script writing": f"{sum(r['script_seconds'] for r in records) / n:.0f} s per call",
        "Kokoro voices": f"{voices.realtime_factor}x realtime ({voices.spoken_seconds:.0f} s of speech in {voices.busy_seconds:.0f} s; model loads {voices.load_seconds:.0f} s)",
        "parakeet check": f"{(transcriber.audio_seconds / transcriber.busy_seconds) if transcriber.busy_seconds else 0:.1f}x realtime ({transcriber.audio_seconds:.0f} s of audio in {transcriber.busy_seconds:.0f} s, {transcriber.batches} batches, model load included)",
        "peak memory": (
            f"generator {memory['generator_peak_footprint_mb']} MB footprint ({memory['generator_peak_rss_mb']} MB RSS, MLX {memory['mlx_peak_mb']} MB); "
            f"parakeet {memory['parakeet_peak_footprint_mb']} MB footprint ({memory['parakeet_peak_rss_mb']} MB RSS); never both at once; "
            f"lowest free system memory {memory['lowest_free_system_memory_mb']} MB"
        ),
        "script tokens": (
            f"{usage['input_tokens']:,} input, {usage['output_tokens']:,} output (thinking included), "
            f"{usage['cache_write_tokens']:,} cache write, {usage['cache_read_tokens']:,} cache read; "
            f"{usage['requests']} requests over {usage['runs']} run(s) into this folder, {usage['this_run_requests']} in this one"
        ),
        "estimated API cost": f"${usage['estimated_cost_usd']:.2f} ({usage['model']}, effort {usage['effort']}, list prices)",
    }
    manifest = {
        "run_label": cfg.run_label, "industry": cfg.industry, "pack_version": pack["version"], "seed": cfg.seed,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "media_base_url": cfg.media_base_url, "notice": notice, "stats": stats, "memory": memory, "api_usage": usage,
        "tts": {"model": voices.repo, "spoken_seconds": round(voices.spoken_seconds, 1),
                "busy_seconds": round(voices.busy_seconds, 1), "load_seconds": round(voices.load_seconds, 1)},
        "stt": {"model": transcriber.model, "audio_seconds": round(transcriber.audio_seconds, 1),
                "busy_seconds": round(transcriber.busy_seconds, 1), "batches": transcriber.batches},
        "wall_seconds": round(wall, 1), "phase_seconds": {"scripts_and_audio": round(phase1_seconds, 1), "checks": round(phase2_seconds, 1)},
        "skipped": skipped,
        "calls": [{k: v for k, v in r.items() if k not in ("script", "transcript_text")} for r in records],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_index(
        out,
        records,
        {
            "title": f"Smart Trunk synthetic calls: {pack['label'].lower()}, {cfg.run_label}",
            "lede": (
                "Invented dealerships, people and numbers (555-0100 to 555-0199). Scripts by Claude, voices by "
                "Kokoro on this machine, phone audio by ffmpeg, transcripts by parakeet-mlx. Every vCon is marked "
                "synthetic. The recording notice plays at the start of each recording."
            ),
            "stats": stats,
        },
    )
    return manifest
