"""Generate a run of Smart Trunk shaped synthetic calls: script, voices, phone audio, check, vCon.

One process, in three phases, so memory stays bounded on a shared machine:

1. For each call: Claude writes the script, Kokoro speaks it, ffmpeg makes the two-channel
   phone audio.
2. Kokoro is released, then parakeet-mlx transcribes each channel of every call in one child
   process. A call that fails the check (word error rate, a line not heard, or a stray number)
   is voiced again with other voices and checked again, up to ``AUDIO_ATTEMPTS`` times. The two
   models are never loaded at the same time.
3. Each call that passed becomes a vCon, validated against syntax 0.5.0.

Before each heavy step the run pauses while free memory is under the floor. Every stage is
timed, memory is sampled, and API tokens are counted against an optional spending cap. Scripts
are cached in ``scripts/``, so a rerun after a failure does not pay for them again.
"""
from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from vcon import compute_content_hash

from .build import build_call_vcon, screen_result, validate
from .check import MAX_WER, Check, Transcriber, check_call
from .memory import MB, MemoryGuard, MemorySampler, peak_footprint_bytes
from .pack import FEMALE_VOICES, MALE_VOICES, ROBOCALL_VOICE, SYSTEM_VOICE, CallPlan, build_plan, load_pack
from .review import write_corpus_index, write_index
from .telephone import MP3_BITRATE, Layout, Tones, lay_out, notice_file, probe, speech_only, to_phone_mp3
from .voice import Voices
from .writer import BudgetExceeded, Script, ScriptRejected, ScriptWriter

AUDIO_ATTEMPTS = 3
SHORT_SECONDS = (15, 45)


@dataclass
class RunConfig:
    industry: str
    out: Path
    count: int
    seed: int
    media_base_url: str
    parakeet: str
    run_label: str
    usage_file: Path | None = None
    max_cost_usd: float | None = None
    min_free_mb: int = 2048
    corpus_root: Path | None = None


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
    expected: list[list[str]] = field(default_factory=list)  # lines meant for each channel
    duration: float = 0.0
    channels: int = 2
    voices: dict[str, str] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    check: Check | None = None


def _voices_for(plan: CallPlan, attempt: int) -> dict[str, tuple[str, float]]:
    """Voices for an attempt. A retry moves every speaker to other voices at normal speed."""
    planned = {"caller": plan.caller, "business": plan.agent}
    if plan.colleague:
        planned["colleague"] = plan.colleague
    if attempt == 1:
        return {k: (s.voice, s.speed) for k, s in planned.items()}
    chosen: dict[str, tuple[str, float]] = {}
    for role, speaker in planned.items():
        if speaker.voice == ROBOCALL_VOICE:
            chosen[role] = (speaker.voice, 1.0)
            continue
        pool = FEMALE_VOICES if speaker.gender == "f" else MALE_VOICES
        i = pool.index(speaker.voice) if speaker.voice in pool else 0
        taken = {v for v, _ in chosen.values()}
        voice = next(
            (pool[(i + attempt * 3 + k) % len(pool)] for k in range(1, len(pool) + 1) if pool[(i + attempt * 3 + k) % len(pool)] not in taken),
            pool[0],
        )
        chosen[role] = (voice, 1.0)
    return chosen


CACHE_KEYS = ("call_id", "scenario", "business", "caller", "agent", "caller_context", "wanted", "start", "opening",
              "length", "ivr_key", "colleague", "interruptions", "people")


def _script_for(writer: ScriptWriter, plan: CallPlan, scripts_dir: Path) -> tuple[Script, bool]:
    path = scripts_dir / f"{plan.call_id}.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        current = json.loads(json.dumps(plan.summary()))
        if all(saved.get("plan", {}).get(k) == current.get(k) for k in CACHE_KEYS):
            return Script(**saved["script"]), True
    script = writer.write(plan)
    path.write_text(json.dumps({"plan": plan.summary(), "script": script.to_dict()}, indent=2))
    return script, False


class Renderer:
    def __init__(self, voices: Voices, tones: Tones, work: Path, guard: MemoryGuard, notice: tuple[np.ndarray, str],
                 announce: tuple[np.ndarray | None, str]) -> None:
        self.voices, self.tones, self.work, self.guard = voices, tones, work, guard
        self.notice_audio, self.notice_text = notice
        self.announce_audio, self.announce_text = announce
        self._ivr: dict[str, np.ndarray] = {}

    def ivr(self, business: dict[str, Any]) -> np.ndarray:
        if business["id"] not in self._ivr:
            self._ivr[business["id"]] = self.voices.say(business["ivr"], SYSTEM_VOICE)
        return self._ivr[business["id"]]

    def render(self, call: Call, attempt: int) -> None:
        p = call.plan
        self.guard.wait(f"voicing {p.call_id}")
        chosen = _voices_for(p, attempt)
        tts_before, spoken_before = self.voices.busy_seconds, self.voices.spoken_seconds
        spoken = [(t["speaker"], self.voices.say(t["text"], *chosen[t["speaker"]]), bool(t.get("interrupts")))
                  for t in call.script.turns]
        tts_seconds = self.voices.busy_seconds - tts_before
        t_mix = time.monotonic()
        robocall = p.kind == "robocall" and self.announce_audio is not None
        audio, call.layout = lay_out(
            kind=p.kind, notice=self.notice_audio, turns=spoken, tones=self.tones, rng=call.rng,
            ivr=self.ivr(p.business) if p.ivr_key else None, key=p.ivr_key,
            announce=self.announce_audio if robocall else None,
        )
        call.mp3 = to_phone_mp3(audio, self.work, p.call_id, call.rng)
        call.duration, call.channels = probe(call.mp3)
        business_lines = [self.notice_text] + ([p.business["ivr"]] if p.ivr_key else [])
        business_lines += [t["text"] for t in call.script.turns if t["speaker"] != "caller"]
        business_lines += [self.announce_text] if robocall else []
        call.expected = [[t["text"] for t in call.script.turns if t["speaker"] == "caller"], business_lines]
        call.voices = {k: v[0] for k, v in chosen.items()}
        call.attempts.append({
            "attempt": attempt, "voices": dict(call.voices), "tts_seconds": round(tts_seconds, 2),
            "spoken_seconds": round(self.voices.spoken_seconds - spoken_before, 2),
            "mix_seconds": round(time.monotonic() - t_mix, 2),
        })


def _accumulate_usage(path: Path, this_run: dict[str, Any]) -> dict[str, Any]:
    """Add this run's API use to the running total in ``path`` (one file can serve a corpus)."""
    record = json.loads(path.read_text()) if path.is_file() else {"runs": []}
    record["runs"].append({"at": datetime.now(UTC).isoformat(timespec="seconds"), **this_run})
    keys = ("requests", "input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens", "refusals", "fallbacks")
    total: dict[str, Any] = {k: sum(r.get(k, 0) for r in record["runs"]) for k in keys}
    total["estimated_cost_usd"] = round(sum(r.get("estimated_cost_usd", 0) for r in record["runs"]), 4)
    total.update({"model": this_run["model"], "effort": this_run["effort"], "runs": len(record["runs"]),
                  "price_per_mtok_usd": this_run["price_per_mtok_usd"]})
    record["total"] = total
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return total


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
    usage_file = cfg.usage_file or out / "api_usage.json"

    sampler = MemorySampler()
    sampler.start()
    guard = MemoryGuard(cfg.min_free_mb)
    writer = ScriptWriter(pack, max_cost_usd=cfg.max_cost_usd, usage_file=usage_file)
    voices = Voices()
    transcriber = Transcriber(cfg.parakeet)

    notice_cfg = pack["notice"]
    guard.wait("loading voices")
    notice_audio = voices.say(notice_cfg["text"], SYSTEM_VOICE)
    served, notice_sha = notice_file(notice_audio, dirs["audio"] / f"{notice_cfg['file']}.wav")
    notice = {"file": served.name, "sha256": notice_sha, "discloses": notice_cfg["discloses"]}
    announce_text = pack.get("screen", {}).get("announce", "")
    announce_audio = voices.say(announce_text, SYSTEM_VOICE) if announce_text else None
    renderer = Renderer(voices, Tones(work), work, guard, (notice_audio, notice_cfg["text"]), (announce_audio, announce_text))

    # Phase 1: scripts and audio.
    calls: list[Call] = []
    skipped: list[dict[str, Any]] = []
    stopped = None
    t_phase1 = time.monotonic()
    for p in plan:
        t0 = time.monotonic()
        try:
            script, cached = _script_for(writer, p, dirs["scripts"])
        except ScriptRejected as rejected:
            print(f"[{p.index:02d}] {p.call_id}: no usable script: {rejected.reasons}", flush=True)
            skipped.append({"call_id": p.call_id, "reasons": rejected.reasons})
            continue
        except BudgetExceeded as over:
            stopped = str(over)
            print(f"stopping: {stopped}", flush=True)
            break
        if script.rejected and not cached:
            print(f"[{p.index:02d}] {p.call_id}: drafts thrown away first: {script.rejected}", flush=True)
        call = Call(p, script, cached, round(time.monotonic() - t0, 2), random.Random(f"{cfg.seed}-{p.call_id}"))
        renderer.render(call, 1)
        calls.append(call)
        print(f"[{p.index:02d}] {p.call_id}: {p.length}, {call.duration:.0f} s, script {call.script_seconds:.0f} s", flush=True)
    phase1_seconds = time.monotonic() - t_phase1

    # Phase 2: the transcript check, channel by channel, with re-voicing of calls that fail it.
    t_phase2 = time.monotonic()
    pending = list(calls)
    for attempt in range(1, AUDIO_ATTEMPTS + 1):
        if not pending:
            break
        voices.release()
        guard.wait("transcribing")
        halves = {c.plan.call_id: speech_only(c.mp3, c.layout, work) for c in pending}
        t0 = time.monotonic()
        heard = transcriber.transcribe_many([path for pair in halves.values() for path, _ in pair])
        share = (time.monotonic() - t0) / max(len(pending), 1)
        for pair in halves.values():
            for path, _ in pair:
                path.unlink(missing_ok=True)
        retry = []
        for c in pending:
            pair = halves[c.plan.call_id]
            transcriber.audio_seconds += c.duration * c.channels
            c.check = check_call([heard[path] for path, _ in pair], c.expected, share, [m for _, m in pair])
            c.attempts[-1].update({
                "wer": c.check.wer, "channel_wer": c.check.channel_wer, "passed": c.check.passed,
                "uncovered": c.check.uncovered, "stray_numbers": c.check.stray_numbers, "stt_seconds": round(share, 2),
            })
            why = "" if c.check.passed else f" ({'; '.join(c.check.uncovered + c.check.stray_numbers) or 'word error rate'})"
            print(f"[{c.plan.index:02d}] {c.plan.call_id}: WER {c.check.wer:.1%} {'pass' if c.check.passed else 'FAIL'}{why}, attempt {attempt}", flush=True)
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
            {"text": c.check.text, "segments": c.check.sentences, "check": c.check.to_dict(), "expected": c.expected}, indent=2))
        record: dict[str, Any] = {
            "index": p.index, "call_id": p.call_id, "scenario": p.scenario["id"], "scenario_label": p.scenario["label"],
            "business": p.business["name"], "line": p.line, "screened": p.screened, "features": p.features(),
            "start_local": p.start.strftime("%a %Y-%m-%d %H:%M %Z"),
            "speakers": {
                "caller": p.caller.name or "automated dialer", "business": f"{p.agent.name}, {p.business['name']}",
                "colleague": f"{p.colleague.name}, {p.colleague_role}" if p.colleague else None,
            },
            "voices": c.voices, "audio_file": final.name, "duration": c.duration, "channels": c.channels,
            "ring_seconds": c.layout.ring_seconds, "mp3_bytes": len(data), "check": c.check.to_dict(),
            "transcript_text": c.check.text, "audio_attempts": len(c.attempts), "attempts": c.attempts,
            "script": c.script.to_dict(), "script_cached": c.script_cached, "script_seconds": c.script_seconds,
        }
        if passed:
            screen = screen_result(p, p.start + timedelta(seconds=c.layout.ring_seconds), c.layout, c.rng)
            v = build_call_vcon(
                plan=p, pack=pack, layout=c.layout, filename=final.name,
                media_url=cfg.media_base_url.rstrip("/") + "/" + final.name, content_hash=compute_content_hash(data),
                duration=c.duration, transcript=c.check, notice=notice, screen=screen, run_label=cfg.run_label,
                channels=c.channels,
            )
            vcon_file = dirs["vcons"] / f"{p.call_id}.vcon.json"
            vcon_file.write_text(json.dumps(v.to_dict(), indent=2))
            record.update({"uuid": v.uuid, "vcon_file": vcon_file.name, "screen": screen, "validation_problems": validate(v)})
        else:
            record.update({"uuid": None, "vcon_file": None, "screen": None, "validation_problems": ["rejected: failed the transcript check"]})
        records.append(record)

    sampler.stop()
    shutil.rmtree(work, ignore_errors=True)
    if not any(dirs["rejected"].iterdir()):
        dirs["rejected"].rmdir()
    wall = time.monotonic() - t_run
    usage = _accumulate_usage(usage_file, writer.usage.to_dict())
    manifest = _manifest(cfg, pack, records, skipped, stopped, writer, voices, transcriber, sampler, guard, usage,
                         notice, wall, phase1_seconds, phase2_seconds)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_index(out, records, {
        "title": f"Smart Trunk synthetic calls: {pack['label'].lower()}, {cfg.run_label}",
        "lede": (
            "Invented businesses, people, places and numbers (555-0100 to 555-0199). Scripts by Claude, voices by "
            "Kokoro on this machine, two-channel phone audio by ffmpeg (left: caller, right: business), transcripts by "
            "parakeet-mlx per channel. The recording starts at answer with the notice. Every vCon is marked synthetic."
        ),
        "stats": manifest["stats"],
    })
    if cfg.corpus_root:
        write_corpus_index(cfg.corpus_root)
    return manifest


def _manifest(cfg, pack, records, skipped, stopped, writer, voices, transcriber, sampler, guard, usage, notice, wall,
              phase1_seconds, phase2_seconds) -> dict[str, Any]:
    passed = [r for r in records if r["check"]["passed"]]
    first_pass = sum(r["audio_attempts"] == 1 for r in passed)
    durations = [r["duration"] for r in passed]
    total_audio = sum(durations)
    footprint = peak_footprint_bytes()
    memory = {
        "generator_peak_footprint_mb": round(footprint / MB) if footprint else None,
        "generator_peak_rss_mb": round(sampler.peak_self / MB),
        "mlx_peak_mb": round((Voices.mlx_peak_bytes() or 0) / MB),
        "parakeet_peak_footprint_mb": round(transcriber.peak_footprint_bytes / MB),
        "parakeet_peak_rss_mb": round(transcriber.peak_rss_bytes / MB),
        "lowest_free_system_memory_mb": round(sampler.min_available / MB),
        "pauses": guard.pauses, "paused_seconds": round(guard.paused_seconds),
    }
    this_run = writer.usage.to_dict()
    n = max(len(records), 1)
    short = sum(SHORT_SECONDS[0] <= d <= SHORT_SECONDS[1] for d in durations)
    wers = [r["check"]["wer"] for r in passed]
    features = {k: sum(k in r["features"] for r in passed) for k in ("ivr", "transfer", "interruptions")}
    stats = {
        "calls": f"{len(passed)} delivered" + (f", {len(skipped)} with no usable script" if skipped else "")
        + (f", {len(records) - len(passed)} rejected by the transcript check" if len(records) > len(passed) else "")
        + (f"; stopped early: {stopped}" if stopped else ""),
        "passed transcript check": f"{len(passed)} of {len(records)} ({first_pass} on the first voices; WER limit {MAX_WER:.0%}, every line heard)",
        "word error rate": f"mean {sum(wers) / max(len(wers), 1):.1%}, worst {max(wers, default=0):.1%}",
        "vCons valid (vcon-lib, core-05 schema, governance)": f"{sum(not r['validation_problems'] for r in passed)} of {len(passed)}",
        "audio": f"{total_audio / 60:.1f} min in all; {min(durations, default=0):.0f} to {max(durations, default=0):.0f} s per call, mean {total_audio / max(len(passed), 1):.0f} s; {short} of {len(passed)} between {SHORT_SECONDS[0]} and {SHORT_SECONDS[1]} s",
        "variety": f"{features['ivr']} through a phone menu, {features['transfer']} transferred, {features['interruptions']} with interruptions, {len({r['features']['opening'] for r in passed})} kinds of opening",
        "MP3 size": f"{sum(r['mp3_bytes'] for r in passed) / 1e6:.2f} MB in all, mean {sum(r['mp3_bytes'] for r in passed) / max(len(passed), 1) / 1e3:.0f} KB (8 kHz stereo, {MP3_BITRATE}bps)",
        "wall time": f"{wall / 60:.1f} min, {wall / n:.0f} s per call (scripts and audio {phase1_seconds / 60:.1f} min, checks {phase2_seconds / 60:.1f} min)",
        "Kokoro voices": f"{voices.realtime_factor}x realtime ({voices.spoken_seconds:.0f} s of speech in {voices.busy_seconds:.0f} s; model loads {voices.load_seconds:.0f} s)",
        "parakeet check": f"{(transcriber.audio_seconds / transcriber.busy_seconds) if transcriber.busy_seconds else 0:.1f}x realtime over both channels ({transcriber.batches} batches, model load included)",
        "peak memory": (
            f"generator {memory['generator_peak_footprint_mb']} MB footprint ({memory['generator_peak_rss_mb']} MB RSS); "
            f"parakeet {memory['parakeet_peak_footprint_mb']} MB footprint; never both at once; lowest free system memory "
            f"{memory['lowest_free_system_memory_mb']} MB; {memory['pauses']} pauses for memory"
        ),
        "script tokens, this run": (
            f"{this_run['input_tokens']:,} input, {this_run['output_tokens']:,} output (thinking included), "
            f"{this_run['cache_write_tokens']:,} cache write, {this_run['cache_read_tokens']:,} cache read; {this_run['requests']} requests, "
            f"${this_run['estimated_cost_usd']:.2f}"
        ),
        "script cost so far": f"${usage['estimated_cost_usd']:.2f} over {usage['runs']} run(s) recorded in {cfg.usage_file or 'this folder'} ({usage['model']}, list prices)",
    }
    return {
        "run_label": cfg.run_label, "industry": cfg.industry, "pack_version": pack["version"], "seed": cfg.seed,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "media_base_url": cfg.media_base_url,
        "notice": notice, "stats": stats, "memory": memory, "api_usage_this_run": this_run, "api_usage_total": usage,
        "tts": {"model": voices.repo, "spoken_seconds": round(voices.spoken_seconds, 1), "busy_seconds": round(voices.busy_seconds, 1)},
        "stt": {"model": transcriber.model, "audio_seconds": round(transcriber.audio_seconds, 1),
                "busy_seconds": round(transcriber.busy_seconds, 1), "batches": transcriber.batches},
        "wall_seconds": round(wall, 1), "phase_seconds": {"scripts_and_audio": round(phase1_seconds, 1), "checks": round(phase2_seconds, 1)},
        "summary": {
            "calls": len(records), "passed": len(passed), "first_pass": first_pass, "mean_wer": round(sum(wers) / max(len(wers), 1), 4),
            "worst_wer": max(wers, default=0), "audio_seconds": round(total_audio, 1),
            "mp3_bytes": sum(r["mp3_bytes"] for r in passed), "short_calls": short, "skipped": len(skipped), "stopped": stopped,
        },
        "skipped": skipped,
        "calls": [{k: v for k, v in r.items() if k not in ("script", "transcript_text")} for r in records],
    }
