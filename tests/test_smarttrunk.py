"""Offline tests for the Smart Trunk call mode: no API, no voices, no recognizer.

Needs ``vcon>=1.0.3``, ``jsonschema``, numpy and soundfile; the stereo MP3 test also needs
ffmpeg. The voice and recognizer stages are exercised by a real run, not here.
"""
import json
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import fake_names
import jsonschema
import numpy as np
import pytest
from vcon import Vcon, compute_content_hash

from vconfaker.governance import governance_problems
from vconfaker.smarttrunk import check as check_module
from vconfaker.smarttrunk import hygiene
from vconfaker.smarttrunk.build import SCHEMA_FILE, build_call_vcon, screen_result, validate
from vconfaker.smarttrunk.check import check_call, stray_numbers, turn_coverage, word_error_rate
from vconfaker.smarttrunk.hygiene import check_script, real_business_matches
from vconfaker.smarttrunk.pack import INDUSTRIES, LENGTHS, OPENINGS, build_plan, fictional_number, load_pack
from vconfaker.smarttrunk.telephone import BUSINESS, CALLER, Layout, Tones, lay_out
from vconfaker.smarttrunk.writer import BudgetExceeded, ScriptWriter, allowed_names, brief, products

PACKS = {industry: load_pack(industry) for industry in INDUSTRIES}
PLANS = {industry: build_plan(PACKS[industry], count=50) for industry in INDUSTRIES}
PACK = PACKS["dealership"]
PLAN = PLANS["dealership"]
SR = 24_000


def _tone(seconds, freq=300.0, level=0.3):
    return (level * np.sin(np.arange(int(seconds * SR)) * 2 * np.pi * freq / SR)).astype("float32")


# --- Packs and the plan ---------------------------------------------------------------------------


@pytest.mark.parametrize("industry", INDUSTRIES)
def test_every_pack_uses_fictional_numbers_and_invented_names(industry):
    pack, plan = PACKS[industry], PLANS[industry]
    assert len(plan) == 50
    for p in plan:
        assert p.business_tel[5:8] == "555" and 100 <= int(p.business_tel[8:]) <= 149
        assert p.caller_tel[5:8] == "555" and 150 <= int(p.caller_tel[8:]) <= 199
    for business in pack["businesses"]:
        assert not real_business_matches(" ".join([business["name"], *business.get("places", [])]), industry)
    assert len({p.caller_tel for p in plan}) == 50


def test_business_numbers_are_unique_across_packs():
    numbers = [line["number"] for pack in PACKS.values() for b in pack["businesses"] for line in b["lines"].values()]
    assert len(numbers) == len(set(numbers))


def test_numbers_outside_the_fictional_range_are_refused():
    with pytest.raises(ValueError):
        fictional_number("614", "555-0200")
    with pytest.raises(ValueError):
        fictional_number("614", "555-0099")


@pytest.mark.parametrize("industry", INDUSTRIES)
def test_plan_varies_openings_lengths_menus_transfers_and_interruptions(industry):
    plan = PLANS[industry]
    assert len({p.opening for p in plan if p.kind != "voicemail"}) >= 5
    assert sum(p.length == "short" for p in plan) >= 0.35 * len(plan)
    assert any(p.ivr_key for p in plan) and any(p.transfer for p in plan) and any(p.interruptions for p in plan)
    assert any(p.kind == "wrong_number" for p in plan)


def test_scenarios_go_only_to_the_businesses_they_fit():
    toothaches = [p for p in PLANS["clinic"] if p.scenario["id"] == "toothache"]
    assert toothaches and {p.business["id"] for p in toothaches} <= {"wendlow", "kestlemoor"}


# --- Script hygiene --------------------------------------------------------------------------------


def _plan(industry="dealership", **want):
    want.setdefault("transfer", False)
    return next(p for p in PLANS[industry] if p.caller.name and all(getattr(p, k) == v for k, v in want.items()))


def _script(plan, extra_turn=None, speaker="caller"):
    turns = [
        {"speaker": "business", "text": f"{plan.business['name']}, this is {plan.agent.name}. How can I help?", "interrupts": False},
        {"speaker": "caller", "text": f"Hi, it's {plan.caller.name}. I need to set something up.", "interrupts": False},
        {"speaker": "business", "text": "Sure. Does next Thursday at ten work?", "interrupts": False},
        {"speaker": "caller", "text": f"Next Thursday works. My number is 555-{plan.caller_tel[-4:]}.", "interrupts": False},
        {"speaker": "business", "text": "You're all set. See you then.", "interrupts": False},
        {"speaker": "caller", "text": "Thanks, bye.", "interrupts": False},
    ]
    if extra_turn:
        turns.insert(4, {"speaker": speaker, "text": extra_turn, "interrupts": False})
    return {"summary": "s", "outcome": "booked", "turns": turns, "names_used": allowed_names(plan)}


def _check(plan, script, **kwargs):
    pack = PACKS[plan.industry]
    return check_script(
        script, allowed_names=allowed_names(plan), products=products(plan), turn_range=(4, 10), industry=plan.industry,
        reject_places=bool(pack.get("reject_places")), call_start=plan.start, colleague=plan.transfer, **kwargs
    )


@pytest.mark.parametrize(
    "line,reason",
    [
        ("Call me back at 555-0200.", "phone number"),
        ("Or try (614) 555-0200.", "phone number"),
        ("Email me at someone@example.com.", "email"),
        ("I saw it on your site, something dot com.", "web address"),
        ("CarMax offered me more.", "real business"),
        ("I looked it up on Kelley Blue Book.", "real business"),
        ("Let me check with {other} in finance.", "person"),
    ],
)
def test_hygiene_rejects_real_world_details(line, reason):
    plan = _plan()
    given = {w for n in allowed_names(plan) for w in n.split()}
    other = next(f"{f} {l}" for f in fake_names.male_names for l in fake_names.last_names if f not in given and l not in given)
    verdict = _check(plan, _script(plan, line.format(other=other)))
    assert not verdict.ok and any(reason in p for p in verdict.problems), verdict.problems


def test_hygiene_passes_a_clean_script_and_rejects_unlisted_names():
    plan = _plan()
    assert _check(plan, _script(plan)).ok
    script = _script(plan)
    script["names_used"].append("Somebody Else")
    assert not _check(plan, script).ok


@pytest.mark.parametrize(
    "line",
    ["Honestly carmax offered me more.", "I priced one at the tesla store.", "Walmart had wiper blades cheaper.",
     "I could get it at Costco.", "Mavis did my tires last time.", "My loan is with Santander.",
     "I saw floor mats at Sam's Club.", "Home Depot sells a charger like that."],
)
def test_real_business_names_are_rejected_in_any_case(line):
    plan = _plan()
    assert any("real business" in p for p in _check(plan, _script(plan, line)).problems)


@pytest.mark.parametrize("line", ["What's your budget for this?", "I'll chase that down for you.", "It's an enterprise fleet thing."])
def test_short_or_ambiguous_names_stay_ordinary_words_in_lower_case(line):
    plan = _plan()
    assert not any("real business" in p for p in _check(plan, _script(plan, line)).problems)


@pytest.mark.parametrize(
    "industry,line",
    [("clinic", "Can you send it to the CVS on the corner?"), ("clinic", "My plan is through Aetna."),
     ("municipal", "We moved here from Ohio."), ("municipal", "Is this the Duke Energy office?"),
     ("home_services", "Roto-Rooter quoted me less."), ("home_services", "I found you on Angie's List.")],
)
def test_each_new_industry_blocks_its_own_real_names(industry, line):
    plan = _plan(industry)
    assert any("real business" in p for p in _check(plan, _script(plan, line)).problems)


def test_a_caller_named_like_a_real_place_is_not_blocked(monkeypatch):
    plan = _plan("municipal")
    plan_named = SimpleNamespace(**{**plan.__dict__})
    plan_named.caller = SimpleNamespace(**{**plan.caller.__dict__, "name": "Dana Washington"})
    plan_named.transfer = plan.transfer
    script = _script(plan_named, "This is Dana Washington again.")
    verdict = check_script(script, allowed_names=allowed_names(plan_named), products=products(plan), turn_range=(4, 10),
                           industry="municipal")
    assert not any("real business" in p for p in verdict.problems), verdict.problems


def test_pack_names_can_extend_the_block_list():
    plan = _plan()
    assert any("real business" in p for p in _check(plan, _script(plan, "Zentrova Pharmacy filled it."), extra_real=["Zentrova"]).problems)


def test_a_one_word_organisation_not_given_rejects_the_script(monkeypatch):
    plan = _plan()
    monkeypatch.setattr(hygiene, "_entities", lambda texts: [("ORG", "Zentrova", False)])
    assert any("organisation" in p for p in _check(plan, _script(plan)).problems)
    monkeypatch.setattr(hygiene, "_entities", lambda texts: [("ORG", "Service", False)])
    assert _check(plan, _script(plan)).ok


def test_the_municipal_pack_rejects_places_it_was_not_given(monkeypatch):
    plan = _plan("municipal")
    monkeypatch.setattr(hygiene, "_entities", lambda texts: [("GPE", "Springfield", False)])
    assert any("place" in p for p in _check(plan, _script(plan)).problems)
    monkeypatch.setattr(hygiene, "_entities", lambda texts: [("FAC", plan.business["places"][1], False)])
    assert _check(plan, _script(plan)).ok
    plan = _plan("dealership")
    monkeypatch.setattr(hygiene, "_entities", lambda texts: [("GPE", "Springfield", False)])
    assert _check(plan, _script(plan)).ok  # elsewhere a place is only noted


def test_spelled_out_numbers_outside_the_range_are_caught():
    plan = _plan()
    verdict = _check(plan, _script(plan, "Call me on five five five zero two zero zero."))
    assert any("phone number" in p for p in verdict.problems), verdict.problems
    assert _check(plan, _script(plan, "Call me on five five five, zero one five two.")).ok
    assert stray_numbers("call me on five five five zero two zero zero")
    assert stray_numbers("call me on five five five zero one five two") == []


def test_a_seven_digit_stock_number_is_not_a_phone_number():
    plan = _plan()
    assert not any("phone number" in p for p in _check(plan, _script(plan, "The stock number is 1234567.")).problems)
    assert stray_numbers("the stock number is 1234567", expected="The stock number is 1234567.") == []
    assert stray_numbers("my number is 5550200", expected="My number is 555-0152.") == ["5550200"]


def test_an_area_coded_number_in_the_fictional_range_passes():
    plan = _plan()
    assert _check(plan, _script(plan, "Or try (614) 555-0142.")).ok


def test_nothing_is_booked_for_the_moment_of_the_call():
    plan = _plan()
    clock = plan.start.strftime("%-I:%M")
    day = plan.start.strftime("%A")
    verdict = _check(plan, _script(plan, f"Could I come in {day} at {clock}? Great, {day} at {clock} it is."))
    assert any("moment of the call" in p for p in verdict.problems), verdict.problems
    assert _check(plan, _script(plan, f"Could I come in next {day} at {clock}?")).ok
    later = (plan.start + timedelta(hours=3)).strftime("%-I:%M %p")
    assert _check(plan, _script(plan, f"Could I come in at {later}?")).ok


def test_a_colleague_speaks_only_on_a_transferred_call():
    plain = _plan(transfer=False)
    assert not _check(plain, _script(plain, "Hi, this is the manager.", speaker="colleague")).ok
    moved = next(p for p in PLAN if p.transfer)
    script = _script(moved, f"Hi {moved.caller.name.split()[0]}, it's {moved.colleague.name}.", speaker="colleague")
    assert _check(moved, script).ok, _check(moved, script).problems


# --- The brief -------------------------------------------------------------------------------------


def test_the_brief_carries_opening_length_transfer_interruptions_and_the_time_rule():
    moved = next(p for p in PLAN if p.transfer)
    text = brief(moved, PACK)
    assert OPENINGS[moved.opening] in text
    assert f"{LENGTHS[moved.length]['words'][0]} to {LENGTHS[moved.length]['words'][1]} words" in text
    assert f"transfers the caller to {moved.colleague.name}" in text
    assert "never for this moment" in text
    interrupted = next(p for p in PLAN if p.interruptions)
    assert "Interruptions:" in brief(interrupted, PACK)
    menu = next(p for p in PLAN if p.ivr_key)
    assert f"the caller pressed {menu.ivr_key}" in brief(menu, PACK)


# --- The writer ------------------------------------------------------------------------------------


def _fake_client(replies):
    def create(**kwargs):
        assert kwargs["model"] == "claude-sonnet-5-5" and kwargs["fallbacks"] == "default"
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        return SimpleNamespace(
            stop_reason="end_turn", stop_details=None,
            content=[SimpleNamespace(type="text", text=json.dumps(next(replies)))],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=0,
                                  cache_read_input_tokens=0, iterations=None),
        )

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def test_writer_retries_a_rejected_script_and_counts_tokens():
    plan = _plan(length="short", transfer=False)
    replies = iter([_script(plan, "Call me at 555-0300."), _script(plan)])
    writer = ScriptWriter(PACK, client=_fake_client(replies))
    script = writer.write(plan)
    assert script.attempts == 2 and script.rejected
    assert writer.usage.requests == 2
    assert writer.usage.cost_usd == pytest.approx((2000 * 2 + 1000 * 10) / 1e6)


def test_writer_stops_at_the_spending_cap(tmp_path):
    usage = tmp_path / "api_usage.json"
    usage.write_text(json.dumps({"runs": [{"estimated_cost_usd": 4.995}]}))
    plan = _plan(length="short", transfer=False)
    writer = ScriptWriter(PACK, client=_fake_client(iter([_script(plan)] * 5)), max_cost_usd=5.0, usage_file=usage)
    writer.write(plan)  # $4.995 is under the cap; this script takes it to $5.002
    with pytest.raises(BudgetExceeded):
        writer.write(plan)


# --- Audio layout ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tones(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    return Tones(tmp_path_factory.mktemp("tones"))


def _layout(tones, kind="conversation", turns=None, **kwargs):
    turns = turns or [("business", _tone(2.0, 400), False), ("caller", _tone(2.0, 600), False), ("business", _tone(2.0, 400), False)]
    return lay_out(kind=kind, notice=_tone(3.0, 900), turns=turns, tones=tones, rng=random.Random(1), **kwargs)


def test_the_recording_starts_at_answer_with_the_notice(tones):
    audio, layout = _layout(tones)
    assert layout.notice_at == 0.0 and layout.ring_seconds >= 5.0
    head = audio[: int(0.5 * SR), BUSINESS]
    assert np.allclose(head, _tone(3.0, 900)[: len(head)], atol=1e-6)  # the notice, not ringback
    assert layout.turns[0][1] >= layout.notice_seconds


def test_caller_and_business_are_on_separate_channels(tones):
    audio, layout = _layout(tones)
    (_, b0, b1), (_, c0, c1) = layout.turns[0], layout.turns[1]
    caller_part = audio[int(c0 * SR) : int(c1 * SR)]
    assert np.abs(caller_part[:, CALLER]).max() > 0.1 and np.abs(caller_part[:, BUSINESS]).max() < 1e-6
    business_part = audio[int(b0 * SR) : int(b1 * SR)]
    assert np.abs(business_part[:, BUSINESS]).max() > 0.1 and np.abs(business_part[:, CALLER]).max() < 1e-6


def test_an_interrupting_turn_overlaps_the_turn_before(tones):
    turns = [("business", _tone(3.0), False), ("caller", _tone(1.5), True)]
    _, layout = _layout(tones, turns=turns)
    assert layout.turns[1][1] < layout.turns[0][2]


def test_holds_play_music_not_silence(tones):
    turns = [("business", _tone(2.0), False), ("caller", _tone(1.0), False), ("caller", _tone(1.0), False)]
    audio, layout = _layout(tones, kind="hangup", turns=turns)
    (h0, h1), = layout.holds
    held = audio[int((h0 + 0.5) * SR) : int((h1 - 0.5) * SR), BUSINESS]
    assert np.sqrt((held**2).mean()) > 0.01
    turns = [("business", _tone(2.0), False), ("caller", _tone(1.0), False), ("colleague", _tone(1.0), False)]
    _, layout = _layout(tones, turns=turns)
    assert len(layout.holds) == 1 and layout.holds[0][1] <= layout.turns[2][1]


def test_a_phone_menu_puts_the_key_on_the_caller_channel(tones):
    audio, layout = _layout(tones, ivr=_tone(2.0, 700), key="2")
    assert layout.key == "2" and layout.key_at > layout.notice_seconds
    pressed = audio[int(layout.key_at * SR) : int((layout.key_at + 0.15) * SR), CALLER]
    assert np.abs(pressed).max() > 0.1


def test_the_phone_audio_is_a_stereo_mp3(tones, tmp_path):
    from vconfaker.smarttrunk.telephone import probe, split_channels, to_phone_mp3

    audio, _ = _layout(tones)
    mp3 = to_phone_mp3(audio, tmp_path, "call", random.Random(2))
    duration, channels = probe(mp3)
    assert channels == 2 and duration > 5
    assert all(p.exists() for p in split_channels(mp3, tmp_path))


def test_recognition_hears_only_the_speech_and_maps_times_back(tones, tmp_path):
    import soundfile as sf

    from vconfaker.smarttrunk.telephone import GAP, MARGIN, speech_only, to_call_time, to_phone_mp3

    audio, layout = _layout(tones)
    mp3 = to_phone_mp3(audio, tmp_path, "call", random.Random(3))
    (caller_wav, caller_map), (business_wav, business_map) = speech_only(mp3, layout, tmp_path)
    _, start, end = layout.turns[1]  # the one caller line
    assert len(caller_map) == 1 and to_call_time(0.5, caller_map) == pytest.approx(start - MARGIN + 0.5, abs=0.01)
    assert sf.info(str(caller_wav)).duration == pytest.approx(end - start + 2 * MARGIN + GAP, abs=0.05)
    assert len(business_map) == 3  # the notice and the two business lines, not the silence between


# --- The transcript check --------------------------------------------------------------------------


def test_word_error_rate_treats_digits_and_number_words_alike():
    assert word_error_rate("Call 555-0152, okay?", "call five five five zero one five two OK") == 0.0
    assert word_error_rate("one two three four", "one two three") == 0.25


def test_one_dropped_line_fails_the_call():
    lines = ["Thanks for calling, this is Dana, how can I help you today?",
             "Sure, let me look that up for you now, it should only take a second.", "Okay.",
             "I have openings on Tuesday and Thursday, both in the morning, if either of those suits you.",
             "You're booked for next Tuesday at nine."]
    caller = ["Hi, I need an oil change sometime next week if you have room for me.",
              "Tuesday works for me, thanks so much for fitting me in."]
    heard_all = [{"text": " ".join(caller)}, {"text": " ".join(lines)}]
    assert check_call(heard_all, [caller, lines]).passed
    dropped = [{"text": " ".join(caller)}, {"text": " ".join(lines[:-1])}]  # the last business line is missing
    result = check_call(dropped, [caller, lines])
    assert not result.passed and any("booked for next Tuesday" in u for u in result.uncovered)
    assert result.wer <= check_module.MAX_WER  # the word error rate alone would have let it through
    assert turn_coverage(lines, " ".join(lines[:-1]))[2] is None  # "Okay." is too short to judge


def test_the_transcript_keeps_each_segments_channel():
    heard = [{"text": "hello", "sentences": [{"start": 1.0, "end": 2.0, "text": "hello"}]},
             {"text": "hi there", "sentences": [{"start": 0.0, "end": 0.9, "text": "hi there"}]}]
    result = check_call(heard, [["hello"], ["hi there"]])
    assert [(s["channel"], s["party"]) for s in result.sentences] == [(1, 1), (0, 0)]
    assert result.text == "hi there hello"


# --- The vCon --------------------------------------------------------------------------------------


def _vcon_for(plan, rng=None):
    rng = rng or random.Random(1)
    turns = [("business", 6.0, 9.0), ("caller", 9.5, 12.0)]
    layout = Layout(ring_seconds=8.0, notice_seconds=5.1, turns=turns, ended_at=40.0,
                    caller_dropped_at=39.0 if plan.kind != "conversation" else None,
                    key_at=7.0 if plan.ivr_key else None, key=plan.ivr_key)
    check = check_call([{"text": "Hello.", "sentences": [{"start": 9.6, "end": 10.0, "text": "Hello."}]},
                        {"text": "Hi.", "sentences": [{"start": 6.1, "end": 6.5, "text": "Hi."}]}], [["Hello."], ["Hi."]])
    answered = plan.start + timedelta(seconds=layout.ring_seconds)
    return build_call_vcon(
        plan=plan, pack=PACKS[plan.industry], layout=layout, filename=f"{plan.call_id}.mp3",
        media_url=f"https://media.invalid/x/{plan.call_id}.mp3", content_hash=compute_content_hash(b"mp3"),
        duration=40.0, transcript=check,
        notice={"file": "synthetic-notice-v1.wav", "sha256": "0" * 64, "discloses": ["recording", "transcription"]},
        screen=screen_result(plan, answered, layout, rng), run_label="test",
    ), layout


SAMPLE = [p for industry in INDUSTRIES for p in PLANS[industry][:15]]


@pytest.mark.parametrize("plan", SAMPLE, ids=[p.call_id for p in SAMPLE])
def test_planned_calls_build_valid_synthetic_two_channel_vcons(plan):
    v, layout = _vcon_for(plan)
    assert validate(v) == []
    d = json.loads(json.dumps(v.to_dict()))
    assert d["vcon"] == "0.5.0" and governance_problems(d) == []
    jsonschema.validate(d, json.loads(SCHEMA_FILE.read_text()))
    assert Vcon(d).is_valid() == (True, [])

    assert [p["tel"] for p in d["parties"]] == [plan.caller_tel, plan.business_tel]
    session, dialog = d["sessions"][0], d["dialog"][0]
    started = datetime.fromisoformat(session["start"])
    recorded = datetime.fromisoformat(dialog["start"])
    assert recorded - started == timedelta(seconds=layout.ring_seconds)  # the recording starts at answer
    assert dialog["parties"] == [0, 1]  # one entry per channel
    tags = next(a for a in d["attachments"] if a["purpose"] == "tags")["body"]
    assert "synthetic:true" in tags and "recording_channels:dual" in tags
    assert f"annotation_announcement_served_at:{dialog['start'].replace('+00:00', '.000+00:00')}" in tags or any(
        t.startswith("annotation_announcement_served_at:") and datetime.fromisoformat(t.split(":", 1)[1]) == recorded for t in tags
    )
    assert f"opening:{plan.opening}" in tags and f"length:{plan.length}" in tags
    events = [e["type"] for e in d["events"]]
    assert events.count("join") == 2 and events.count("drop") == 2
    assert ("keydown" in events) == bool(plan.ivr_key)
    segments = d["analysis"][0]["body"]["segments"]
    assert {s["channel"] for s in segments} == {0, 1} and d["analysis"][0]["body"]["channels"] == 2
    assert len([a for a in d["analysis"] if a["type"] == "voice-screen"]) == (1 if plan.screened else 0)
    basis = [a for a in d["attachments"] if a["purpose"] == "lawful_basis"]
    assert len(basis) == 1 and basis[0]["body"]["lawful_basis"] == "legitimate_interests"
    assert not [a for a in d["attachments"] if a["purpose"] == "tenant"]


def test_robocall_screen_result_follows_the_receiver_contract():
    for industry in INDUSTRIES:
        plan = next(p for p in PLANS[industry] if p.kind == "robocall")
        assert plan.screened
        layout = Layout(ring_seconds=6.0, notice_seconds=5.0, turns=[("business", 6.0, 8.0), ("caller", 8.5, 19.0)], ended_at=22.0)
        body = screen_result(plan, plan.start, layout, random.Random(1))
        assert body["status"] == "acted" and body["action_result"] == "accepted" and body["telnyx_status"] == 200


# --- The recognizer process ------------------------------------------------------------------------


def _fake_parakeet(calls, envs, fail_first_with=None):
    def run(cmd, **kwargs):
        envs.append(kwargs["env"]["HF_HUB_OFFLINE"])
        if fail_first_with and len(envs) == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr=fail_first_with)
        files = [a for a in cmd if a.endswith(".mp3")]
        calls.append(files)
        out = Path(cmd[cmd.index("--output-dir") + 1]) if "--output-dir" in cmd else Path(cmd[cmd.index(files[0]) - 1])
        for f in files:
            (out / f"{Path(f).stem}.json").write_text(json.dumps({"text": "hello", "sentences": []}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def test_the_plain_cli_fallback_runs_at_most_five_files_per_process(monkeypatch, tmp_path):
    calls, envs = [], []
    monkeypatch.setattr(check_module, "_interpreter_of", lambda binary: None)
    monkeypatch.setattr(check_module.subprocess, "run", _fake_parakeet(calls, envs))
    audio = [tmp_path / f"call-{i:02d}.mp3" for i in range(12)]
    results = check_module.Transcriber("parakeet-mlx").transcribe_many(audio)
    assert len(results) == 12
    assert [len(c) for c in calls] == [5, 5, 2]


def test_parakeet_goes_online_only_on_a_cache_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(check_module, "_interpreter_of", lambda binary: None)
    calls, envs = [], []
    monkeypatch.setattr(
        check_module.subprocess, "run", _fake_parakeet(calls, envs, fail_first_with="RuntimeError: Metal device lost")
    )
    with pytest.raises(RuntimeError):
        check_module.Transcriber("parakeet-mlx").transcribe_many([tmp_path / "a.mp3"])
    assert envs == ["1"]

    calls, envs = [], []
    miss = "huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache"
    monkeypatch.setattr(check_module.subprocess, "run", _fake_parakeet(calls, envs, fail_first_with=miss))
    assert len(check_module.Transcriber("parakeet-mlx").transcribe_many([tmp_path / "a.mp3"])) == 1
    assert envs == ["1", "0"]
