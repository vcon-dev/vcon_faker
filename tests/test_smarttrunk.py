"""Offline tests for the Smart Trunk call mode: no API, no voices, no recognizer.

Needs ``vcon>=1.0.3`` and ``jsonschema``. The audio stages are exercised by a real run, not here.
"""
import json
import random
from types import SimpleNamespace

import fake_names
import jsonschema
import pytest
from vcon import Vcon, compute_content_hash

from vconfaker.governance import governance_problems
from vconfaker.smarttrunk.build import SCHEMA_FILE, build_call_vcon, screen_result, validate
from vconfaker.smarttrunk.check import Check, word_error_rate
from vconfaker.smarttrunk.hygiene import check_script
from vconfaker.smarttrunk.pack import build_plan, fictional_number, load_pack
from vconfaker.smarttrunk.telephone import Layout
from vconfaker.smarttrunk.writer import ScriptWriter, allowed_names, products

PACK = load_pack("dealership")
PLAN = build_plan(PACK)


def test_pilot_plan_is_twenty_calls_on_fictional_numbers():
    assert len(PLAN) == 20
    assert {p.scenario["id"] for p in PLAN} == set(PACK["pilot_mix"])
    for p in PLAN:
        for tel in (p.caller_tel, p.business_tel):
            assert tel.startswith("+1") and tel[5:8] == "555" and 100 <= int(tel[8:]) <= 199
        assert p.caller.voice != p.agent.voice
    assert len({p.caller_tel for p in PLAN}) == 20
    assert len({(p.caller.voice, p.agent.voice) for p in PLAN}) > 10  # voices vary across calls


def test_numbers_outside_the_fictional_range_are_refused():
    with pytest.raises(ValueError):
        fictional_number("614", "555-0200")
    with pytest.raises(ValueError):
        fictional_number("614", "555-0099")


def _script(plan, extra_turn=None):
    turns = [
        {"speaker": "business", "text": f"{plan.business['name']} service, this is {plan.agent.name}. How can I help?"},
        {"speaker": "caller", "text": f"Hi, it's {plan.caller.name}. I need an oil change on my {plan.owned_vehicle}."},
        {"speaker": "business", "text": "Sure. Does Thursday at ten work?"},
        {"speaker": "caller", "text": f"Thursday works. My number is 555-{plan.caller_tel[-4:]}."},
        {"speaker": "business", "text": "You're all set. See you then."},
        {"speaker": "caller", "text": "Thanks, bye."},
    ]
    if extra_turn:
        turns.insert(4, {"speaker": "caller", "text": extra_turn})
    return {"summary": "s", "outcome": "booked", "turns": turns, "names_used": allowed_names(plan)}


def _check(plan, script):
    return check_script(script, allowed_names=allowed_names(plan), products=products(plan, PACK), turn_range=(4, 10))


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
    plan = next(p for p in PLAN if p.caller.name)
    given = {w for n in allowed_names(plan) for w in n.split()}
    other = next(f"{f} {l}" for f in fake_names.male_names for l in fake_names.last_names if f not in given and l not in given)
    line = line.format(other=other)
    verdict = _check(plan, _script(plan, line))
    assert not verdict.ok and any(reason in p for p in verdict.problems), verdict.problems


def test_hygiene_passes_a_clean_script_and_rejects_unlisted_names():
    plan = next(p for p in PLAN if p.caller.name)
    assert _check(plan, _script(plan)).ok
    script = _script(plan)
    script["names_used"].append("Somebody Else")
    assert not _check(plan, script).ok


def test_word_error_rate_treats_digits_and_number_words_alike():
    assert word_error_rate("Call 555-0152, okay?", "call five five five zero one five two OK") == 0.0
    assert word_error_rate("one two three four", "one two three") == 0.25


def _vcon_for(plan):
    layout = Layout(notice_at=2.8, notice_seconds=5.1, answered_at=11.0, ended_at=61.0,
                    caller_dropped_at=60.4 if plan.kind != "conversation" else None,
                    turns=[(11.0, 20.0), (20.5, 40.0)])
    check = Check(True, 0.05, "Hello there.", [{"start": 0.0, "end": 1.0, "text": "Hello there.", "confidence": 0.9}])
    rng = random.Random(1)
    return build_call_vcon(
        plan=plan, pack=PACK, layout=layout, filename=f"{plan.call_id}.mp3",
        media_url=f"https://media.invalid/x/{plan.call_id}.mp3", content_hash=compute_content_hash(b"mp3"),
        duration=61.0, transcript=check,
        notice={"file": "synthetic-notice-v1.wav", "sha256": "0" * 64, "discloses": ["recording", "transcription"]},
        screen=screen_result(plan, plan.start, layout, rng), run_label="test",
    )


@pytest.mark.parametrize("plan", PLAN, ids=[p.call_id for p in PLAN])
def test_every_planned_call_builds_a_valid_synthetic_vcon(plan):
    v = _vcon_for(plan)
    assert validate(v) == []
    d = json.loads(json.dumps(v.to_dict()))
    assert d["vcon"] == "0.5.0"
    assert governance_problems(d) == []
    jsonschema.validate(d, json.loads(SCHEMA_FILE.read_text()))
    assert Vcon(d).is_valid() == (True, [])

    assert [p["tel"] for p in d["parties"]] == [plan.caller_tel, plan.business_tel]
    assert d["sessions"][0]["parties"] == [0, 1] and d["sessions"][0]["dialog"] == [0]
    assert d["dialog"][0]["type"] == "recording" and d["dialog"][0]["content_hash"].startswith("sha512-")
    assert [e["type"] for e in d["events"]] == ["join", "join", "drop", "drop"]
    assert d["analysis"][0]["type"] == "transcript"
    screens = [a for a in d["analysis"] if a["type"] == "voice-screen"]
    assert len(screens) == (1 if plan.screened else 0)

    basis = [a for a in d["attachments"] if a["purpose"] == "lawful_basis"]
    assert len(basis) == 1 and basis[0]["body"]["lawful_basis"] == "legitimate_interests"
    assert basis[0]["session"] == 0 and basis[0]["party"] == 0
    tags = next(a for a in d["attachments"] if a["purpose"] == "tags")["body"]
    assert "synthetic:true" in tags and f"scenario:{plan.scenario['id']}" in tags
    assert any(t.startswith("annotation_announcement_sha256:") for t in tags)
    assert not [a for a in d["attachments"] if a["purpose"] == "tenant"]  # the loader adds it
    assert all(p["validation"] == "synthetic" for p in d["parties"])


def test_robocall_screen_result_follows_the_receiver_contract():
    plan = next(p for p in PLAN if p.kind == "robocall")
    layout = Layout(answered_at=11.0, ended_at=30.0, turns=[(11.0, 14.0), (14.5, 25.0)])
    assert plan.screened  # the plan puts robocalls on a number that screens
    body = screen_result(plan, plan.start, layout, random.Random(1))
    assert body["status"] == "acted" and body["action_result"] == "accepted"
    assert body["acted_at"] and body["telnyx_status"] == 200


def test_writer_retries_a_rejected_script_and_counts_tokens():
    plan = next(p for p in PLAN if p.kind == "hangup")  # 4 to 6 turns, like the test script
    bad = _script(plan, "Call me at 555-0300.")
    good = _script(plan)
    replies = iter([bad, good])

    def create(**kwargs):
        assert kwargs["model"] == "claude-sonnet-5-5"
        assert kwargs["fallbacks"] == "default"
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        body = next(replies)
        return SimpleNamespace(
            stop_reason="end_turn", stop_details=None,
            content=[SimpleNamespace(type="text", text=json.dumps(body))],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=0,
                                  cache_read_input_tokens=0, iterations=None),
        )

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    writer = ScriptWriter(PACK, client=client)
    script = writer.write(plan)
    assert script.attempts == 2 and script.rejected
    assert writer.usage.requests == 2
    assert writer.usage.cost_usd == pytest.approx((2000 * 2 + 1000 * 10) / 1e6)


# --- Review fixes: each test below failed before its fix. ---------------------------------------

from pathlib import Path  # noqa: E402

from vconfaker.smarttrunk import check as check_module  # noqa: E402
from vconfaker.smarttrunk import hygiene  # noqa: E402
from vconfaker.smarttrunk.check import stray_numbers  # noqa: E402


def _caller_plan():
    return next(p for p in PLAN if p.caller.name)


def _real_business_hits(line, **kwargs):
    plan = _caller_plan()
    verdict = check_script(
        _script(plan, line), allowed_names=allowed_names(plan), products=products(plan, PACK), turn_range=(4, 10), **kwargs
    )
    return [p for p in verdict.problems if "real business" in p]


@pytest.mark.parametrize(
    "line",
    [
        "Honestly carmax offered me more.",  # lower case, five or more letters
        "I priced one at the tesla store.",
        "Walmart had wiper blades cheaper.",
        "I could get it at Costco.",
        "Mavis did my tires last time.",
        "My loan is with Santander.",
        "I saw floor mats at Sam's Club.",
        "Home Depot sells a charger like that.",
    ],
)
def test_real_business_names_are_rejected_in_any_case(line):
    assert _real_business_hits(line)


@pytest.mark.parametrize(
    "line", ["What's your budget for this?", "I'll chase that down for you.", "It's an enterprise fleet thing."]
)
def test_short_or_ambiguous_names_stay_ordinary_words_in_lower_case(line):
    assert not _real_business_hits(line)


def test_each_industry_keeps_its_own_list_of_real_names(monkeypatch):
    assert {"common", "dealership", "clinic", "municipal", "home_services"} <= set(hygiene.REAL_NAMES)
    monkeypatch.setitem(hygiene.REAL_NAMES, "clinic", ("Examplecare Health",))
    line = "Examplecare Health has my records."
    assert _real_business_hits(line, industry="clinic")
    assert not _real_business_hits(line, industry="dealership")
    assert _real_business_hits("Zentrova Pharmacy filled it.", extra_real=["Zentrova"])  # a pack's own names


def test_a_one_word_organisation_not_given_rejects_the_script(monkeypatch):
    plan = _caller_plan()
    monkeypatch.setattr(hygiene, "_entities", lambda texts: [("ORG", "Zentrova", False)])
    assert any("organisation" in p for p in _check(plan, _script(plan)).problems)
    monkeypatch.setattr(hygiene, "_entities", lambda texts: [("ORG", "Service", False)])
    assert _check(plan, _script(plan)).ok


def test_spelled_out_numbers_outside_the_range_are_caught():
    plan = _caller_plan()
    verdict = _check(plan, _script(plan, "Call me on five five five zero two zero zero."))
    assert any("phone number" in p for p in verdict.problems), verdict.problems
    assert _check(plan, _script(plan, "Call me on five five five, zero one five two.")).ok
    assert stray_numbers("call me on five five five zero two zero zero")
    assert stray_numbers("call me on five five five zero one five two") == []


def test_a_seven_digit_stock_number_is_not_a_phone_number():
    plan = _caller_plan()
    verdict = _check(plan, _script(plan, "The stock number is 1234567."))
    assert not any("phone number" in p for p in verdict.problems), verdict.problems
    assert stray_numbers("the stock number is 1234567", expected="The stock number is 1234567.") == []
    # a phone number the recognizer ran together is still caught
    assert stray_numbers("my number is 5550200", expected="My number is 555-0152.") == ["5550200"]


def test_an_area_coded_number_in_the_fictional_range_passes():
    plan = _caller_plan()
    assert _check(plan, _script(plan, "Or try (614) 555-0142.")).ok


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
