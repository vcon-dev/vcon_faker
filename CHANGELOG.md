# Changelog

## Unreleased

### Smart Trunk call mode, scale-up

Four industries, and calls that sound less alike.

- Packs for medical and dental clinics, municipal offices and home services beside dealerships,
  with invented businesses, towns and streets, and a real-name block list for each industry
  (hospital systems, pharmacies, insurers; states, cities, agencies and utilities; national
  service chains and marketplaces). The municipal pack also refuses any place it was not given.
- Variety: six ways of opening a call, short, medium and long calls (many 15 to 45 s), phone
  menus with a key press (`keydown`/`keyup` Events), transfers to a colleague after hold music,
  interruptions, and wrong numbers.
- Scripts that book something for the moment of the call are thrown away.
- The recording starts at answer with the notice; the ringback before it sets only the
  Session's start.
- Two-channel recordings: caller on channel 0, business side on channel 1, so the recording
  Dialog's `parties` is `[0, 1]` with one entry per channel, and a `recording_channels:dual`
  tag as the carrier adapters write it. The transcript keeps each segment's channel and party.
- The transcript check hears each channel's speech on its own and fails a call when any line of
  three words or more was not heard, not only on the word error rate.
- Holds play hold music instead of silence.
- The run pauses while free memory is under a floor, and stops writing scripts at a spending cap
  shared by a corpus's runs; an overall index page covers a corpus.

### Smart Trunk call mode

`python -m vconfaker.smarttrunk`, for the Smart Trunk staging corpus.

- Synthetic calls in Smart Trunk's call shape, vCon syntax `0.5.0`: caller and business parties,
  one Session, a `recording` Dialog with an external MP3 and `content_hash`, join and drop Events,
  the receiver's notice evidence as `annotation_*` tags, a `transcript` analysis entry and, on
  screened numbers, a simulated `voice-screen` entry.
- A dealership scenario pack with invented businesses and numbers in 555-0100 to 555-0199.
- Scripts by Claude (`claude-sonnet-5-5`, structured output, server-side fallbacks), checked for
  real names, companies, numbers, emails and web addresses before any audio is made. Real
  business names are kept per industry (`REAL_NAMES`: common, dealership, clinic, municipal,
  home services) and matched in any case from five letters up; phone numbers are matched by
  shape, written or spoken digit by digit, so a bare stock number is not one.
- Local voices with Kokoro-82M (mlx-audio), telephone audio with ffmpeg, and a transcript check
  with parakeet-mlx that re-voices garbled calls.
- `vconfaker/governance.py`: the synthetic markers (`lawful_basis` with `legitimate_interests`,
  `tags` with `synthetic:true`, `validation: "synthetic"` on every party), ported from the
  `modernize-phase-2` branch to core-05 scoping (`party` plus `session`, JSON bodies as values).
- `CONSENT.md`, from `modernize-phase-2`, updated for core-05.
- Offline tests in `tests/test_smarttrunk.py`.

## 0.2.0

Port to vCon core-05 (syntax "0.5.0").

- Depend on `vcon>=1.0.3,<2` (was `vcon==0.5.0`). vcon 1.x needs `requests>=2.33` and
  `authlib>=1.6.12`, so `requests` is now 2.34.2 and `poetry.lock` / `requirements.txt` are
  regenerated (authlib 1.9.9, cryptography 50.0.2, mutagen and joserfc added; pytest
  dropped from the runtime requirements).
- Generated vCons carry `vcon: "0.5.0"` and one Session per conversation. Session
  `parties` lists the participants (the originator first); Dialog `parties` lists
  the contributors.
- Dialog: `mimetype` is now `mediatype`; `alg` and `signature` are replaced by
  `content_hash` (`sha512-<base64url>`); `originator` is gone (it lives in the Session).
  Text Dialogs set `encoding: "none"`. Voice conversations without audio carry the
  transcript as the text Dialog body.
- Transcript analysis body is a plain string (core-05 schema); confidence and language
  moved to the analysis `meta`.
- No Events are emitted: the faker has never produced joins, drops, holds or transfers.
- No `lawful_basis` attachment is added: the faker has never modelled consent, and a
  consent record is not invented.
- Add pytest smoke test (`tests/`) validating generated vCons against the vendored
  core-05 JSON schema (`tests/appendix-c-json-schema.json`) and with `vcon`'s
  `is_valid()` / `get_validation_warnings()`. The LLM call is mocked. Run with
  `pip install pytest && pytest`.
- Bump version to 0.2.0.
