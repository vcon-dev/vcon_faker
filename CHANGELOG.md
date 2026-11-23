# Changelog

## Unreleased

Smart Trunk call mode (`python -m vconfaker.smarttrunk`), for the Smart Trunk staging corpus.

- Synthetic calls in Smart Trunk's call shape, vCon syntax `0.5.0`: caller and business parties,
  one Session, a `recording` Dialog with an external MP3 and `content_hash`, join and drop Events,
  the receiver's notice evidence as `annotation_*` tags, a `transcript` analysis entry and, on
  screened numbers, a simulated `voice-screen` entry.
- A dealership scenario pack with invented businesses and numbers in 555-0100 to 555-0199.
- Scripts by Claude (`claude-sonnet-5-5`, structured output, server-side fallbacks), checked for
  real names, companies, numbers, emails and web addresses before any audio is made.
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
