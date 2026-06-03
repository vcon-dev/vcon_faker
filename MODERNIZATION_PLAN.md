# vcon_faker — Modernization Plan ("State of the Art")

## Context

`vcon_faker` is a Streamlit app (one ~1,030-line `vcon_faker.py`) that generates
synthetic customer/agent conversations: OpenAI Chat writes the dialogue, OpenAI
TTS synthesizes audio, the `vcon` library assembles a vCon, and everything lands
in S3 behind presigned URLs. Its output is published as the `fake-vcons` corpus.

It works, but it has drifted from the modern stack in three ways:

1. **Spec drift.** It pins `vcon ^0.5.0`; the current library is **0.9.4**. It
   therefore predates the entire 0.9.0 extension framework (lawful_basis, WTF
   transcription, the `build_new()` → `"vcon":"0.4.0"` fix, the `add_tag`
   compliance fix). The generated vCons carry **no synthetic / lawful_basis
   markers**, a wrong audio mimetype, a dict-valued analysis body with
   `encoding:"none"`, and a hardcoded transcript confidence of `0.99`.
2. **Engineering debt.** No tests, no CI, no type hints, no linter; a single
   monolith tightly coupled to Streamlit; secrets read at import time.
3. **AI staleness.** Single-provider (OpenAI only), freeform `json_object`
   prompting, `tts-1`-era voices, sequential generation, two-party-only
   conversations with no real acoustics or real transcription.

**Goal:** turn `vcon_faker` from a demo script into a state-of-the-art,
spec-perfect, multi-provider **synthetic conversation engine** that produces
governed, signed, dataset-ready vCon corpora at scale — and showcases the vCon
standard the way its reference generator should.

---

## Target architecture (the bold version)

```
                 ┌─────────────────────────────────────────────┐
                 │              vcon_faker (engine)             │
  Streamlit UI ──┤                                              │
  CLI ───────────┤  generators/   providers/        vconbuild/  │
  FastAPI ───────┤   conversation  llm (claude|openai|local)    │
  (REST/MCP)     │   persona       tts (openai|elevenlabs)       │
                 │   scenario      stt (whisper)    storage      │
                 │   acoustics     ──────────────   (s3|gcs|     │
                 │                                   local|mcp)  │
                 └───────────────┬──────────────────────────────┘
                                 │ signed, lawful_basis-marked vCons
                                 ▼
                 fake-vcons corpus · vcon-server · vcon-mcp · HF datasets
```

Core principle (from the vCon speckit): **library-first, always.** Build every
vCon through `vcon` 0.9.4 helpers (`add_party`, `add_dialog`,
`add_analysis`, `add_attachment`, `add_tag`, `add_wtf_transcription_analysis`,
lawful_basis helpers, `sign`). No hand-rolled dicts.

---

## Phase 0 — Foundation & truth (compliance + hygiene) · highest ROI

Make the output correct and the project safe to change. Most of this is small,
high-leverage edits plus the library bump.

**Spec / library**
- Upgrade `vcon ^0.5.0` → **`vcon 0.9.4`** (pinned). Re-verify against
  `vcon-dev/vcon-lib` `CHANGELOG.md` at execution time — source wins.
- Confirm `Vcon.build_new()` emits `"vcon":"0.4.0"` and no empty
  `group`/`redacted` (fixed in 0.9.2) — remove any manual version/placeholder
  handling.
- **Fix the analysis bodies** in `create_vcon_object()` (`vcon_faker.py:376-410`):
  the transcript body is a `dict` with `encoding:"none"` — non-compliant. Body
  must be a **string**; use `encoding:"json"` with `json.dumps(...)`. Field is
  `schema`, never `schema_version`; `vendor` stays required.
- **Replace the transcript analysis** with `add_wtf_transcription_analysis()`
  (type `"transcription"`, `vendor`, `product`, `schema` = WTF draft URL,
  `encoding:"json"`). Drop the redundant plain-text transcript — the WTF doc
  already carries `transcript.text`. Keep `summary`; reconsider `diarized`.
- **Fix media metadata** (`vcon_faker.py:346-363`): mimetype is `"audio/x-wav"`
  but the file is MP3 → `"audio/mpeg"`. External media must carry `url` **and**
  `content_hash` formatted `sha512-<base64url>` (no padding/hex). Align the
  existing SHA-512 hashing to the `content_hash` field/format the 0.9.4 `Dialog`
  expects.
- Fix the messaging-path party where `name` is set to the phone number
  (`vcon_faker.py:273`) → use the real synthetic name.

**Synthetic-corpus governance (the canonical pattern, currently absent)**
- Add a **lawful_basis attachment** (`purpose:"lawful_basis"`,
  `lawful_basis:"legitimate_interests"`, `expiration:null`, explicit
  `purpose_grants` for recording/transcription/analysis/redistribution, a
  `proof_mechanism` of `external_system` documenting synthetic origin, and
  `terms_of_service` → a repo-root **`CONSENT.md`** URL). Add `"lawful_basis"`
  to top-level `extensions[]`.
- Add a **tags attachment** carrying `synthetic:"true"` plus business/scenario
  metadata, via `add_tag`.
- Mark every party `validation:"synthetic"`.
- Write **`CONSENT.md`** describing exactly those three markers (do **not**
  invent a `synthetic_data_consent` attachment — that retired design is a
  known doc-drift trap).

**Engineering hygiene**
- Add a **pytest suite**: golden-vCon snapshot tests, `Vcon.is_valid()`
  assertions, and a spec-compliance test that runs the `vcon-compliance` audit
  rules (legacy field names, syntax `0.4.0`, attachment `purpose`, analysis
  `schema`+`vendor`, media `content_hash`). Mock OpenAI/Anthropic and S3.
- Add **GitHub Actions** CI: `ruff` + `mypy` + `pytest` + Docker build, with a
  **vCon-compliance gate** that fails the build on any drift.
- Add **type hints** throughout, `ruff` + `mypy` config in `pyproject.toml`, and
  `pre-commit`.
- Move config to **`pydantic-settings`** with `os.environ` fallback (decouple
  from `st.secrets`); add `.env.example`. Pin floating deps (`ffmpeg`, `vcon`).

---

## Phase 1 — Re-architect: monolith → engine

Split `vcon_faker.py` into a real package so the generator is usable from a UI,
a CLI, an API, or a cron job — and testable in isolation.

```
vcon_faker/
  config.py            # pydantic-settings
  providers/
    llm.py             # provider-agnostic chat (claude | openai | local)
    tts.py             # openai | elevenlabs | gtts
    stt.py             # whisper (real transcripts)
    storage.py         # s3 | gcs | local | vcon-mcp | vcon-server
  generators/
    conversation.py    # dialogue (structured output)
    persona.py         # party personas, voices, demographics
    scenario.py        # business/problem/emotion → scenario spec
    acoustics.py       # telephony codec, hold, overlap, noise
  vconbuild/builder.py # library-first vCon assembly + sign()
  cli.py               # `vcon-faker generate ...`
  api.py               # FastAPI (REST + optional MCP tool surface)
  ui/streamlit_app.py  # thin client over the engine
```

- Engine is **Streamlit-free**; Streamlit, CLI, and FastAPI are thin clients.
- **Async + concurrent** batch generation (today it's a sequential loop up to
  100). A job/queue model for large corpora with progress + resume.
- **Pluggable storage** including direct write into `vcon-server` / `vcon-mcp`
  so generated data lands in the real conserver, not just S3.

---

## Phase 2 — AI modernization (the leap)

**Multi-provider LLM, Claude first-class**
- Provider abstraction with **Claude (Opus/Sonnet 4.x)** as the default for
  dialogue generation, OpenAI as an alternative, and a local option.
- Replace freeform `response_format=json_object` with **structured output**
  (tool use / JSON schema validated by Pydantic) so turns, speakers, and
  metadata are guaranteed-shaped — no post-hoc name-consistency patching.

**Richer, more realistic conversations**
- **Personas & emotional arcs** instead of a single static emotion: tension
  that builds/resolves, satisfaction trajectories.
- **Multi-party** (3+: customer, agent, supervisor/transfer, IVR), interruptions,
  disfluencies, **multilingual & code-switching**.
- **Domain profiles** straight from the speckit `VCON_USAGE_GUIDE`
  (healthcare, contact-center, sales, support, legal) — each with correct
  party roles, vocabulary, and analysis expectations.

**Modern media**
- TTS: **OpenAI `gpt-4o-mini-tts`** (steerable/emotional) and **ElevenLabs**
  (reuse `vcon-eleven-labs-adapter` patterns); per-party voice assignment.
- **Realistic acoustics**: telephony codec/bandwidth, hold music, background
  noise, light speaker overlap — so audio reads as real call recordings.
- **Real transcription round-trip**: run generated audio back through
  **Whisper** to produce a genuine **WTF transcript with real confidence**,
  replacing the hardcoded `0.99`.

**Realism evals**
- LLM-as-judge realism scoring, distributional checks (length, sentiment,
  topic spread), and dedup, gated in CI so corpus quality is measured, not
  assumed.

---

## Phase 3 — Corpus & dataset platform

Treat the output (`fake-vcons`) as a first-class, governed product.

- **Deterministic, seeded generation** for reproducible corpora; a scenario/
  template library checked into the repo.
- **Corpus manifests + provenance**: what generated each vCon, with which
  models/seeds; enforce lawful_basis + synthetic markers corpus-wide.
- **Signed vCons** via `vcon.sign()` / `verify()` (RS256 JWS); optional
  **SCITT / lifecycle** registration hooks tying into the lifecycle work.
- **Dataset packaging**: export to JSONL / Parquet / **HuggingFace datasets**
  for ML training, alongside the raw vCon JSON.
- **Publish pipeline** into `fake-vcons`, `vcon-server`, and `vcon-mcp`.

---

## Phase 4 — UX, deploy, ops

- Keep Streamlit for quick interactive use; if hosted, add a richer front end
  (optional, bolder) and **auth**.
- Deploy: hardened container → Cloud Run / Fly / Render; **cost controls**
  (per-run token/$ budgets), rate limiting.
- **Observability**: structured logging, OpenTelemetry traces, and token/cost
  metrics per generation.
- Docs site + examples + `CONTRIBUTING`; refresh the `.vcon-repo/graph.json`
  knowledge graph (currently 11 nodes, 0 edges) to reflect the new architecture.

---

## Quick wins (do these first, in order)

1. Bump `vcon` → 0.9.4; adopt `build_new()`, fix analysis body
   (`encoding:"json"` + string), fix mimetype → `audio/mpeg`, fix media
   `content_hash`.
2. Add lawful_basis + tags(`synthetic:"true"`) + party `validation:"synthetic"`
   + `CONSENT.md`.
3. Switch transcript to `add_wtf_transcription_analysis()`.
4. pytest + vCon-compliance test + GitHub Actions CI gate.
5. Structured-output dialogue generation (kills the name-consistency hack).

---

## Verification

- **Compliance:** run the `vcon-compliance` skill against the repo and a
  freshly generated vCon; assert syntax `0.4.0`, no legacy field names
  (`schema_version`, `appended`, `must_support`, `did`, attachment `type`),
  `vendor` present on every analysis, media `content_hash` formatted
  `sha512-<base64url>`, and `lawful_basis` in `extensions[]`. `Vcon.is_valid()`
  must pass.
- **Functional:** `pytest` green; generate 1 voice + 1 messaging vCon via CLI
  and via Streamlit; confirm audio plays, WTF transcript carries real
  confidence, and the vCon round-trips through `vcon.verify()`.
- **Integration:** write a generated vCon into `vcon-mcp` / `vcon-server` and
  read it back; load a generated batch as a HuggingFace dataset.
- **CI:** the Actions workflow blocks merges on lint, type, test, or compliance
  failure.

---

## Key decisions for you (forks in this plan)

1. **Scope:** full re-architecture (Phases 0–4) vs. just the spec/compliance +
   CI hardening (Phase 0). The plan assumes you want bold = all of it.
2. **LLM provider:** Claude-first multi-provider (assumed) vs. stay OpenAI-only.
3. **Surface:** keep Streamlit-only vs. add CLI + FastAPI/MCP engine (assumed).
4. **Output target:** stay S3 + `fake-vcons` vs. also push into
   `vcon-server` / `vcon-mcp` as the corpus backbone (assumed).
