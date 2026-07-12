# Fake Conversation Generator (vcon_faker)

Generate synthetic conversations between a customer and an agent and package them
as spec-compliant **vCon** (Virtualized Conversation) files. Conversation text is
written by an LLM; for voice calls, audio is synthesized with text-to-speech.
Output can be written locally or uploaded to Amazon S3.

This code is available on GitHub: https://github.com/vcon-dev/vcon_faker

Every generated vCon is marked **synthetic** (a `lawful_basis` attachment, a
`tags` attachment with `synthetic: "true"`, and `validation: "synthetic"` on each
party). See [`CONSENT.md`](CONSENT.md). vCons conform to IETF
`draft-ietf-vcon-vcon-core-02` (syntax `0.4.0`) and are built with the official
`vcon` library (0.9.6+).

## Features

- Five conversation modes, four of which need **no audio synthesis at all**:
  - `voice` - phone call; optional TTS audio, otherwise transcript-only with a
    WTF transcription analysis
  - `messaging` - SMS-style thread, one text dialog per message
  - `email` - email exchange with hours-apart timestamps
  - `incomplete` - a call attempt that never connected (`no-answer`, `busy`,
    ...), generated fully offline with no LLM call
  - `transfer` - three parties (customer, agent, supervisor) with a spec
    `transfer` dialog at the handoff
- Structured LLM output (JSON schema / tool use) keeps agent, customer, and
  supervisor names consistent
- LLM providers: **OpenAI**, **Anthropic Claude**, or offline **`scripted`**
- Synthesizes per-turn audio via OpenAI text-to-speech (voice calls)
- Optional **Whisper round-trip** on synthesized audio for real recognizer confidence
- Builds spec-compliant vCons with WTF transcription analysis and synthetic-data governance
- Optional **JWS signing** (`--sign`) with a generated or configured RSA key pair
- Writes output locally, to S3, or straight into a **vcon-server ingress**
- Three front ends over one engine: Streamlit UI, CLI, and a FastAPI service
- Concurrent batch generation via `--concurrency` / API `concurrency` field

## Architecture

Generation logic lives in the `vconfaker` engine package, independent of Streamlit:

- `vconfaker/config.py` — typed settings from the environment
- `vconfaker/schemas/` — Pydantic models for structured dialogue output
- `vconfaker/providers/` — LLM (`scripted` | `openai` | `anthropic`), TTS, STT
- `vconfaker/generators/` — scenario + conversation generation (shared prompts)
- `vconfaker/storage.py` — output sinks (`local` | `s3`)
- `vconfaker/vconbuild/builder.py` — spec-compliant vCon assembly
- `vconfaker/engine.py` — the `generate_vcon` / `generate_many` pipeline

The Streamlit app (`vcon_faker.py`), the CLI (`python -m vconfaker`), and the API
(`vconfaker/api.py`) are thin clients over the engine. The modernization roadmap
is in [`MODERNIZATION_PLAN.md`](MODERNIZATION_PLAN.md).

## Prerequisites

- Python 3.12+
- Packages from `requirements.txt` (`pip install -r requirements.txt`)
- For real generation: an OpenAI and/or Anthropic API key (and AWS credentials + an S3 bucket if using S3 storage)

## Configuration

The engine reads configuration from the environment (the CLI and API) or from
Streamlit secrets (the UI). Recognized keys:

```
OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TTS_MODEL, WHISPER_MODEL
ANTHROPIC_API_KEY, ANTHROPIC_MODEL
AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET
CONSERVER_URL, CONSERVER_API_TOKEN, CONSERVER_INGRESS
VCONFAKER_LLM_PROVIDER, VCONFAKER_STORAGE, VCONFAKER_OUTPUT_DIR
VCONFAKER_CONCURRENCY, VCONFAKER_WHISPER, VCONFAKER_SIGNING_KEY
```

For the Streamlit UI, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and fill in your values (this file is git-ignored).
The CLI and API read the same names from the environment.

## Running

### Streamlit UI

```
streamlit run vcon_faker.py
```

Open http://localhost:8501, choose options, edit the prompt in the sidebar, and
click "Generate Conversation(s)".

### CLI (headless, no AWS required)

```
# Fully offline + deterministic; writes vCons under ./out
python -m vconfaker generate --provider scripted --type voice --count 5

# OpenAI with audio uploaded to S3, four workers in parallel
python -m vconfaker generate --provider openai --storage s3 --count 100 --audio --concurrency 4

# Claude-authored dialogue
python -m vconfaker generate --provider anthropic --count 10

# Whisper STT round-trip on synthesized audio (real recognizer confidence)
python -m vconfaker generate --provider openai --audio --whisper --count 5

# No-synthesis modes: email thread, missed call, transferred call
python -m vconfaker generate --provider scripted --type email --count 3
python -m vconfaker generate --provider scripted --type incomplete --disposition busy
python -m vconfaker generate --provider anthropic --type transfer

# JWS-sign each vCon (key from VCONFAKER_SIGNING_KEY, or generated in --out-dir)
python -m vconfaker generate --provider scripted --type messaging --sign

# POST straight into a vcon-server ingress (text modes; audio needs local/s3)
CONSERVER_URL=http://localhost:8000 CONSERVER_INGRESS=faker \
  python -m vconfaker generate --provider scripted --type messaging --storage conserver
```

### API

```
uvicorn vconfaker.api:app
# GET  /healthz
# POST /generate  {"count": 2, "conversation_type": "voice", "provider": "anthropic", "concurrency": 4}
```

## Development

```
pip install -r requirements.txt pytest ruff mypy
ruff check vconfaker vcon_builder.py tests
mypy vconfaker vcon_builder.py
pytest -q
```

CI runs lint, type-check, and the vCon compliance test suite on every push and PR
(`.github/workflows/ci.yml`).

## Customization

- Default prompts live in `vconfaker/generators/conversation.py`
  (`DEFAULT_VOICE_PROMPT`, `DEFAULT_MESSAGING_PROMPT`); the Streamlit sidebar lets
  you override them per run.
- `fake_names.py` holds the name / business / problem / emotion lists used to
  randomize scenarios.
- Add a new LLM or TTS backend by subclassing `LLMProvider` / `TTSProvider` and
  extending the corresponding `get_*` factory.

## License

Open-source under the [MIT License](LICENSE).

## Acknowledgements

- [OpenAI](https://openai.com/) for the language and text-to-speech models.
- [Anthropic](https://www.anthropic.com/) for Claude dialogue generation.
- [Streamlit](https://streamlit.io/) for the web interface.
- [Amazon Web Services](https://aws.amazon.com/) for S3 storage.

Contributions welcome via pull requests or issues on the GitHub repository.
