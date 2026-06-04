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
`vcon` library.

## Features

- Generates conversations from editable prompts (voice or text messaging)
- Synthesizes per-turn audio via OpenAI text-to-speech (voice calls)
- Builds spec-compliant vCons with transcript/summary analysis and synthetic-data governance
- Writes output locally or to S3
- Three front ends over one engine: Streamlit UI, CLI, and a FastAPI service
- An offline `scripted` provider needs no API key (great for tests and demos)

## Architecture

Generation logic lives in the `vconfaker` engine package, independent of Streamlit:

- `vconfaker/config.py` — typed settings from the environment
- `vconfaker/providers/` — LLM (`scripted` | `openai`) and TTS providers
- `vconfaker/generators/` — scenario + conversation generation (shared prompts)
- `vconfaker/storage.py` — output sinks (`local` | `s3`)
- `vconfaker/vconbuild/builder.py` — spec-compliant vCon assembly
- `vconfaker/engine.py` — the `generate_vcon` pipeline shared by every front end

The Streamlit app (`vcon_faker.py`), the CLI (`python -m vconfaker`), and the API
(`vconfaker/api.py`) are thin clients over the engine. The modernization roadmap
is in [`MODERNIZATION_PLAN.md`](MODERNIZATION_PLAN.md).

## Prerequisites

- Python 3.12+
- Packages from `requirements.txt` (`pip install -r requirements.txt`)
- For real generation: an OpenAI API key (and AWS credentials + an S3 bucket if using S3 storage)

## Configuration

The engine reads configuration from the environment (the CLI and API) or from
Streamlit secrets (the UI). Recognized keys:

```
OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TTS_MODEL
AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET
VCONFAKER_LLM_PROVIDER, VCONFAKER_STORAGE, VCONFAKER_OUTPUT_DIR
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

# Real OpenAI generation with audio, uploaded to S3
python -m vconfaker generate --provider openai --storage s3 --count 100 --audio
```

### API

```
uvicorn vconfaker.api:app
# GET  /healthz
# POST /generate  {"count": 2, "conversation_type": "voice", "provider": "openai"}
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
- [Streamlit](https://streamlit.io/) for the web interface.
- [Amazon Web Services](https://aws.amazon.com/) for S3 storage.

Contributions welcome via pull requests or issues on the GitHub repository.
