"""Command-line entry point for the vcon_faker engine.

Generate synthetic vCons headlessly, with no Streamlit and no AWS required:

    python -m vconfaker generate --provider scripted --type voice --count 5
    python -m vconfaker generate --provider anthropic --storage s3 --count 100 --audio
    python -m vconfaker generate --provider openai --audio --concurrency 4 --whisper

``--provider scripted`` produces deterministic offline output (no API key); add
``--audio`` to attach recording media (silent stub under the scripted provider).
"""
from __future__ import annotations

import argparse

from vconfaker.config import Settings
from vconfaker.engine import generate_many, load_signing_key
from vconfaker.generators.scenario import random_spec
from vconfaker.models import CONVERSATION_TYPES
from vconfaker.providers.llm import get_provider
from vconfaker.providers.stt import get_stt_provider
from vconfaker.providers.tts import get_tts_provider
from vconfaker.storage import get_storage


def cmd_generate(args: argparse.Namespace) -> list[str]:
    settings = Settings.from_env(
        llm_provider=args.provider,
        storage_backend=args.storage,
        output_dir=args.out_dir,
        concurrency=args.concurrency,
        whisper_enabled=args.whisper,
    )
    provider = get_provider(settings, args.provider)
    storage = get_storage(settings)

    want_audio = args.audio and args.type == "voice"
    tts = None
    stt = None
    if want_audio:
        tts_name = "scripted" if args.provider in ("scripted", "fake") else None
        tts = get_tts_provider(settings, tts_name)
        stt = get_stt_provider(settings)

    specs = [
        random_spec(
            business=args.business,
            business_name=args.business_name,
            problem=args.problem,
            emotion=args.emotion,
            conversation_type=args.type,
            disposition=args.disposition,
        )
        for _ in range(args.count)
    ]
    sign_key = load_signing_key(settings.signing_key, args.out_dir) if args.sign else None

    results = generate_many(
        specs,
        settings=settings,
        provider=provider,
        storage=storage,
        generate_audio=want_audio,
        tts=tts,
        stt=stt,
        concurrency=settings.concurrency,
        sign_key=sign_key,
    )
    written = [r.vcon_locator for r in results]
    for locator in written:
        print(locator)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vconfaker", description="Generate synthetic vCons.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate one or more synthetic vCons.")
    gen.add_argument("--count", type=int, default=1, help="How many to generate.")
    gen.add_argument("--concurrency", type=int, default=1, help="Parallel workers.")
    gen.add_argument("--type", choices=list(CONVERSATION_TYPES), default="voice")
    gen.add_argument(
        "--disposition",
        default=None,
        help="Incomplete-call reason (e.g. no-answer, busy); random if omitted.",
    )
    gen.add_argument(
        "--provider",
        default=None,
        help="LLM provider: openai | anthropic | scripted (default: env or openai).",
    )
    gen.add_argument("--storage", choices=["local", "s3", "conserver"], default="local")
    gen.add_argument(
        "--sign",
        action="store_true",
        help="JWS-sign each vCon (uses VCONFAKER_SIGNING_KEY, or generates a "
        "key pair in --out-dir).",
    )
    gen.add_argument("--out-dir", default="./out", help="Local output dir (local storage).")
    gen.add_argument("--audio", action="store_true", help="Attach recording media (voice only).")
    gen.add_argument(
        "--whisper",
        action="store_true",
        help="Run Whisper STT on synthesized audio (requires OPENAI_API_KEY).",
    )
    gen.add_argument("--business", default="Insurance Agency")
    gen.add_argument("--business-name", default="Acme Insurance")
    gen.add_argument("--problem", default="a billing question")
    gen.add_argument("--emotion", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        cmd_generate(args)
        return 0
    return 1
