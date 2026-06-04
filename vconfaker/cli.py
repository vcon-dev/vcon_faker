"""Command-line entry point for the vcon_faker engine.

Generate synthetic vCons headlessly, with no Streamlit and no AWS required:

    python -m vconfaker generate --provider scripted --type voice --count 5
    python -m vconfaker generate --provider openai --storage s3 --count 100 --audio

``--provider scripted`` produces deterministic offline output (no API key); add
``--audio`` to attach recording media (silent stub under the scripted provider).
"""
from __future__ import annotations

import argparse

from vconfaker.config import Settings
from vconfaker.engine import generate_vcon
from vconfaker.generators.scenario import random_spec
from vconfaker.providers.llm import get_provider
from vconfaker.providers.tts import get_tts_provider
from vconfaker.storage import get_storage


def cmd_generate(args: argparse.Namespace) -> list[str]:
    settings = Settings.from_env(
        llm_provider=args.provider,
        storage_backend=args.storage,
        output_dir=args.out_dir,
    )
    provider = get_provider(settings, args.provider)
    storage = get_storage(settings)

    tts = None
    want_audio = args.audio and args.type == "voice"
    if want_audio:
        tts_name = "scripted" if args.provider in ("scripted", "fake") else None
        tts = get_tts_provider(settings, tts_name)

    written: list[str] = []
    for _ in range(args.count):
        spec = random_spec(
            business=args.business,
            business_name=args.business_name,
            problem=args.problem,
            emotion=args.emotion,
            conversation_type=args.type,
        )
        result = generate_vcon(
            spec,
            provider=provider,
            storage=storage,
            generate_audio=want_audio,
            tts=tts,
            chat_model=settings.openai_model,
            tts_model=settings.openai_tts_model,
        )
        print(result.vcon_locator)
        written.append(result.vcon_locator)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vconfaker", description="Generate synthetic vCons.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate one or more synthetic vCons.")
    gen.add_argument("--count", type=int, default=1, help="How many to generate.")
    gen.add_argument("--type", choices=["voice", "messaging"], default="voice")
    gen.add_argument(
        "--provider",
        default=None,
        help="LLM provider: 'openai' or 'scripted' (default: env VCONFAKER_LLM_PROVIDER or 'openai').",
    )
    gen.add_argument("--storage", choices=["local", "s3"], default="local")
    gen.add_argument("--out-dir", default="./out", help="Local output dir (local storage).")
    gen.add_argument("--audio", action="store_true", help="Attach recording media (voice only).")
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
