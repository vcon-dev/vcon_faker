"""Command-line entry point for the vcon_faker engine.

Generate synthetic vCons headlessly, with no Streamlit and no AWS required:

    python -m vconfaker generate --provider scripted --type voice --count 5
    python -m vconfaker generate --provider openai --storage s3 --count 100

``--provider scripted`` produces deterministic offline output (no API key).
"""
from __future__ import annotations

import argparse
import random
import uuid
from datetime import UTC, datetime

from vconfaker.config import Settings
from vconfaker.generators.conversation import generate_conversation
from vconfaker.models import ConversationSpec
from vconfaker.providers.llm import get_provider
from vconfaker.storage import get_storage
from vconfaker.vconbuild.builder import build_vcon


def _random_identities(business: str) -> dict[str, str]:
    """Generate synthetic party identities. Uses the project's name lists."""
    from fake_names import female_names, last_names, male_names

    agent = f"{random.choice(male_names)} {random.choice(last_names)}"
    customer = f"{random.choice(female_names)} {random.choice(last_names)}"
    domain = (business.replace(" ", "").lower() or "acme")
    return {
        "agent_name": agent,
        "customer_name": customer,
        "agent_phone": f"+1{random.randint(2000000000, 9999999999)}",
        "customer_phone": f"+1{random.randint(2000000000, 9999999999)}",
        # RFC 2606 reserved TLD/domain so addresses are unmistakably synthetic.
        "agent_email": f"{agent.replace(' ', '.').lower()}@{domain}.example",
        "customer_email": f"{customer.replace(' ', '.').lower()}@example.com",
    }


def cmd_generate(args: argparse.Namespace) -> list[str]:
    settings = Settings.from_env(
        llm_provider=args.provider,
        storage_backend=args.storage,
        output_dir=args.out_dir,
    )
    provider = get_provider(settings, args.provider)
    storage = get_storage(settings)

    written: list[str] = []
    for _ in range(args.count):
        ident = _random_identities(args.business)
        spec = ConversationSpec(
            agent_name=ident["agent_name"],
            customer_name=ident["customer_name"],
            business=args.business,
            business_name=args.business_name,
            problem=args.problem,
            emotion=args.emotion,
            conversation_type=args.type,
        )
        turns = generate_conversation(spec, provider)

        conv_date = datetime.now(UTC)
        vcon_uuid = str(uuid.uuid4())
        vcon = build_vcon(
            agent_name=ident["agent_name"],
            customer_name=ident["customer_name"],
            agent_phone=ident["agent_phone"],
            customer_phone=ident["customer_phone"],
            agent_email=ident["agent_email"],
            customer_email=ident["customer_email"],
            business_name=args.business_name,
            business=args.business,
            problem=args.problem,
            emotion=args.emotion,
            conversation=turns,
            conversation_type=args.type,
            conversation_date=conv_date,
            chat_model=settings.openai_model,
        )
        key = f"{conv_date:%Y/%m/%d}/{vcon_uuid}.vcon.json"
        locator = storage.save(key, vcon.to_json().encode("utf-8"), "application/json")
        written.append(locator)
        print(locator)
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
