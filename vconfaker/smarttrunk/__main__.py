"""Command line: ``python -m vconfaker.smarttrunk --industry dealership --count 20 --out DIR``.

Run from the repo root (``fake_names`` and the core-05 schema are read from it), in a venv
built from ``requirements-smarttrunk.txt``. The Anthropic key comes from ``ANTHROPIC_API_KEY``,
or from an env file given with ``--anthropic-env-file``; it is put into this process's
environment only and never printed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .run import RunConfig, run


def _load_key(path: Path, var: str) -> None:
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.strip().removeprefix("export ").strip() == var:
            os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("'\"")
            return
    sys.exit(f"{var} is not set in {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vconfaker.smarttrunk", description=__doc__.splitlines()[0])
    ap.add_argument("--industry", default="dealership", help="scenario pack in vconfaker/smarttrunk/packs")
    ap.add_argument("--count", type=int, default=20, help="calls to make, in the pack's mix")
    ap.add_argument("--out", type=Path, required=True, help="output folder, kept outside any repo")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--run-label", default=None, help="dataset_version tag (default: the output folder's name)")
    ap.add_argument(
        "--media-base-url",
        default="https://media.invalid/smarttrunk-synthetic",
        help="where the MP3s will be served; the loader that publishes them sets the real one",
    )
    ap.add_argument("--parakeet", default=shutil.which("parakeet-mlx") or "parakeet-mlx")
    ap.add_argument("--usage-file", type=Path, default=None, help="running token record, shared by a corpus's runs")
    ap.add_argument("--max-cost-usd", type=float, default=None, help="stop writing scripts once the record reaches this")
    ap.add_argument("--min-free-mb", type=int, default=2048, help="pause while free memory is under this")
    ap.add_argument("--corpus-root", type=Path, default=None, help="also write an overall index.html here")
    ap.add_argument("--anthropic-env-file", type=Path, default=None)
    ap.add_argument("--anthropic-env-var", default="ANTHROPIC_API_KEY")
    args = ap.parse_args(argv)

    if args.anthropic_env_file:
        _load_key(args.anthropic_env_file.expanduser(), args.anthropic_env_var)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("note: ANTHROPIC_API_KEY is not set; only cached scripts can be used", file=sys.stderr)

    out = args.out.expanduser().resolve()
    manifest = run(
        RunConfig(
            industry=args.industry,
            out=out,
            count=args.count,
            seed=args.seed,
            media_base_url=args.media_base_url,
            parakeet=args.parakeet,
            run_label=args.run_label or out.name,
            usage_file=args.usage_file.expanduser().resolve() if args.usage_file else None,
            max_cost_usd=args.max_cost_usd,
            min_free_mb=args.min_free_mb,
            corpus_root=args.corpus_root.expanduser().resolve() if args.corpus_root else None,
        )
    )
    for key, value in manifest["stats"].items():
        print(f"{key}: {value}")
    print(f"index: {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
