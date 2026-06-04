"""End-to-end CLI tests: the engine produces valid, governed vCons headlessly."""
from __future__ import annotations

import json

import pytest
from vcon import Vcon

from vconfaker.cli import main


@pytest.mark.parametrize("conv_type", ["voice", "messaging"])
def test_cli_generate_writes_valid_vcons(tmp_path, conv_type):
    rc = main(
        [
            "generate",
            "--provider", "scripted",
            "--type", conv_type,
            "--count", "3",
            "--out-dir", str(tmp_path),
            "--business", "Insurance Agency",
            "--business-name", "Acme Insurance",
            "--problem", "a billing question",
        ]
    )
    assert rc == 0

    files = list(tmp_path.rglob("*.vcon.json"))
    assert len(files) == 3
    for path in files:
        d = json.loads(path.read_text())
        assert d["vcon"] == "0.4.0"
        ok, errors = Vcon(d).is_valid()
        assert ok, errors
        purposes = {a["purpose"] for a in d["attachments"]}
        assert {"tags", "lawful_basis"} <= purposes
        assert all(p.get("validation") == "synthetic" for p in d["parties"])


def test_cli_generate_with_emotion(tmp_path):
    rc = main(
        [
            "generate",
            "--provider", "scripted",
            "--type", "messaging",
            "--count", "1",
            "--out-dir", str(tmp_path),
            "--emotion", "frustrated",
        ]
    )
    assert rc == 0
    assert len(list(tmp_path.rglob("*.vcon.json"))) == 1
