"""Backward-compatible shim.

The vCon builder moved to ``vconfaker.vconbuild.builder`` during the Phase 1
engine extraction. This module re-exports it so existing imports
(``from vcon_builder import build_vcon, sha512_base64url``) keep working
unchanged. ``sha512_base64url`` is now the library's ``compute_content_hash``
(vcon >= 0.9.5), which emits the same ``sha512-<base64url>`` form.
"""
from vcon import compute_content_hash as sha512_base64url  # noqa: F401

from vconfaker.vconbuild.builder import (  # noqa: F401
    LAWFUL_BASIS_TOS_URL,
    SYNTHETIC_VENDOR,
    WTF_SCHEMA_URL,
    Turn,
    build_vcon,
)

__all__ = [
    "build_vcon",
    "sha512_base64url",
    "WTF_SCHEMA_URL",
    "LAWFUL_BASIS_TOS_URL",
    "SYNTHETIC_VENDOR",
    "Turn",
]
