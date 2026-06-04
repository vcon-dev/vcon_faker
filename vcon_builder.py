"""Backward-compatible shim.

The vCon builder moved to ``vconfaker.vconbuild.builder`` during the Phase 1
engine extraction. This module re-exports it so existing imports
(``from vcon_builder import build_vcon, sha512_base64url``) and the Streamlit app
keep working unchanged.
"""
from vconfaker.vconbuild.builder import (  # noqa: F401
    LAWFUL_BASIS_TOS_URL,
    SYNTHETIC_VENDOR,
    VCON_SYNTAX_VERSION,
    WTF_SCHEMA_URL,
    Turn,
    build_vcon,
    sha512_base64url,
)

__all__ = [
    "build_vcon",
    "sha512_base64url",
    "WTF_SCHEMA_URL",
    "LAWFUL_BASIS_TOS_URL",
    "SYNTHETIC_VENDOR",
    "VCON_SYNTAX_VERSION",
    "Turn",
]
