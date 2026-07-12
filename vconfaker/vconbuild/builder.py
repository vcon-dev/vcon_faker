"""Spec-compliant vCon assembly for vcon_faker.

This is a pure module: it imports only the official ``vcon`` library and the
standard library, so it can be unit-tested without Streamlit, OpenAI, or AWS.

It conforms to the vcon-speckit (IETF ``draft-ietf-vcon-vcon-core-02``, syntax
``"0.4.0"``) and is library-first throughout: ``Vcon.build_new``, ``add_party``,
``add_dialog``, ``add_analysis``, ``add_extension``.

Synthetic-corpus governance follows the canonical pattern used across the vCon
ecosystem (see ``CONSENT.md``):

* every party is marked ``validation: "synthetic"``;
* a ``tags`` attachment carries ``synthetic: "true"`` alongside scenario metadata;
* a ``lawful_basis`` attachment (``purpose: "lawful_basis"``,
  ``legitimate_interests``) documents the synthetic origin and points at
  ``CONSENT.md`` via ``terms_of_service``, and ``"lawful_basis"`` is added to the
  top-level ``extensions`` array.

Provenance is honest: vcon_faker authors conversation text with an LLM and (for
voice) synthesizes audio with TTS. No real ASR vendor is named, and the
transcript is the exact source text (confidence ``1.0``) rather than a fabricated
recognizer output. A real speech-to-text round-trip is planned for a later phase.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any

from vcon import Vcon
from vcon.dialog import Dialog
from vcon.party import Party

logger = logging.getLogger(__name__)

# Canonical extension / consent references.
WTF_SCHEMA_URL = "https://datatracker.ietf.org/doc/draft-howe-vcon-wtf-extension/"
LAWFUL_BASIS_TOS_URL = "https://github.com/vcon-dev/vcon_faker/blob/main/CONSENT.md"

# Honest provenance label for synthetic output (not a real ASR/analysis vendor).
SYNTHETIC_VENDOR = "vcon-faker"

# vCon syntax version this module emits (matches IETF draft-ietf-vcon-vcon-core-02).
VCON_SYNTAX_VERSION = "0.4.0"

Turn = dict[str, str]


def sha512_base64url(data: bytes) -> str:
    """Return a vCon-spec ``content_hash``: ``sha512-<base64url, no padding>``.

    This is the format required for externally referenced media by
    ``draft-ietf-vcon-vcon-core-02`` and the speckit glossary. (The Python
    library's internal helpers still compute an unprefixed SHA-256 digest; the
    spec and the vcon-js validator require this SHA-512 prefixed form.)
    """
    digest = hashlib.sha512(data).digest()
    return "sha512-" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _build_wtf_document(
    turns: list[Turn],
    duration: float,
    *,
    provider: str,
    model: str,
    created_at: str,
    language: str = "en",
) -> dict[str, Any]:
    """Build a World Transcription Format (WTF) document from known turns.

    vcon_faker synthesizes audio from authored text, so the transcript is exact
    (``confidence`` 1.0). Segment timings are apportioned across the audio
    duration in proportion to each message's length, which keeps segments
    non-overlapping and monotonically increasing as the WTF schema requires.
    """
    full_text = "\n".join(t["message"] for t in turns)
    total_chars = sum(len(t["message"]) for t in turns) or 1

    segments: list[dict[str, Any]] = []
    cursor = 0.0
    for idx, turn in enumerate(turns):
        seg_dur = round(duration * (len(turn["message"]) / total_chars), 3) if duration else 0.0
        start = round(cursor, 3)
        end = round(cursor + seg_dur, 3)
        segments.append(
            {
                "id": idx,
                "start": start,
                "end": end,
                "text": turn["message"],
                "confidence": 1.0,
                "speaker": turn.get("speaker", ""),
            }
        )
        cursor = end

    return {
        "transcript": {
            "text": full_text,
            "language": language,
            "duration": round(duration, 3),
            "confidence": 1.0,
        },
        "segments": segments,
        "metadata": {
            "created_at": created_at,
            "processed_at": created_at,
            "provider": provider,
            "model": model,
            "note": (
                "Synthetic: audio synthesized via TTS from LLM-authored text; "
                "transcript is the exact source text, not recognizer output."
            ),
        },
    }


def _add_synthetic_governance(v: Vcon, *, now: str, tags: dict[str, Any]) -> None:
    """Attach the canonical synthetic-corpus markers: ``tags`` + ``lawful_basis``.

    Both are vCon-level attachments (``party: 0``, ``dialog: 0``) with
    JSON-string bodies and ``encoding: "json"`` per the speckit non-negotiables.
    """
    tag_body = dict(tags)
    tag_body["synthetic"] = "true"

    attachments = v.vcon_dict.setdefault("attachments", [])
    attachments.append(
        {
            "purpose": "tags",
            "party": 0,
            "dialog": 0,
            "encoding": "json",
            "body": json.dumps(tag_body),
            "start": now,
        }
    )

    lawful_basis_body = {
        "lawful_basis": "legitimate_interests",
        "expiration": None,
        "purpose_grants": [
            {"purpose": "recording", "granted": True, "granted_at": now},
            {"purpose": "transcription", "granted": True, "granted_at": now},
            {"purpose": "analysis", "granted": True, "granted_at": now},
            {
                "purpose": "redistribution",
                "granted": True,
                "granted_at": now,
                "conditions": ["MIT license", "synthetic-data marker preserved"],
            },
        ],
        "proof_mechanisms": [
            {
                "proof_type": "external_system",
                "timestamp": now,
                "proof_data": {
                    "system": "vcon-faker-synthetic-pipeline",
                    "note": (
                        "Conversation authored by an LLM; any audio synthesized via "
                        "text-to-speech. No real person is represented."
                    ),
                },
            }
        ],
        "terms_of_service": LAWFUL_BASIS_TOS_URL,
        "metadata": {"synthetic": True, "license": "MIT"},
    }
    attachments.append(
        {
            "purpose": "lawful_basis",
            "party": 0,
            "dialog": 0,
            "encoding": "json",
            "body": json.dumps(lawful_basis_body),
            "start": now,
        }
    )

    if "lawful_basis" not in (v.get_extensions() or []):
        v.add_extension("lawful_basis")


def build_vcon(
    *,
    agent_name: str,
    customer_name: str,
    agent_phone: str,
    customer_phone: str,
    agent_email: str,
    customer_email: str,
    business_name: str,
    business: str,
    problem: str,
    conversation: list[Turn],
    emotion: str | None = None,
    conversation_type: str = "voice",
    conversation_date: datetime | None = None,
    generate_audio: bool = False,
    audio_url: str | None = None,
    audio_filename: str | None = None,
    audio_content_hash: str | None = None,
    audio_duration: float = 0.0,
    chat_model: str | None = None,
    tts_model: str | None = None,
    wtf_document: dict[str, Any] | None = None,
    validate: bool = True,
) -> Vcon:
    """Assemble a spec-compliant, synthetic-marked vCon and return it.

    Args:
        agent_name/customer_name: Display names of the two parties.
        *_phone/*_email: Party identifiers.
        business_name/business/problem/emotion: Scenario metadata (also tagged).
        conversation: List of ``{"speaker", "message"}`` turns.
        conversation_type: ``"voice"`` or ``"messaging"``.
        conversation_date: Timezone-aware start time of the conversation.
        generate_audio: Whether external recording media is attached (voice only).
        audio_url/audio_filename/audio_content_hash/audio_duration: Media refs.
            ``audio_content_hash`` must already be in ``sha512-<base64url>`` form
            (see ``sha512_base64url``).
        chat_model/tts_model: Recorded for provenance in analysis/WTF metadata.
        wtf_document: Optional pre-built WTF payload (``transcript``, ``segments``,
            ``metadata``) from an STT round-trip; defaults to source-text WTF.
        validate: Run ``Vcon.is_valid`` and raise ``ValueError`` on failure.

    Returns:
        A ``vcon.Vcon`` instance.
    """
    if conversation_date is None:
        conversation_date = datetime.now()

    turns: list[Turn] = [
        t for t in conversation if isinstance(t, dict) and t.get("message")
    ]
    if not turns:
        raise ValueError("conversation has no usable turns")

    start_iso = conversation_date.isoformat()
    v = Vcon.build_new()
    v.vcon_dict["subject"] = f"{problem} ({business_name})"

    last_dialog_time = conversation_date

    if conversation_type == "messaging":
        # Agent index 0, customer index 1.
        v.add_party(
            Party(
                id=agent_email,
                name=agent_name,
                tel=agent_phone,
                mailto=agent_email,
                role="agent",
                validation="synthetic",
            )
        )
        v.add_party(
            Party(
                id=customer_phone,
                name=customer_name,
                tel=customer_phone,
                mailto=customer_email,
                role="contact",
                validation="synthetic",
            )
        )

        msg_time = conversation_date
        for i, turn in enumerate(turns):
            party_index = 1 if turn.get("speaker") == "Customer" else 0
            if i > 0:
                msg_time += timedelta(
                    minutes=2 + (i % 4), seconds=(i * 7) % 60
                )
            v.add_dialog(
                Dialog(
                    type="text",
                    start=msg_time.isoformat(),
                    parties=[party_index],
                    originator=party_index,
                    mediatype="text/plain",
                    body=turn["message"],
                    encoding="none",
                )
            )
            last_dialog_time = msg_time
    else:
        # Voice. Customer index 0, agent index 1 (preserves prior convention).
        v.add_party(
            Party(
                id=f"{customer_phone}_{customer_email}_1100",
                name=customer_name,
                tel=customer_phone,
                mailto=customer_email,
                role="customer",
                validation="synthetic",
            )
        )
        v.add_party(
            Party(
                id=agent_email,
                name=agent_name,
                tel=agent_phone,
                mailto=agent_email,
                role="agent",
                validation="synthetic",
            )
        )

        transcript_text = "\n".join(t["message"] for t in turns)

        if generate_audio:
            v.add_dialog(
                Dialog(
                    type="recording",
                    start=start_iso,
                    parties=[1, 0],
                    originator=1,
                    mediatype="audio/mpeg",
                    filename=audio_filename,
                    url=audio_url,
                    content_hash=audio_content_hash,
                    duration=audio_duration,
                    meta={"disposition": "ANSWERED", "direction": "out"},
                )
            )
            # Transcript carried as a WTF document in analysis[].
            wtf = wtf_document or _build_wtf_document(
                turns,
                audio_duration,
                provider=SYNTHETIC_VENDOR,
                model=tts_model or "unknown",
                created_at=start_iso,
            )
            v.add_wtf_transcription_analysis(
                transcript=wtf["transcript"],
                segments=wtf["segments"],
                metadata=wtf["metadata"],
                dialog_index=0,
            )
        else:
            # No audio: keep the conversation inline as a single text dialog.
            v.add_dialog(
                Dialog(
                    type="text",
                    start=start_iso,
                    parties=[1, 0],
                    originator=1,
                    mediatype="text/plain",
                    body=transcript_text,
                    encoding="none",
                )
            )

    # Deterministic, templated summary (vendor reflects the generator, not an LLM).
    emotion_clause = f" The customer is {emotion}." if emotion else ""
    summary_text = (
        f"{agent_name} from {business_name} (a {business}) assists {customer_name} "
        f"regarding {problem}.{emotion_clause}"
    )
    v.add_analysis(
        type="summary",
        dialog=0,
        vendor=SYNTHETIC_VENDOR,
        product=chat_model or "unknown",
        body=summary_text,
        encoding="none",
    )

    # Synthetic-corpus governance: tags + lawful_basis.
    _add_synthetic_governance(
        v,
        now=start_iso,
        tags={
            "business": business,
            "business_name": business_name,
            "problem": problem,
            "emotion": emotion or "",
            "conversation_type": conversation_type,
        },
    )

    # created_at: 15s after the last dialog (matches prior behavior).
    v.vcon_dict["created_at"] = (last_dialog_time + timedelta(seconds=15)).isoformat()

    # Defensive post-processing the library mostly does for us in >=0.9.2.
    v.vcon_dict["vcon"] = VCON_SYNTAX_VERSION
    if v.vcon_dict.get("group") == []:
        v.vcon_dict.pop("group", None)
    if v.vcon_dict.get("redacted") == {}:
        v.vcon_dict.pop("redacted", None)

    if validate:
        is_valid, errors = v.is_valid()
        if not is_valid:
            logger.error("vCon validation failed: %s", errors)
            raise ValueError(f"Invalid vCon object: {errors}")

    return v


def build_wtf_from_turns(
    turns: list[Turn],
    duration: float,
    *,
    provider: str,
    model: str,
    created_at: str,
    language: str = "en",
) -> dict[str, Any]:
    """Build a WTF document from authored turns (public helper for STT providers)."""
    return _build_wtf_document(
        turns, duration, provider=provider, model=model, created_at=created_at, language=language
    )
