"""One-call orchestration shared by the CLI, the API, and the Streamlit UI.

``generate_vcon`` runs the full pipeline for a single conversation:
scenario spec -> conversation turns -> (optional audio) -> spec-compliant vCon
-> stored blob, returning a :class:`GenerationResult`. ``generate_many`` runs
the same pipeline concurrently for batch generation.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from vcon import Vcon, compute_content_hash

from vconfaker.config import Settings
from vconfaker.generators.conversation import generate_conversation
from vconfaker.models import ConversationSpec
from vconfaker.providers.llm import LLMProvider
from vconfaker.providers.stt import STTProvider
from vconfaker.providers.tts import TTSProvider
from vconfaker.storage import Storage
from vconfaker.vconbuild.builder import build_vcon

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    vcon_uuid: str
    vcon_locator: str
    conversation_date: str
    summary: str
    audio_locator: str | None = None
    vcon_dict: dict[str, Any] = field(default_factory=dict)


def generate_vcon(
    spec: ConversationSpec,
    *,
    provider: LLMProvider,
    storage: Storage,
    generate_audio: bool = False,
    tts: TTSProvider | None = None,
    stt: STTProvider | None = None,
    chat_model: str | None = None,
    tts_model: str | None = None,
    conversation_date: datetime | None = None,
) -> GenerationResult:
    """Generate one synthetic vCon and persist it via ``storage``."""
    conversation_date = conversation_date or datetime.now(UTC)
    turns = generate_conversation(spec, provider)

    # Library UUID8, generated up front so the audio/vCon storage keys match
    # the vCon's own uuid field.
    vcon_uuid = Vcon.uuid8_domain_name("vcon-faker.example")
    date_path = conversation_date.strftime("%Y/%m/%d")
    want_audio = generate_audio and spec.conversation_type == "voice"

    audio_url: str | None = None
    audio_filename: str | None = None
    audio_content_hash: str | None = None
    audio_duration = 0.0
    audio_locator: str | None = None
    wtf_document = None

    if want_audio:
        if tts is None:
            raise ValueError("generate_audio=True requires a tts provider")
        audio = tts.synthesize(turns)
        audio_filename = f"{vcon_uuid}.mp3"
        audio_content_hash = compute_content_hash(audio.data)
        audio_duration = audio.duration
        audio_locator = storage.save(f"{date_path}/{audio_filename}", audio.data, audio.mediatype)
        audio_url = audio_locator
        if stt is not None:
            wtf_document = stt.transcribe(
                audio,
                turns,
                created_at=conversation_date.isoformat(),
                tts_model=tts_model,
            )

    vcon = build_vcon(
        agent_name=spec.agent_name,
        customer_name=spec.customer_name,
        agent_phone=spec.agent_phone,
        customer_phone=spec.customer_phone,
        agent_email=spec.agent_email,
        customer_email=spec.customer_email,
        business_name=spec.business_name,
        business=spec.business,
        problem=spec.problem,
        emotion=spec.emotion,
        conversation=turns,
        conversation_type=spec.conversation_type,
        conversation_date=conversation_date,
        generate_audio=want_audio,
        audio_url=audio_url,
        audio_filename=audio_filename,
        audio_content_hash=audio_content_hash,
        audio_duration=audio_duration,
        chat_model=chat_model,
        tts_model=tts_model,
        wtf_document=wtf_document,
        uuid=vcon_uuid,
    )
    vcon_locator = storage.save(
        f"{date_path}/{vcon_uuid}.vcon.json",
        vcon.to_json().encode("utf-8"),
        "application/json",
    )
    summary = (
        f"{spec.agent_name} and {spec.customer_name} - {spec.business_name} "
        f"({spec.business}): {spec.problem}"
    )
    logger.info("generated vCon %s (%s)", vcon_uuid, spec.conversation_type)
    return GenerationResult(
        vcon_uuid=vcon_uuid,
        vcon_locator=vcon_locator,
        conversation_date=conversation_date.isoformat(),
        summary=summary,
        audio_locator=audio_locator,
        vcon_dict=vcon.to_dict(),
    )


def generate_many(
    specs: list[ConversationSpec],
    *,
    settings: Settings,
    provider: LLMProvider,
    storage: Storage,
    generate_audio: bool = False,
    tts: TTSProvider | None = None,
    stt: STTProvider | None = None,
    conversation_dates: list[datetime] | None = None,
    concurrency: int | None = None,
) -> list[GenerationResult]:
    """Generate multiple vCons, optionally in parallel."""
    if not specs:
        return []

    workers = max(1, concurrency if concurrency is not None else settings.concurrency)
    chat_model = settings.chat_model()
    tts_model = settings.openai_tts_model

    def _one(index: int, spec: ConversationSpec) -> GenerationResult:
        conv_date = None
        if conversation_dates and index < len(conversation_dates):
            conv_date = conversation_dates[index]
        return generate_vcon(
            spec,
            provider=provider,
            storage=storage,
            generate_audio=generate_audio,
            tts=tts,
            stt=stt,
            chat_model=chat_model,
            tts_model=tts_model,
            conversation_date=conv_date,
        )

    if workers <= 1 or len(specs) == 1:
        return [_one(i, spec) for i, spec in enumerate(specs)]

    results: list[GenerationResult | None] = [None] * len(specs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, i, spec): i for i, spec in enumerate(specs)}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    return [r for r in results if r is not None]
