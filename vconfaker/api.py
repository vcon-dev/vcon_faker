"""FastAPI surface over the engine.

Run with: ``uvicorn vconfaker.api:app`` (set OPENAI_API_KEY / S3_* in the env, or
use ``provider: "scripted"`` + local storage for offline demos).

Endpoints:
* ``GET  /healthz``  - liveness probe
* ``POST /generate`` - generate one or more vCons; returns their locators
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vconfaker import __version__
from vconfaker.config import Settings
from vconfaker.engine import generate_many
from vconfaker.generators.scenario import random_spec
from vconfaker.providers.llm import get_provider
from vconfaker.providers.stt import get_stt_provider
from vconfaker.providers.tts import get_tts_provider
from vconfaker.storage import get_storage


class GenerateRequest(BaseModel):
    business: str = "Insurance Agency"
    business_name: str = "Acme Insurance"
    problem: str = "a billing question"
    emotion: str | None = None
    conversation_type: str = "voice"
    count: int = 1
    concurrency: int = Field(default=1, ge=1, le=32)
    provider: str | None = None  # openai | anthropic | scripted
    audio: bool = False
    whisper: bool = False


class GeneratedItem(BaseModel):
    vcon_uuid: str
    vcon_locator: str
    audio_locator: str | None = None
    conversation_date: str
    summary: str


class GenerateResponse(BaseModel):
    items: list[GeneratedItem]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. Pass ``settings`` to pin config (e.g. in tests)."""
    app = FastAPI(title="vcon_faker", version=__version__)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        cfg = settings or Settings.from_env(
            llm_provider=req.provider,
            concurrency=req.concurrency,
            whisper_enabled=req.whisper,
        )
        try:
            provider = get_provider(cfg, req.provider)
            storage = get_storage(cfg)
            want_audio = req.audio and req.conversation_type == "voice"
            tts = None
            stt = None
            if want_audio:
                tts_name = "scripted" if req.provider in ("scripted", "fake") else None
                tts = get_tts_provider(cfg, tts_name)
                stt = get_stt_provider(cfg)

            specs = [
                random_spec(
                    business=req.business,
                    business_name=req.business_name,
                    problem=req.problem,
                    emotion=req.emotion,
                    conversation_type=req.conversation_type,
                )
                for _ in range(req.count)
            ]
            results = generate_many(
                specs,
                settings=cfg,
                provider=provider,
                storage=storage,
                generate_audio=want_audio,
                tts=tts,
                stt=stt,
                concurrency=req.concurrency,
            )
            items = [
                GeneratedItem(
                    vcon_uuid=r.vcon_uuid,
                    vcon_locator=r.vcon_locator,
                    audio_locator=r.audio_locator,
                    conversation_date=r.conversation_date,
                    summary=r.summary,
                )
                for r in results
            ]
            return GenerateResponse(items=items)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
