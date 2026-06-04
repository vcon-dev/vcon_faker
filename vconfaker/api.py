"""FastAPI surface over the engine.

Run with: ``uvicorn vconfaker.api:app`` (set OPENAI_API_KEY / S3_* in the env, or
use ``provider: "scripted"`` + local storage for offline demos).

Endpoints:
* ``GET  /healthz``  - liveness probe
* ``POST /generate`` - generate one or more vCons; returns their locators
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vconfaker import __version__
from vconfaker.config import Settings
from vconfaker.engine import generate_vcon
from vconfaker.generators.scenario import random_spec
from vconfaker.providers.llm import get_provider
from vconfaker.providers.tts import get_tts_provider
from vconfaker.storage import get_storage


class GenerateRequest(BaseModel):
    business: str = "Insurance Agency"
    business_name: str = "Acme Insurance"
    problem: str = "a billing question"
    emotion: str | None = None
    conversation_type: str = "voice"
    count: int = 1
    provider: str | None = None  # "openai" | "scripted" (default: server config)
    audio: bool = False


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
        cfg = settings or Settings.from_env(llm_provider=req.provider)
        try:
            provider = get_provider(cfg, req.provider)
            storage = get_storage(cfg)
            want_audio = req.audio and req.conversation_type == "voice"
            tts = None
            if want_audio:
                tts_name = "scripted" if req.provider in ("scripted", "fake") else None
                tts = get_tts_provider(cfg, tts_name)

            items: list[GeneratedItem] = []
            for _ in range(req.count):
                spec = random_spec(
                    business=req.business,
                    business_name=req.business_name,
                    problem=req.problem,
                    emotion=req.emotion,
                    conversation_type=req.conversation_type,
                )
                result = generate_vcon(
                    spec,
                    provider=provider,
                    storage=storage,
                    generate_audio=want_audio,
                    tts=tts,
                    chat_model=cfg.openai_model,
                    tts_model=cfg.openai_tts_model,
                )
                items.append(
                    GeneratedItem(
                        vcon_uuid=result.vcon_uuid,
                        vcon_locator=result.vcon_locator,
                        audio_locator=result.audio_locator,
                        conversation_date=result.conversation_date,
                        summary=result.summary,
                    )
                )
            return GenerateResponse(items=items)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
