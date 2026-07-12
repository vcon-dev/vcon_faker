"""Streamlit UI for vcon_faker - a thin client over the vconfaker engine.

Generation logic lives entirely in ``vconfaker.*`` (config, providers,
generators, storage, engine). This module only reads secrets, renders the form,
maps inputs to ``ConversationSpec``s, calls the engine, and shows the results.

Run: ``streamlit run vcon_faker.py``

Runs with zero credentials using the ``scripted`` provider and ``local``
storage; API keys and AWS/conserver settings are only needed for the backends
that use them.
"""
import json
import logging
import random
from datetime import datetime, timedelta
from datetime import time as dt_time

import streamlit as st

from fake_names import businesses, emotions, problems
from vconfaker.config import Settings
from vconfaker.engine import generate_many, load_signing_key
from vconfaker.generators.conversation import (
    DEFAULT_EMAIL_PROMPT,
    DEFAULT_MESSAGING_PROMPT,
    DEFAULT_TRANSFER_PROMPT,
    DEFAULT_VOICE_PROMPT,
)
from vconfaker.generators.scenario import random_business_datetime, random_spec
from vconfaker.models import CONVERSATION_TYPES
from vconfaker.providers.llm import get_provider
from vconfaker.providers.stt import get_stt_provider
from vconfaker.providers.tts import get_tts_provider
from vconfaker.storage import get_storage

logger = logging.getLogger(__name__)

TIMEZONES = ["UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "Europe/London", "Asia/Tokyo"]
FALLBACK_MODELS = ["gpt-4o", "gpt-4o-mini", "o3-mini"]

TYPE_LABELS = {
    "voice": "Voice Call",
    "messaging": "Text Messaging",
    "email": "Email Thread",
    "incomplete": "Missed Call",
    "transfer": "Transferred Call",
}
TYPE_PROMPTS = {
    "voice": DEFAULT_VOICE_PROMPT,
    "messaging": DEFAULT_MESSAGING_PROMPT,
    "email": DEFAULT_EMAIL_PROMPT,
    "transfer": DEFAULT_TRANSFER_PROMPT,
}


def _settings() -> Settings:
    """Engine Settings from the environment, overlaid with Streamlit secrets."""
    try:
        have_secrets = st.secrets.load_if_toml_exists()
    except Exception:
        have_secrets = False
    if not have_secrets:
        # No .streamlit/secrets.toml: env-only (fine for scripted + local).
        return Settings.from_env()
    secrets = st.secrets
    return Settings.from_env(
        openai_api_key=secrets.get("OPENAI_API_KEY"),
        openai_model=secrets.get("OPENAI_MODEL"),
        openai_tts_model=secrets.get("OPENAI_TTS_MODEL"),
        anthropic_api_key=secrets.get("ANTHROPIC_API_KEY"),
        anthropic_model=secrets.get("ANTHROPIC_MODEL"),
        aws_access_key=secrets.get("AWS_ACCESS_KEY"),
        aws_secret_key=secrets.get("AWS_SECRET_KEY"),
        s3_bucket=secrets.get("S3_BUCKET"),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _chat_models(api_key: str | None) -> list[str]:
    if not api_key:
        return FALLBACK_MODELS
    try:
        from openai import OpenAI

        ids = sorted(model.id for model in OpenAI(api_key=api_key).models.list().data)
        return ids or FALLBACK_MODELS
    except Exception as exc:
        logger.warning("Could not fetch OpenAI models: %s", exc)
        return FALLBACK_MODELS


def main() -> None:
    st.title("Fake Conversation Generator")
    base = _settings()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Scenario")
        business = st.selectbox("Business", ["Pick Random Business Type", *businesses])
        problem = st.selectbox("Problem", ["random situation", *problems])
        business_name = st.text_input("Business Name", "Acme Insurance")
        today = datetime.now().date()
        start_date = st.date_input("Start Date", today - timedelta(days=30))
        end_date = st.date_input("End Date", today)
        timezone = st.selectbox("Timezone", TIMEZONES, index=0)
        hcol1, hcol2 = st.columns(2)
        start_hour = hcol1.number_input("Start Hour", 0, 23, 9)
        end_hour = hcol2.number_input("End Hour", 0, 23, 17)
        add_emotion = st.checkbox("Add emotion")

    with col2:
        st.subheader("Generation")
        provider_name = st.selectbox(
            "LLM Provider",
            ["openai", "anthropic", "scripted"],
            format_func=lambda p: {"scripted": "scripted (offline, no API key)"}.get(p, p),
        )
        chat_model = None
        if provider_name == "openai":
            models = _chat_models(base.openai_api_key)
            model_index = models.index(base.openai_model) if base.openai_model in models else 0
            chat_model = st.selectbox("Chat Model", models, index=model_index)
        elif provider_name == "anthropic":
            chat_model = st.text_input("Claude Model", base.anthropic_model)

        conv_type = st.radio(
            "Type", list(CONVERSATION_TYPES), format_func=lambda t: TYPE_LABELS.get(t, t)
        )
        generate_audio = conv_type == "voice" and st.checkbox("Generate audio", value=False)
        use_whisper = generate_audio and st.checkbox(
            "Whisper round-trip (real recognizer confidence)", value=False
        )
        sign_vcons = st.checkbox("Sign vCons (JWS)", value=False)

        storage_backend = st.selectbox("Storage", ["local", "s3", "conserver"])
        output_dir = base.output_dir
        s3_bucket = base.s3_bucket or ""
        conserver_url = base.conserver_url or ""
        if storage_backend == "local":
            output_dir = st.text_input("Output Directory", base.output_dir)
        elif storage_backend == "s3":
            s3_bucket = st.text_input("S3 Bucket", s3_bucket)
        else:
            conserver_url = st.text_input("Conserver URL", conserver_url)

        count = int(st.number_input("How many", 1, 100, 1))
        concurrency = int(st.slider("Concurrency", 1, 16, 1)) if count > 1 else 1
        go = st.button("Generate Conversation(s)", type="primary")

    with st.sidebar:
        if conv_type == "incomplete":
            st.caption("Missed calls have no dialogue; no prompt needed.")
            prompt_text = None
        else:
            prompt_text = st.text_area(
                "Conversation Prompt (editable)", TYPE_PROMPTS[conv_type], height=320
            )

    if not go:
        return

    try:
        settings = _settings()
        settings.llm_provider = provider_name
        settings.storage_backend = storage_backend
        settings.output_dir = output_dir
        settings.s3_bucket = s3_bucket or None
        settings.conserver_url = conserver_url or None
        settings.whisper_enabled = use_whisper
        if chat_model and provider_name == "openai":
            settings.openai_model = chat_model
        if chat_model and provider_name == "anthropic":
            settings.anthropic_model = chat_model
        provider = get_provider(settings)
        storage = get_storage(settings)
        tts = get_tts_provider(settings, "scripted" if provider_name == "scripted" else None) if generate_audio else None
        stt = get_stt_provider(settings) if generate_audio else None
        sign_key = load_signing_key(settings.signing_key, settings.output_dir) if sign_vcons else None
    except ValueError as exc:
        st.error(str(exc))
        return

    start_dt = datetime.combine(start_date, dt_time.min)
    end_dt = datetime.combine(end_date, dt_time.max)
    specs = []
    dates = []
    for _ in range(count):
        biz = random.choice(businesses) if business == "Pick Random Business Type" else business
        prob = random.choice(problems) if problem == "random situation" else problem
        emotion = random.choice(emotions) if add_emotion else None
        specs.append(
            random_spec(
                business=biz,
                business_name=business_name,
                problem=prob,
                emotion=emotion,
                conversation_type=conv_type,
                prompt=prompt_text,
            )
        )
        dates.append(
            random_business_datetime(
                start_dt, end_dt, business_hours=(int(start_hour), int(end_hour)), tz=timezone
            )
        )

    progress = st.progress(0.0, text="Generating...")
    results = []
    # Chunk by concurrency so the progress bar still moves on big batches.
    for offset in range(0, count, max(concurrency, 1)):
        chunk = slice(offset, offset + max(concurrency, 1))
        try:
            results.extend(
                generate_many(
                    specs[chunk],
                    settings=settings,
                    provider=provider,
                    storage=storage,
                    generate_audio=generate_audio,
                    tts=tts,
                    stt=stt,
                    conversation_dates=dates[chunk],
                    concurrency=concurrency,
                    sign_key=sign_key,
                )
            )
        except Exception as exc:
            st.error(f"Batch starting at {offset + 1} failed: {exc}")
        progress.progress(min((offset + concurrency) / count, 1.0))
    progress.empty()

    if sign_key is not None and not settings.signing_key:
        st.info(f"Signing key pair written to {settings.output_dir}/vcon-faker-signing-key.pem")

    st.markdown(f"## Completed Conversations ({len(results)})")
    for result in results:
        with st.expander(f"{result.conversation_date} - {result.summary}"):
            st.markdown(f"Stored at: {result.vcon_locator}")
            if result.audio_locator:
                if result.audio_locator.startswith("file://"):
                    with open(result.audio_locator.removeprefix("file://"), "rb") as fh:
                        st.audio(fh.read(), format="audio/mpeg")
                else:
                    st.audio(result.audio_locator, format="audio/mpeg")
            st.download_button(
                "Download vCon JSON",
                data=json.dumps(result.vcon_dict, indent=2),
                file_name=f"{result.vcon_uuid}.vcon.json",
                mime="application/json",
                key=f"dl-{result.vcon_uuid}",
            )
            st.json(result.vcon_dict, expanded=False)


if __name__ == "__main__":
    main()
