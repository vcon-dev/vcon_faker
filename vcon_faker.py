"""Streamlit UI for vcon_faker - a thin client over the vconfaker engine.

Generation logic lives entirely in ``vconfaker.*`` (config, providers,
generators, storage, engine). This module only reads secrets, renders the form,
maps inputs to a ``ConversationSpec``, calls ``engine.generate_vcon`` per
conversation, and shows the results.

Run: ``streamlit run vcon_faker.py``
"""
import logging
import random
from datetime import datetime, time as dt_time, timedelta

import streamlit as st

from fake_names import businesses, emotions, problems
from vconfaker.config import Settings
from vconfaker.engine import generate_vcon
from vconfaker.generators.conversation import DEFAULT_MESSAGING_PROMPT, DEFAULT_VOICE_PROMPT
from vconfaker.generators.scenario import random_business_datetime, random_spec
from vconfaker.providers.llm import get_provider
from vconfaker.providers.stt import get_stt_provider
from vconfaker.providers.tts import get_tts_provider
from vconfaker.storage import get_storage

logger = logging.getLogger(__name__)

TIMEZONES = ["UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "Europe/London", "Asia/Tokyo"]
FALLBACK_MODELS = ["gpt-4o", "gpt-4o-mini", "o3-mini"]


def _settings(bucket: str | None = None) -> Settings:
    """Build engine Settings from Streamlit secrets (read at call time, not import)."""
    secrets = st.secrets
    return Settings.from_env(
        llm_provider="openai",
        storage_backend="s3",
        openai_api_key=secrets.get("OPENAI_API_KEY"),
        openai_model=secrets.get("OPENAI_MODEL", "gpt-4o"),
        openai_tts_model=secrets.get("OPENAI_TTS_MODEL", "tts-1"),
        aws_access_key=secrets.get("AWS_ACCESS_KEY"),
        aws_secret_key=secrets.get("AWS_SECRET_KEY"),
        s3_bucket=bucket or secrets.get("S3_BUCKET"),
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
        business = st.selectbox("Business", ["Pick Random Business Type", *businesses])
        problem = st.selectbox("Problem", ["random situation", *problems])
        business_name = st.text_input("Business Name", "Acme Insurance")
        s3_bucket = st.text_input("S3 Bucket", base.s3_bucket or "")
        today = datetime.now().date()
        start_date = st.date_input("Start Date", today - timedelta(days=30))
        end_date = st.date_input("End Date", today)
        timezone = st.selectbox("Timezone", TIMEZONES, index=0)
        hcol1, hcol2 = st.columns(2)
        start_hour = hcol1.number_input("Start Hour", 0, 23, 9)
        end_hour = hcol2.number_input("End Hour", 0, 23, 17)

    with col2:
        models = _chat_models(base.openai_api_key)
        model_index = models.index(base.openai_model) if base.openai_model in models else 0
        chat_model = st.selectbox("Chat Model", models, index=model_index)
        conv_type = st.radio(
            "Type",
            ["messaging", "voice"],
            format_func=lambda x: "Text Messaging" if x == "messaging" else "Voice Call",
        )
        generate_audio = conv_type == "voice" and st.checkbox("Generate audio", value=False)
        add_emotion = st.checkbox("Add emotion")
        count = int(st.number_input("How many", 1, 100, 1))
        go = st.button("Generate Conversation(s)")

    with st.sidebar:
        default_prompt = DEFAULT_MESSAGING_PROMPT if conv_type == "messaging" else DEFAULT_VOICE_PROMPT
        prompt_text = st.text_area("Conversation Prompt (editable)", default_prompt, height=320)

    if not go:
        return

    try:
        settings = _settings(bucket=s3_bucket)
        settings.openai_model = chat_model
        provider = get_provider(settings)
        storage = get_storage(settings)
        tts = get_tts_provider(settings) if generate_audio else None
        stt = get_stt_provider(settings) if generate_audio else None
    except ValueError as exc:
        st.error(str(exc))
        return

    start_dt = datetime.combine(start_date, dt_time.min)
    end_dt = datetime.combine(end_date, dt_time.max)
    progress = st.progress(0.0, text="Generating...")
    results = []
    for i in range(count):
        biz = random.choice(businesses) if business == "Pick Random Business Type" else business
        prob = random.choice(problems) if problem == "random situation" else problem
        emotion = random.choice(emotions) if add_emotion else None
        spec = random_spec(
            business=biz,
            business_name=business_name,
            problem=prob,
            emotion=emotion,
            conversation_type=conv_type,
            prompt=prompt_text,
        )
        conv_date = random_business_datetime(
            start_dt, end_dt, business_hours=(int(start_hour), int(end_hour)), tz=timezone
        )
        try:
            results.append(
                generate_vcon(
                    spec,
                    provider=provider,
                    storage=storage,
                    generate_audio=generate_audio,
                    tts=tts,
                    stt=stt,
                    chat_model=settings.chat_model(),
                    tts_model=settings.openai_tts_model,
                    conversation_date=conv_date,
                )
            )
        except Exception as exc:
            st.error(f"Conversation {i + 1} failed: {exc}")
        progress.progress((i + 1) / count)
    progress.empty()

    st.markdown("## Completed Conversations")
    for result in results:
        links = f"[vCon]({result.vcon_locator})"
        if result.audio_locator:
            links += f" - [audio]({result.audio_locator})"
        st.markdown(f"**{result.conversation_date}** - {result.summary}")
        st.markdown(links)
        st.markdown("---")


if __name__ == "__main__":
    main()
