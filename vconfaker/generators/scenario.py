"""Scenario helpers: synthetic party identities and conversation timing."""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from vconfaker.models import CONVERSATION_TYPES, ConversationSpec

# Spec-plausible reasons for an incomplete (never-connected) conversation.
INCOMPLETE_DISPOSITIONS = ("no-answer", "busy", "voicemail-no-message", "hung-up")


def random_spec(
    *,
    business: str,
    business_name: str,
    problem: str,
    emotion: str | None = None,
    conversation_type: str = "voice",
    disposition: str | None = None,
    prompt: str | None = None,
) -> ConversationSpec:
    """Build a fully populated :class:`ConversationSpec` with synthetic identities.

    Names come from the project's ``fake_names`` lists; emails use the RFC 2606
    reserved ``.example`` domain so they are unmistakably synthetic.
    """
    from fake_names import female_names, last_names, male_names

    if conversation_type not in CONVERSATION_TYPES:
        raise ValueError(f"unknown conversation type: {conversation_type!r}")

    agent = f"{random.choice(male_names)} {random.choice(last_names)}"
    customer = f"{random.choice(female_names)} {random.choice(last_names)}"
    supervisor = f"{random.choice(female_names)} {random.choice(last_names)}"
    if conversation_type == "incomplete" and disposition is None:
        disposition = random.choice(INCOMPLETE_DISPOSITIONS)
    domain = business.replace(" ", "").lower() or "acme"
    return ConversationSpec(
        agent_name=agent,
        customer_name=customer,
        agent_phone=f"+1{random.randint(2000000000, 9999999999)}",
        customer_phone=f"+1{random.randint(2000000000, 9999999999)}",
        agent_email=f"{agent.replace(' ', '.').lower()}@{domain}.example",
        customer_email=f"{customer.replace(' ', '.').lower()}@example.com",
        business=business,
        business_name=business_name,
        problem=problem,
        emotion=emotion,
        conversation_type=conversation_type,
        supervisor_name=supervisor,
        disposition=disposition,
        prompt=prompt,
    )


def random_business_datetime(
    start: datetime,
    end: datetime,
    *,
    business_hours: tuple[int, int] = (9, 17),
    tz: str = "UTC",
) -> datetime:
    """Return a timezone-aware datetime within ``[start, end]`` and business hours."""
    zone = ZoneInfo(tz)
    if start.tzinfo is None:
        start = start.replace(tzinfo=zone)
    if end.tzinfo is None:
        end = end.replace(tzinfo=zone)
    span_days = max((end.date() - start.date()).days, 0)
    day = start + timedelta(days=random.randint(0, span_days))
    low, high = business_hours
    hour = random.randint(low, max(low, high - 1))
    return datetime(
        day.year, day.month, day.day, hour, random.randint(0, 59), random.randint(0, 59), tzinfo=zone
    )
