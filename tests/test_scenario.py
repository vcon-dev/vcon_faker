"""Tests for scenario helpers: synthetic identities and conversation timing."""
from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from vconfaker.generators.scenario import random_business_datetime, random_spec


def test_random_spec_fills_identities():
    spec = random_spec(
        business="Auto Repair Shop",
        business_name="Mike's Garage",
        problem="warranty claim",
        emotion="annoyed",
        conversation_type="voice",
    )
    assert spec.agent_name and spec.customer_name
    assert spec.agent_phone.startswith("+1") and spec.customer_phone.startswith("+1")
    assert spec.agent_email.endswith(".example")
    assert spec.customer_email.endswith("@example.com")
    assert spec.business == "Auto Repair Shop"
    assert spec.emotion == "annoyed"
    assert spec.conversation_type == "voice"


def test_random_business_datetime_within_range_and_hours():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 31, tzinfo=UTC)
    for _ in range(50):
        dt = random_business_datetime(start, end, business_hours=(9, 17), tz="US/Eastern")
        assert dt.tzinfo == ZoneInfo("US/Eastern")
        assert 9 <= dt.hour <= 16
        assert start.date() <= dt.date() <= end.date()
