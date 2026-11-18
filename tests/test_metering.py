"""Unit tests for the TTS cost-metering shim.

The field marshalling, tenant parsing (a falsy or non-UUID tenant now RAISES
upstream rather than being coerced to a sentinel), canonical-stream resolution and
fire-and-forget contract now live in :mod:`w2v_common.metering` (tested there
once for the whole fleet). These tests pin the TTS-specific contract: the
routing slugs, that the shim re-exports the canonical helpers (not a local
re-implementation), and that the emitter produces the expected field shape via
the shim's exports.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
import w2v_common.metering as w2v_metering

from services.metering import (
    COST_SOURCE,
    SERVICE_KEY,
    close_metering_redis,
    emit_cost_event,
    init_metering_redis,
    resolve_stream,
    schedule_cost_event,
)


def test_routing_slugs_are_w2v_tts() -> None:
    assert COST_SOURCE == "w2v-tts"
    assert SERVICE_KEY == "w2v-tts"  # pragma: allowlist secret -- routing slug


def test_shim_reexports_are_w2v_common_helpers() -> None:
    """The shim must delegate to w2v_common, not re-implement marshalling."""
    assert emit_cost_event is w2v_metering.emit_cost_event
    assert schedule_cost_event is w2v_metering.schedule_cost_event
    assert init_metering_redis is w2v_metering.init_metering_redis
    assert close_metering_redis is w2v_metering.close_metering_redis


def test_shim_does_not_reexport_the_nil_sentinel() -> None:
    """Product-owner ruling: no sentinel may be reachable from this package.

    The shim used to re-export ``UNATTRIBUTED_TENANT_ID``. The nil UUID is a
    VALID uuid, so it passed every cast and billing bucketed real TTS usage as
    __unattributed__ and NEVER CHARGED IT. Nothing in this service may hand a
    sentinel to the cost bus.
    """
    import services.metering as shim

    assert not hasattr(shim, "UNATTRIBUTED_TENANT_ID")
    assert "UNATTRIBUTED_TENANT_ID" not in shim.__all__


def test_resolves_to_current_canonical_not_dead_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the cost-bus stream was renamed metering:costs ->
    improvement-loop:costs (2026-06-24) -> billing:costs. The resolved default
    must be the current canonical, never a dead name, and METERING_STREAM
    overrides it."""
    monkeypatch.delenv("METERING_STREAM", raising=False)
    assert resolve_stream() == "billing:costs"
    assert resolve_stream() not in ("metering:costs", "improvement-loop:costs")
    monkeypatch.setenv("METERING_STREAM", "custom:costs")
    assert resolve_stream() == "custom:costs"


@pytest.mark.asyncio
async def test_emit_cost_event_kokoro_shape() -> None:
    redis = AsyncMock()
    await emit_cost_event(
        redis,
        service_key=SERVICE_KEY,
        cost_source=COST_SOURCE,
        tenant_id="3f2504e0-4f89-11d3-9a0c-0305e82c3301",
        subject_id="user-1",
        provider="kokoro",
        model="/Users/mini2/.cache/tts/kokoro-v1.0.onnx",
        input_tokens=12,
        output_tokens=0,
        latency_ms=87,
        cost_usd=0.0,
        usage_extra={
            "unit": "characters",
            "duration_s": 1.23,
            "voice": "af_heart",
            "speed": 1.0,
            "backend": "kokoro",
        },
    )
    redis.xadd.assert_awaited_once()
    args = redis.xadd.await_args
    assert args.args[0] == "billing:costs"
    fields: dict[str, str] = args.args[1]
    # Every value is a string at the Redis layer.
    assert all(isinstance(v, str) for v in fields.values())
    assert fields["tenant_id"] == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    assert fields["subject_id"] == "user-1"
    assert fields["provider"] == "kokoro"
    assert fields["model"] == "/Users/mini2/.cache/tts/kokoro-v1.0.onnx"
    assert fields["input_tokens"] == "12"
    assert fields["output_tokens"] == "0"
    assert fields["latency_ms"] == "87"
    assert fields["cost_source"] == COST_SOURCE == "w2v-tts"
    assert fields["service_key"] == SERVICE_KEY == "w2v-tts"
    extra: dict[str, Any] = json.loads(fields["usage_extra"])
    assert extra["unit"] == "characters"
    assert extra["voice"] == "af_heart"
    assert extra["backend"] == "kokoro"
    # maxlen kwarg passed through
    assert args.kwargs["maxlen"] == 100_000


@pytest.mark.asyncio
async def test_emit_cost_event_kitten_shape() -> None:
    redis = AsyncMock()
    await emit_cost_event(
        redis,
        service_key=SERVICE_KEY,
        cost_source=COST_SOURCE,
        tenant_id="7b1c9d24-1a2b-4c3d-8e4f-5a6b7c8d9e0f",
        subject_id="user-2",
        provider="kitten",
        model="KittenML/kitten-tts-mini-0.8",
        input_tokens=42,
        output_tokens=0,
        latency_ms=200,
        cost_usd=0.0,
        usage_extra={
            "unit": "characters",
            "duration_s": 3.5,
            "voice": "expr-voice-2-f",
            "speed": 1.25,
            "backend": "kitten",
        },
    )
    redis.xadd.assert_awaited_once()
    fields = redis.xadd.await_args.args[1]
    assert fields["provider"] == "kitten"
    assert fields["model"] == "KittenML/kitten-tts-mini-0.8"
    assert json.loads(fields["usage_extra"])["backend"] == "kitten"


@pytest.mark.parametrize("bad_tenant", ["", "   "])
def test_a_blank_tenant_cannot_even_be_CONSTRUCTED(bad_tenant: str) -> None:
    """From 0.67.10 a blank tenant dies at the model, one layer earlier.

    ``User.tenant_id`` is ``uuid.UUID | None``, so pydantic refuses a blank
    string outright. These two cases used to reach ``require_tenant`` and be
    401ed there; they can no longer get that far, which is strictly stronger.
    """
    from pydantic import ValidationError

    from auth.models import User

    with pytest.raises(ValidationError):
        User(sub="", preferred_username="u", tenant_id=bad_tenant)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_tenant", [None, "00000000-0000-0000-0000-000000000000"])
async def test_no_cost_event_can_be_emitted_without_a_real_tenant(
    bad_tenant: str | None,
) -> None:
    """This test previously asserted the SUBSTITUTION and is now inverted.

    It used to pin "a falsy tenant_id must be emitted as the nil-UUID sentinel".
    Per the 2026-08-26 product-owner ruling that behaviour is the bug: the nil
    UUID is a VALID uuid, so it passes every cast, and billing then buckets the
    event as __unattributed__ and NEVER CHARGES IT. Real TTS usage, unbilled.

    As of w2v-common 0.67.10 the coercion is GONE from the shared fleet code
    too — ``schedule_cost_event`` parses the tenant synchronously and RAISES on
    a missing or non-UUID one. So the guarantee is now defended twice, and this
    test pins the near half: ``require_tenant`` refuses the request with 401
    before any route body runs, therefore no ``xadd`` ever happens for a
    tenant-less caller.
    """
    from fastapi import HTTPException

    from auth.dependencies import require_tenant
    from auth.models import User

    redis = AsyncMock()
    user = User(sub=bad_tenant or "", preferred_username="u", tenant_id=bad_tenant)

    with pytest.raises(HTTPException) as exc:
        require_tenant(user)
    assert exc.value.status_code == 401

    # The guard fired, so the route body — and with it schedule_cost_event —
    # never ran. Nothing reached the cost bus.
    redis.xadd.assert_not_awaited()


@pytest.mark.asyncio
async def test_emit_cost_event_swallows_redis_errors() -> None:
    redis = AsyncMock()
    redis.xadd.side_effect = RuntimeError("redis went away")
    # Must not raise; emit is fire-and-forget.
    await emit_cost_event(
        redis,
        service_key=SERVICE_KEY,
        cost_source=COST_SOURCE,
        tenant_id="3f2504e0-4f89-11d3-9a0c-0305e82c3301",
        subject_id="s",
        provider="kokoro",
        model="m",
        input_tokens=1,
        output_tokens=0,
        latency_ms=1,
    )


@pytest.mark.asyncio
async def test_schedule_cost_event_is_gc_safe_and_emits() -> None:
    """schedule_cost_event must fire-and-forget without being awaited and still
    reach xadd (the GC-safe replacement for asyncio.create_task(emit_cost_event)).
    """
    redis = AsyncMock()
    schedule_cost_event(
        redis,
        service_key=SERVICE_KEY,
        cost_source=COST_SOURCE,
        tenant_id="3f2504e0-4f89-11d3-9a0c-0305e82c3301",
        subject_id="user-1",
        provider="kokoro",
        model="m",
        input_tokens=7,
        output_tokens=0,
        latency_ms=5,
        cost_usd=0.0,
        usage_extra={"unit": "characters", "backend": "kokoro"},
    )
    # Yield control so the scheduled background task runs to completion.
    for _ in range(5):
        await asyncio.sleep(0)
        if redis.xadd.await_count:
            break
    redis.xadd.assert_awaited_once()
    fields = redis.xadd.await_args.args[1]
    assert fields["service_key"] == "w2v-tts"
    assert fields["input_tokens"] == "7"


@pytest.mark.asyncio
async def test_init_metering_redis_empty_url_returns_none() -> None:
    assert await init_metering_redis("") is None


@pytest.mark.asyncio
async def test_close_metering_redis_none_is_noop() -> None:
    await close_metering_redis(None)  # must not raise
