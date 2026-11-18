"""G5: every tts cost event carries a declared ``funding`` and a deterministic ``event_key``.

Port of tts_service #133 onto the LIVE narrate path. Route tests drive the REAL
``/api/tts/synthesize`` and capture what the emitter was handed; one test runs
the real ``w2v_common`` emitter into a mock Redis, because the XADD fields are
what billing actually reads.
"""

from __future__ import annotations

import asyncio
import shutil
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
from w2v_common.metering import cost_event_key

from services.metering import (
    EVENT_KIND_IDEMPOTENCY,
    EVENT_KIND_REQUEST,
    SERVICE_KEY,
    UnknownFundingProviderError,
    funding_for_provider,
    synthesis_event_key,
)
from services.tts_service import SynthesisResult
from tests.conftest import TEST_TENANT_ID

_SR = 24000
_OTHER_TENANT = "22222222-2222-4222-8222-222222222222"

_needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


# ── unit: funding ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("provider", ["kokoro", "kitten"])
def test_local_engines_are_local_gpu(provider: str) -> None:
    assert funding_for_provider(provider) == "local_gpu"


@pytest.mark.parametrize("provider", ["openai", "elevenlabs", "stub", "", "Kokoro"])
def test_undeclared_provider_raises_instead_of_guessing(provider: str) -> None:
    with pytest.raises(UnknownFundingProviderError):
        funding_for_provider(provider)


# ── unit: event_key ──────────────────────────────────────────────────────────


def test_same_request_id_yields_the_same_key() -> None:
    a = synthesis_event_key(tenant_id=TEST_TENANT_ID, idempotency_key=None, request_id="req-1")
    b = synthesis_event_key(tenant_id=TEST_TENANT_ID, idempotency_key=None, request_id="req-1")
    assert a == b == cost_event_key(SERVICE_KEY, EVENT_KIND_REQUEST, TEST_TENANT_ID, "req-1")


def test_distinct_requests_yield_distinct_keys() -> None:
    a = synthesis_event_key(tenant_id=TEST_TENANT_ID, idempotency_key=None, request_id="req-1")
    b = synthesis_event_key(tenant_id=TEST_TENANT_ID, idempotency_key=None, request_id="req-2")
    assert a != b


def test_idempotency_key_wins_over_request_id() -> None:
    a = synthesis_event_key(tenant_id=TEST_TENANT_ID, idempotency_key="idem-1", request_id="req-1")
    b = synthesis_event_key(tenant_id=TEST_TENANT_ID, idempotency_key="idem-1", request_id="req-2")
    assert a == b == cost_event_key(SERVICE_KEY, EVENT_KIND_IDEMPOTENCY, TEST_TENANT_ID, "idem-1")


def test_idempotency_and_request_id_spaces_never_collide() -> None:
    via_idem = synthesis_event_key(tenant_id=TEST_TENANT_ID, idempotency_key="x", request_id=None)
    via_req = synthesis_event_key(tenant_id=TEST_TENANT_ID, idempotency_key=None, request_id="x")
    assert via_idem != via_req


def test_tenant_is_part_of_the_key() -> None:
    """Billing dedupes globally; caller-chosen ids must not collapse across tenants."""
    a = synthesis_event_key(tenant_id=TEST_TENANT_ID, idempotency_key="1", request_id=None)
    b = synthesis_event_key(tenant_id=_OTHER_TENANT, idempotency_key="1", request_id=None)
    assert a != b


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_no_id_at_all_defers_to_the_cost_bus(blank: str | None) -> None:
    assert synthesis_event_key(tenant_id=TEST_TENANT_ID, idempotency_key=blank, request_id=blank) is None


def test_blank_idempotency_key_falls_back_to_request_id() -> None:
    k = synthesis_event_key(tenant_id=TEST_TENANT_ID, idempotency_key="  ", request_id="req-1")
    assert k == cost_event_key(SERVICE_KEY, EVENT_KIND_REQUEST, TEST_TENANT_ID, "req-1")


# ── route: what /synthesize hands the cost bus ───────────────────────────────


class _FakeService:
    def __init__(self, provider: str = "kokoro") -> None:
        self._provider = provider

    def loaded_models(self) -> list[str]:
        return [self._provider]

    def synthesize(
        self,
        text: str,
        provider: str | None,
        voice: str | None,
        speed: float,
        ssml: bool | None = None,
    ) -> SynthesisResult:
        t = np.linspace(0.0, 1.0, _SR, endpoint=False)
        pcm = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        return SynthesisResult(
            audio_pcm=pcm,
            sample_rate=_SR,
            duration_sec=1.0,
            provider=self._provider,
            voice=voice or "af_heart",
            model="fake",
        )


@pytest.fixture
def captured_emit(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _capture(_redis: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    import routers.tts_routes as tts_routes

    monkeypatch.setattr(tts_routes, "schedule_cost_event", _capture)
    return calls


@pytest.fixture
def stub_state(authenticated_app: Any) -> Any:
    """Swap tts_service/metering_redis on the shared app and RESTORE them after."""
    state = authenticated_app.state
    sentinel = object()
    saved = {n: getattr(state, n, sentinel) for n in ("tts_service", "metering_redis")}

    def _install(provider: str = "kokoro", redis: Any = None) -> None:
        state.tts_service = _FakeService(provider)
        state.metering_redis = redis if redis is not None else object()

    try:
        yield _install
    finally:
        for name, prev in saved.items():
            if prev is sentinel:
                if hasattr(state, name):
                    delattr(state, name)
            else:
                setattr(state, name, prev)


async def _post(client: Any, headers: dict[str, str] | None = None) -> Any:
    return await client.post("/api/tts/synthesize", json={"text": "hello", "voice": "af_heart"}, headers=headers or {})


@_needs_ffmpeg
@pytest.mark.asyncio
async def test_route_emits_funding_and_request_keyed_event_key(
    client: Any, stub_state: Any, captured_emit: list[dict[str, Any]]
) -> None:
    stub_state()
    resp = await _post(client, {"X-Request-ID": "gw-req-42"})
    assert resp.status_code == 200, resp.text

    assert captured_emit, "the route emitted no cost event — the test proves nothing"
    kwargs = captured_emit[0]
    assert kwargs["funding"] == "local_gpu"
    assert kwargs["event_key"] == cost_event_key(SERVICE_KEY, EVENT_KIND_REQUEST, TEST_TENANT_ID, "gw-req-42")


@_needs_ffmpeg
@pytest.mark.asyncio
async def test_route_retry_with_same_request_id_reuses_the_key(
    client: Any, stub_state: Any, captured_emit: list[dict[str, Any]]
) -> None:
    stub_state()
    for _ in range(2):
        assert (await _post(client, {"X-Request-ID": "gw-req-7"})).status_code == 200
    assert (await _post(client, {"X-Request-ID": "gw-req-8"})).status_code == 200
    keys = [c["event_key"] for c in captured_emit]
    assert keys[0] == keys[1]
    assert keys[2] != keys[0]


@_needs_ffmpeg
@pytest.mark.asyncio
async def test_route_without_headers_keys_on_the_minted_request_id(
    client: Any, stub_state: Any, captured_emit: list[dict[str, Any]]
) -> None:
    """No caller id: the key is derived from the uuid7 the middleware minted and echoed."""
    stub_state()
    resp = await _post(client)
    assert resp.status_code == 200, resp.text
    minted = resp.headers["X-Request-ID"]
    assert captured_emit[0]["event_key"] == cost_event_key(SERVICE_KEY, EVENT_KIND_REQUEST, TEST_TENANT_ID, minted)


@_needs_ffmpeg
@pytest.mark.asyncio
async def test_route_idempotency_key_wins(client: Any, stub_state: Any, captured_emit: list[dict[str, Any]]) -> None:
    stub_state()
    await _post(client, {"Idempotency-Key": "narrate-1", "X-Request-ID": "a"})
    await _post(client, {"Idempotency-Key": "narrate-1", "X-Request-ID": "b"})
    k0, k1 = (c["event_key"] for c in captured_emit)
    assert k0 == k1 == cost_event_key(SERVICE_KEY, EVENT_KIND_IDEMPOTENCY, TEST_TENANT_ID, "narrate-1")


@_needs_ffmpeg
@pytest.mark.asyncio
async def test_route_undeclared_provider_fails_loud_and_emits_nothing(
    authenticated_app: Any, stub_state: Any, captured_emit: list[dict[str, Any]]
) -> None:
    import httpx

    stub_state(provider="some-new-vendor")
    transport = httpx.ASGITransport(app=authenticated_app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await _post(ac)
    assert resp.status_code == 500
    assert captured_emit == []


@_needs_ffmpeg
@pytest.mark.asyncio
async def test_real_emitter_writes_funding_and_event_key_to_the_stream(client: Any, stub_state: Any) -> None:
    """End to end through w2v_common: the XADD fields billing reads."""
    redis = AsyncMock()
    stub_state(redis=redis)
    resp = await _post(client, {"X-Request-ID": "gw-req-99"})
    assert resp.status_code == 200, resp.text
    for _ in range(50):
        if redis.xadd.await_count:
            break
        await asyncio.sleep(0)
    redis.xadd.assert_awaited_once()
    fields: dict[str, str] = redis.xadd.await_args.args[1]
    assert fields["funding"] == "local_gpu"
    assert fields["event_key"] == cost_event_key(SERVICE_KEY, EVENT_KIND_REQUEST, TEST_TENANT_ID, "gw-req-99")
    assert fields["service_key"] == SERVICE_KEY
