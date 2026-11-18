"""The TTS cost event must carry the licence its own auth middleware resolved.

WHY THIS EXISTS
---------------
Measured on prod 2026-08-27: ``w2v-tts`` emitted **1,120 of 1,120** cost events
over 7 days with ``license_id IS NULL`` -- 100%. Its sibling ``w2v-asr``, running
the IDENTICAL ``w2v_common`` ``AuthMiddleware``, was **0%** NULL over 1,063
events. The only difference was one missing kwarg at the emit call.

Nothing failed and nothing warned, because
``w2v_common.metering.cost_bus`` declares ``license_id: str = ""`` -- a default
with no context fallback. Licence-less spend is still billed and still capped,
but ``CostDaily.license_attributed_clause`` drops it from every licence-keyed
aggregate, so it can never reach a customer invoice line.

This test drives the REAL route and captures what the emitter was actually
handed. A test that merely grepped the source for the kwarg would pass against a
line that never executes.
"""

from __future__ import annotations

import shutil
from typing import Any

import numpy as np
import pytest

from services.tts_service import SynthesisResult

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")

_SR = 24000
_LICENCE = "019f7f7c-5006-72c0-8fa7-e6317340c0c9"


class _FakeService:
    def loaded_models(self) -> list[str]:
        return ["kokoro"]

    def synthesize(
        self,
        text: str,
        provider: str | None,
        voice: str | None,
        speed: float,
        ssml: bool | None = None,
    ) -> SynthesisResult:
        t = np.linspace(0.0, 3.0, int(_SR * 3.0), endpoint=False)
        pcm = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        return SynthesisResult(
            audio_pcm=pcm,
            sample_rate=_SR,
            duration_sec=3.0,
            provider="kokoro",
            voice=voice or "af_heart",
            model="fake",
        )


@pytest.fixture
def captured_emit(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the kwargs the route hands the cost bus, without emitting."""
    calls: list[dict[str, Any]] = []

    def _capture(_redis: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    import routers.tts_routes as tts_routes

    monkeypatch.setattr(tts_routes, "schedule_cost_event", _capture)
    return calls


@pytest.fixture
def stubbed_app(authenticated_app: Any) -> Any:
    """Install the fake provider and RESTORE app.state afterwards.

    ``main.app`` is a module-level singleton shared by every test in the suite,
    so leaving ``tts_service`` set here made an unrelated test
    (``test_synthesize_503_when_no_provider``) fail while still passing in
    isolation -- a leaked singleton, not a real regression. Caught by running the
    whole suite rather than only the new file.
    """
    state = authenticated_app.state
    sentinel = object()
    prev_service = getattr(state, "tts_service", sentinel)
    prev_redis = getattr(state, "metering_redis", sentinel)

    state.tts_service = _FakeService()
    # A non-None metering_redis is what opens the emit branch; the capture
    # fixture means nothing is actually published.
    state.metering_redis = object()
    try:
        yield authenticated_app
    finally:
        for name, prev in (("tts_service", prev_service), ("metering_redis", prev_redis)):
            if prev is sentinel:
                if hasattr(state, name):
                    delattr(state, name)
            else:
                setattr(state, name, prev)


async def _drive(client: Any) -> Any:
    return await client.post("/api/tts/synthesize", json={"text": "hello", "voice": "af_heart"})


@pytest.mark.asyncio
async def test_cost_event_carries_the_licence_from_the_identity_context(
    client: Any,
    stubbed_app: Any,
    captured_emit: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import routers.tts_routes as tts_routes

    monkeypatch.setattr(tts_routes, "get_license_id_from_context", lambda: _LICENCE)

    resp = await _drive(client)
    assert resp.status_code == 200, resp.text

    assert captured_emit, "the route emitted no cost event — the test proves nothing"
    kwargs = captured_emit[0]
    # The assertion the defect would fail: the kwarg must be PRESENT and carry
    # the resolved licence. Before the fix it was absent entirely.
    assert "license_id" in kwargs, "cost event was emitted with NO license_id kwarg"
    assert kwargs["license_id"] == _LICENCE


@pytest.mark.asyncio
async def test_absent_licence_stays_empty_and_is_never_invented(
    client: Any,
    stubbed_app: Any,
    captured_emit: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No licence in context must yield "" — never the tenant, never a sentinel.

    Substituting some other id here would put real spend on the wrong invoice
    line, which is worse than the unattributed row this change removes.
    """
    import routers.tts_routes as tts_routes

    monkeypatch.setattr(tts_routes, "get_license_id_from_context", lambda: "")

    resp = await _drive(client)
    assert resp.status_code == 200, resp.text

    assert captured_emit, "the route emitted no cost event — the test proves nothing"
    kwargs = captured_emit[0]
    assert kwargs.get("license_id") == ""
    assert kwargs["license_id"] != kwargs.get("tenant_id")
