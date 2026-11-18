"""A synthesis the caller abandons must be COUNTED, not silent.

`/api/tts/synthesize` runs the blocking provider on a worker thread via
the ``synth`` admission lane's worker thread. Cancelling the awaiting task -- a client hang-up, or an
orchestrator hitting its own timeout -- raises ``CancelledError`` at the await
point and unwinds the handler BEFORE ``schedule_cost_event``. The worker thread
is not cancelled; it runs to completion on the GPU. So the request costs real
wall time and leaves no trace: no cost event for billing, no response for the
access log.

These tests pin the counter AND the two properties that must not change with
it: the cost event still must not fire (billing for audio nobody received would
be a worse defect), and ``CancelledError`` must still propagate (swallowing it
breaks cooperative cancellation).
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import routers.tts_routes as tts_routes
from models import SynthesizeRequest
from services.tts_service import SynthesisResult

_SR = 24000


class _BlockingService:
    """Stands in for TTSService, blocking inside the worker thread until released.

    ``entered`` lets the test wait until execution is genuinely inside
    the ``synth`` admission lane's worker thread before cancelling -- cancelling earlier would prove
    nothing about this code path.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.completed = False

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
        self.entered.set()
        # Mirrors the real failure: the thread is NOT cancelled with the task,
        # it runs to completion. Bounded so a broken test cannot hang the suite.
        self.release.wait(timeout=10.0)
        self.completed = True
        t = np.linspace(0.0, 1.0, _SR, endpoint=False)
        pcm = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        return SynthesisResult(
            audio_pcm=pcm,
            sample_rate=_SR,
            duration_sec=1.0,
            provider="kokoro",
            voice=voice or "af_heart",
            model="fake",
        )


def _counter_value() -> float:
    return tts_routes.TTS_CANCELLED._value.get()


def _make_request(svc: _BlockingService, metering_redis: Any) -> Any:
    """Minimal stand-in for the Request the handler actually touches.

    The handler reads ``request.app.state.tts_service`` (via
    ``_service_from_request``), ``request.app.state.metering_redis`` and
    ``request.headers`` for Accept negotiation, so a namespace is enough and
    keeps the test off the network/auth stack.
    """

    async def _connected() -> bool:
        return False

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(tts_service=svc, metering_redis=metering_redis)),
        headers={},
        # The admission gate asks this once, when the job is admitted.
        is_disconnected=_connected,
    )


@pytest.mark.asyncio
async def test_cancelled_synthesis_increments_counter_and_emits_no_cost_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        tts_routes,
        "schedule_cost_event",
        lambda *a, **kw: emitted.append(kw),
    )

    svc = _BlockingService()
    # Non-None so the emit branch is ARMED: if the handler ever emitted on this
    # path, `emitted` would record it. A None redis would make the assertion
    # below vacuous -- it would pass for the wrong reason.
    request = _make_request(svc, metering_redis=object())
    user = SimpleNamespace(effective_tenant_id="3f2504e0-4f89-11d3-9a0c-0305e82c3301", sub="test-subject")

    before = _counter_value()

    task = asyncio.create_task(
        tts_routes.synthesize(request, SynthesizeRequest(text="Hello world."), user)  # type: ignore[arg-type]
    )
    # Wait until the blocking call is genuinely running on the worker thread.
    await asyncio.get_running_loop().run_in_executor(None, svc.entered.wait, 10.0)
    assert svc.entered.is_set(), "never reached the worker thread; the test would prove nothing"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _counter_value() - before == 1.0, "cancelled synthesis was not counted"
    assert emitted == [], "a cancelled synthesis must not emit a cost event for audio nobody received"

    # Let the orphaned worker thread finish so it does not outlive the test.
    svc.release.set()


@pytest.mark.asyncio
async def test_successful_synthesis_does_not_increment_the_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The counter must discriminate. A counter that only ever goes up is not a
    measurement -- this is the negative control for the test above."""
    monkeypatch.setattr(tts_routes, "schedule_cost_event", lambda *a, **kw: None)

    svc = _BlockingService()
    svc.release.set()  # do not block: run straight through
    request = _make_request(svc, metering_redis=None)
    user = SimpleNamespace(effective_tenant_id="3f2504e0-4f89-11d3-9a0c-0305e82c3301", sub="test-subject")

    before = _counter_value()
    resp = await tts_routes.synthesize(request, SynthesizeRequest(text="Hello world."), user)  # type: ignore[arg-type]

    assert resp.status_code == 200
    assert _counter_value() == before
