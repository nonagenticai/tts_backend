"""GPU admission lane ``synth`` on /api/tts/synthesize (#70).

``_SYNTH_LOCK`` serialises inference (concurrent kokoro/espeak garbles audio)
but its waiters were unbounded, had no deadline, and a request whose client had
gone still synthesised. The lane bounds all three. These tests drive the REAL
app over ASGI with the model stubbed to block, so every number asserted below
is what a caller would observe.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from w2v_common.admission import AdmissionGate, ClientGone, GpuBusy, LaneConfig

import routers.tts_routes as tts_routes
from models import SynthesizeRequest
from services.admission import SYNTH_LANE, SYNTH_LANE_DEFAULT, build_gpu_gate
from services.tts_service import SynthesisResult

_SR = 24000
_USER = SimpleNamespace(effective_tenant_id="3f2504e0-4f89-11d3-9a0c-0305e82c3301", sub="test-subject")


class _BlockingService:
    """TTSService stand-in whose synthesis blocks until ``release`` is set.

    ``running``/``peak`` measure how many synthesis calls overlap -- the property
    the lane (and the lock behind it) exists to bound.
    """

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        self.calls = 0
        self.running = 0
        self.peak = 0
        self._mu = threading.Lock()

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
        with self._mu:
            self.calls += 1
            self.running += 1
            self.peak = max(self.peak, self.running)
        self.entered.set()
        try:
            # Bounded so a broken test cannot hang the suite.
            self.release.wait(timeout=15.0)
        finally:
            with self._mu:
                self.running -= 1
        pcm = np.zeros(_SR // 10, dtype=np.float32)
        return SynthesisResult(
            audio_pcm=pcm,
            sample_rate=_SR,
            duration_sec=0.1,
            provider="kokoro",
            voice=voice or "af_heart",
            model="fake",
        )


async def _until(cond: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


@pytest.fixture
def svc() -> _BlockingService:
    return _BlockingService()


@pytest.fixture
def gated_app(authenticated_app: Any, svc: _BlockingService) -> Iterator[Any]:
    """The real app with a FRESH gate at the issue's numbers and the blocking stub."""
    state = authenticated_app.state
    saved = {k: getattr(state, k, None) for k in ("tts_service", "gpu_gate", "metering_redis")}
    state.tts_service = svc
    state.metering_redis = None
    state.gpu_gate = AdmissionGate({SYNTH_LANE: LaneConfig(max_active=1, max_waiting=16, max_wait_s=20.0)}, environ={})
    try:
        yield authenticated_app
    finally:
        svc.release.set()
        for k, v in saved.items():
            setattr(state, k, v)


@pytest.fixture
async def gated_client(gated_app: Any) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=gated_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as ac:
        yield ac


def test_code_defaults_are_the_issue_numbers() -> None:
    assert SYNTH_LANE == "synth"
    assert (SYNTH_LANE_DEFAULT.max_active, SYNTH_LANE_DEFAULT.max_waiting, SYNTH_LANE_DEFAULT.max_wait_s) == (
        1,
        16,
        20.0,
    )


def test_env_overrides_the_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_LANE_SYNTH_MAX_WAITING", "64")
    monkeypatch.setenv("GPU_LANE_SYNTH_MAX_WAIT_S", "120")
    cfg = build_gpu_gate().lane(SYNTH_LANE).config
    assert (cfg.max_active, cfg.max_waiting, cfg.max_wait_s) == (1, 64, 120.0)


def test_handlers_are_installed() -> None:
    from main import app

    assert GpuBusy in app.exception_handlers
    assert ClientGone in app.exception_handlers


async def test_burst_of_20_is_bounded_1_running_16_queued_3_rejected(
    gated_client: httpx.AsyncClient, gated_app: Any, svc: _BlockingService
) -> None:
    lane = gated_app.state.gpu_gate.lane(SYNTH_LANE)
    tasks = [
        asyncio.create_task(gated_client.post("/api/tts/synthesize", json={"text": f"line {i}"})) for i in range(20)
    ]

    # The overflow answers IMMEDIATELY, while the model is still blocked.
    await _until(lambda: sum(t.done() for t in tasks) == 3)
    rejected = [t.result() for t in tasks if t.done()]
    for r in rejected:
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["error"] == "gpu_busy"
        assert body["lane"] == "synth"
        assert int(r.headers["Retry-After"]) == body["retry_after"]
        assert 1 <= body["retry_after"] <= 120

    assert lane.active == 1
    assert lane.waiting == 16
    assert svc.calls == 1, "exactly one synthesis may be running"

    # /health must stay responsive while the lane is saturated.
    t0 = time.monotonic()
    h = await gated_client.get("/health")
    assert h.status_code == 200
    assert time.monotonic() - t0 < 0.2, "/health blocked behind synthesis"

    svc.release.set()
    done = await asyncio.gather(*tasks)
    assert sorted(r.status_code for r in done) == [200] * 17 + [503] * 3
    assert svc.calls == 17
    assert svc.peak == 1, "synthesis calls overlapped"
    assert lane.active == 0 and lane.waiting == 0


async def test_waiting_past_max_wait_s_is_a_503(authenticated_app: Any, svc: _BlockingService) -> None:
    state = authenticated_app.state
    saved = {k: getattr(state, k, None) for k in ("tts_service", "gpu_gate", "metering_redis")}
    state.tts_service = svc
    state.metering_redis = None
    state.gpu_gate = AdmissionGate({SYNTH_LANE: LaneConfig(max_active=1, max_waiting=16, max_wait_s=0.3)}, environ={})
    try:
        transport = httpx.ASGITransport(app=authenticated_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as ac:
            first = asyncio.create_task(ac.post("/api/tts/synthesize", json={"text": "a"}))
            await asyncio.get_running_loop().run_in_executor(None, svc.entered.wait, 5.0)
            queued = await ac.post("/api/tts/synthesize", json={"text": "b"})
            assert queued.status_code == 503
            assert queued.json()["error"] == "gpu_busy"
            assert "Retry-After" in queued.headers
            svc.release.set()
            assert (await first).status_code == 200
        assert svc.calls == 1, "the timed-out request must not have synthesised"
    finally:
        svc.release.set()
        for k, v in saved.items():
            setattr(state, k, v)


def _direct_request(app_state: Any, gone: dict[str, bool]) -> Any:
    async def _is_disconnected() -> bool:
        return gone["v"]

    return SimpleNamespace(app=SimpleNamespace(state=app_state), headers={}, is_disconnected=_is_disconnected)


async def test_client_gone_while_queued_is_skipped(svc: _BlockingService) -> None:
    state = SimpleNamespace(
        tts_service=svc,
        metering_redis=None,
        gpu_gate=AdmissionGate({SYNTH_LANE: SYNTH_LANE_DEFAULT}, environ={}),
    )
    first = asyncio.create_task(
        tts_routes.synthesize(_direct_request(state, {"v": False}), SynthesizeRequest(text="a"), _USER)  # type: ignore[arg-type]
    )
    await asyncio.get_running_loop().run_in_executor(None, svc.entered.wait, 5.0)

    gone = {"v": False}
    second = asyncio.create_task(
        tts_routes.synthesize(_direct_request(state, gone), SynthesizeRequest(text="b"), _USER)  # type: ignore[arg-type]
    )
    await _until(lambda: state.gpu_gate.lane(SYNTH_LANE).waiting == 1)
    gone["v"] = True  # client hangs up while queued
    svc.release.set()

    assert (await first).status_code == 200
    with pytest.raises(ClientGone):
        await second
    assert svc.calls == 1, "a request whose client is gone must not synthesise"


async def test_cancel_while_queued_is_not_counted_and_never_runs(svc: _BlockingService) -> None:
    state = SimpleNamespace(
        tts_service=svc,
        metering_redis=None,
        gpu_gate=AdmissionGate({SYNTH_LANE: SYNTH_LANE_DEFAULT}, environ={}),
    )
    first = asyncio.create_task(
        tts_routes.synthesize(_direct_request(state, {"v": False}), SynthesizeRequest(text="a"), _USER)  # type: ignore[arg-type]
    )
    await asyncio.get_running_loop().run_in_executor(None, svc.entered.wait, 5.0)
    second = asyncio.create_task(
        tts_routes.synthesize(_direct_request(state, {"v": False}), SynthesizeRequest(text="b"), _USER)  # type: ignore[arg-type]
    )
    await _until(lambda: state.gpu_gate.lane(SYNTH_LANE).waiting == 1)

    before = tts_routes.TTS_CANCELLED._value.get()
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert tts_routes.TTS_CANCELLED._value.get() == before, "a queued request did no GPU work"

    svc.release.set()
    assert (await first).status_code == 200
    assert svc.calls == 1


async def test_metrics_render_the_synth_lane(gated_client: httpx.AsyncClient) -> None:
    r = await gated_client.get("/metrics")
    assert r.status_code == 200
    assert 'gpu_admission_active{lane="synth"}' in r.text
    assert 'gpu_admission_rejected_total{lane="synth",reason="full"}' in r.text


async def test_a_shut_down_gate_answers_503_not_500(
    gated_client: httpx.AsyncClient, gated_app: Any, svc: _BlockingService
) -> None:
    """GateClosedError is a RuntimeError; the route's generic RuntimeError arm
    used to turn a clean shutdown into a 500 "Synthesis failed"."""
    await gated_app.state.gpu_gate.shutdown(timeout=1.0)
    r = await gated_client.post("/api/tts/synthesize", json={"text": "late"})
    assert r.status_code == 503, r.text
    assert svc.calls == 0


async def test_lifespan_replaces_a_closed_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    monkeypatch.setattr(main, "run_premaster_self_test", lambda: None)
    stale = AdmissionGate({SYNTH_LANE: SYNTH_LANE_DEFAULT}, environ={})
    await stale.shutdown()
    main.app.state.gpu_gate = stale
    async with main.lifespan(main.app):
        live = main.app.state.gpu_gate
        assert live is not stale
        assert not live.lane(SYNTH_LANE).closed
    assert live.lane(SYNTH_LANE).closed, "lifespan shutdown must close the gate"


async def test_client_gone_while_queued_is_skipped_through_the_real_middleware_stack(
    gated_app: Any, svc: _BlockingService
) -> None:
    """Raw ASGI through ``main.app`` -- RequestId, SecurityHeaders and
    RequestLogging are BaseHTTPMiddleware, which hide ``http.disconnect`` from
    ``request.is_disconnected()``. Without DisconnectProbeMiddleware outermost the
    queued job still synthesised (2 calls, second answered 200)."""
    body = b'{"text": "hi"}'

    async def call(gone: asyncio.Event) -> list[dict[str, Any]]:
        sent = False
        out: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await gone.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            out.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/tts/synthesize",
            "raw_path": b"/api/tts/synthesize",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json"), (b"host", b"test")],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
        }
        await gated_app(scope, receive, send)
        return out

    stays, leaves = asyncio.Event(), asyncio.Event()
    first = asyncio.create_task(call(stays))
    await asyncio.get_running_loop().run_in_executor(None, svc.entered.wait, 5.0)
    second = asyncio.create_task(call(leaves))
    await _until(lambda: gated_app.state.gpu_gate.lane(SYNTH_LANE).waiting == 1)
    leaves.set()  # client hangs up while queued
    await asyncio.sleep(0.05)
    svc.release.set()

    def status(msgs: list[dict[str, Any]]) -> int | None:
        return next((m["status"] for m in msgs if m["type"] == "http.response.start"), None)

    assert status(await asyncio.wait_for(first, 10)) == 200
    assert status(await asyncio.wait_for(second, 10)) == 499
    assert svc.calls == 1, "the abandoned queued job must never run"
