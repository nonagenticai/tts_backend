"""The TTS cost event must name the app truthfully and must never rename the subject.

WHY THIS EXISTS
---------------
Two independent attribution defects sat in the same emit call in
``routers/tts_routes.synthesize``:

1. **``app_id`` was never stamped at all.** The kwarg was simply absent, and
   ``w2v_common.metering.cost_bus`` declares ``app_id: str = ""`` with no
   context fallback, so nothing failed and nothing warned. billing measured the
   consequence fleet-wide on 2026-09-02: ``app_id`` NULL on 401,316 of 401,316
   rows.

2. **``subject_id`` fell back to ``user.sub``.** On the bearer-licence lane --
   the lane every production request to this route takes -- ``AuthMiddleware``
   builds ``User(sub=str(tenant_id), ...)``, so that fallback wrote the
   TENANT's uuid into billing's subject column whenever the licence named no
   subject. A gap you can see is repairable; an identity silently replaced by
   another reads 100% attributed while naming nobody, forever.

Both halves are asserted here against the REAL route, capturing what the
emitter was actually handed. A test that grepped the source for a kwarg would
pass against a line that never executes.

⛔ The ABSENCE assertions matter as much as the presence ones. A defaulted
identity -- this service's own app row, the tenant uuid, any sentinel -- is
green on every coverage metric and wrong on every row, and ``billing_service``
never overwrites an app_id the caller presented, so it would be unrepairable.
``""`` is stored as NULL and lets billing derive the app from ``license_id``.
"""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np
import pytest

from services.tts_service import SynthesisResult

_SR = 24000
_APP_ID = "019f8a10-4c31-7b52-9d0e-2b6c9f4a1d77"
_SUBJECT = "019f8a10-4c31-7b52-9d0e-000000000001"
# Mirrors tests/conftest.py::TEST_TENANT_ID -- the value the deleted fallback
# would have smuggled into the subject column.
_TENANT = "11111111-1111-4111-8111-111111111111"


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
        t = np.linspace(0.0, 0.25, int(_SR * 0.25), endpoint=False)
        pcm = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        return SynthesisResult(
            audio_pcm=pcm,
            sample_rate=_SR,
            duration_sec=0.25,
            provider="kokoro",
            voice=voice or "af_heart",
            model="fake",
        )


def _with_license_state(app: Any, licence: Any) -> Any:
    """Wrap the ASGI app so ``request.state.license`` is what the middleware would set.

    ``AuthMiddleware`` assigns ``request.state.license = <introspection body>``,
    and starlette's ``Request.state`` is backed by ``scope["state"]``. Seeding
    the scope reproduces that exactly WITHOUT mutating ``main.app`` -- which is
    a module-level singleton shared by the whole suite, and leaking state onto
    it has already broken an unrelated test in this repo once.

    ``licence is None`` seeds nothing, which is the honest shape of the lanes
    that carry no licence object (Keycloak JWT, trusted-network) -- not a
    licence key holding ``None``.
    """

    async def wrapped(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and licence is not None:
            scope.setdefault("state", {})
            scope["state"]["license"] = licence
        await app(scope, receive, send)

    return wrapped


@pytest.fixture
def captured_emit(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _capture(_redis: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    import routers.tts_routes as tts_routes

    monkeypatch.setattr(tts_routes, "schedule_cost_event", _capture)
    return calls


@pytest.fixture
def stubbed_app(authenticated_app: Any) -> Any:
    """Install the fake provider and RESTORE app.state afterwards (shared singleton)."""
    state = authenticated_app.state
    sentinel = object()
    prev_service = getattr(state, "tts_service", sentinel)
    prev_redis = getattr(state, "metering_redis", sentinel)

    state.tts_service = _FakeService()
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


async def _drive(app: Any, licence: Any = None) -> dict[str, Any] | None:
    transport = httpx.ASGITransport(app=_with_license_state(app, licence))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/tts/synthesize", json={"text": "hello", "voice": "af_heart"})
    assert resp.status_code == 200, resp.text
    return None


# --------------------------------------------------------------------------
# app_id
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_id_comes_from_the_licence_introspection(
    stubbed_app: Any, captured_emit: list[dict[str, Any]]
) -> None:
    """The vouched app on the licence must reach the cost bus.

    FAILS on origin/main with ``cost event was emitted with NO app_id kwarg``:
    the emit call listed no ``app_id`` at all.
    """
    await _drive(stubbed_app, licence={"active": True, "app": {"id": _APP_ID, "key": "w2v"}})

    assert captured_emit, "the route emitted no cost event -- the test proves nothing"
    kwargs = captured_emit[0]
    assert "app_id" in kwargs, "cost event was emitted with NO app_id kwarg"
    assert kwargs["app_id"] == _APP_ID


@pytest.mark.asyncio
async def test_app_id_is_empty_when_no_licence_app_can_be_proven(
    stubbed_app: Any, captured_emit: list[dict[str, Any]]
) -> None:
    """No licence object -> ``""``, never a substitute.

    This is the assertion a "fix" that stamps this service's own app row would
    fail. billing stores "" as NULL and then derives the app from ``license_id``;
    an asserted-but-wrong app is never overwritten, so it would be permanent.
    """
    await _drive(stubbed_app, licence=None)

    assert captured_emit, "the route emitted no cost event -- the test proves nothing"
    kwargs = captured_emit[0]
    assert "app_id" in kwargs, "cost event was emitted with NO app_id kwarg"
    assert kwargs["app_id"] == ""
    # Not the tenant, not the service's own routing slug, not a sentinel.
    assert kwargs["app_id"] != str(kwargs["tenant_id"])
    assert kwargs["app_id"] != kwargs["service_key"]


@pytest.mark.asyncio
async def test_a_null_app_object_never_becomes_the_string_None(
    stubbed_app: Any, captured_emit: list[dict[str, Any]]
) -> None:
    """A JSON ``null`` app id must not stringify into billing as real data."""
    await _drive(stubbed_app, licence={"active": True, "app": {"id": None}})

    assert captured_emit, "the route emitted no cost event -- the test proves nothing"
    assert captured_emit[0]["app_id"] == ""


@pytest.mark.asyncio
async def test_a_licence_without_an_app_object_yields_empty(
    stubbed_app: Any, captured_emit: list[dict[str, Any]]
) -> None:
    """A tenant-scoped licence carrying no app is an absence, not an error."""
    await _drive(stubbed_app, licence={"active": True, "tenant_id": _TENANT})

    assert captured_emit, "the route emitted no cost event -- the test proves nothing"
    assert captured_emit[0]["app_id"] == ""


# --------------------------------------------------------------------------
# subject_id
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subject_id_is_never_substituted_by_the_tenant(
    stubbed_app: Any,
    captured_emit: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No vouched subject -> ``""``. NOT ``user.sub``, which is the tenant uuid.

    FAILS on origin/main: the emitter read ``get_subject_id_from_context() or
    user.sub``, and on the bearer-licence lane ``user.sub`` IS ``str(tenant_id)``
    -- so this asserted-empty case came out carrying the tenant's uuid in
    billing's subject column.
    """
    import routers.tts_routes as tts_routes

    monkeypatch.setattr(tts_routes, "get_subject_id_from_context", lambda: "")

    await _drive(stubbed_app, licence={"active": True, "tenant_id": _TENANT})

    assert captured_emit, "the route emitted no cost event -- the test proves nothing"
    kwargs = captured_emit[0]
    assert kwargs["subject_id"] == "", (
        "an absent subject was replaced by another identity: "
        f"{kwargs['subject_id']!r} (tenant is {kwargs['tenant_id']!r})"
    )
    assert kwargs["subject_id"] != str(kwargs["tenant_id"])


@pytest.mark.asyncio
async def test_subject_id_carries_the_vouched_subject_when_there_is_one(
    stubbed_app: Any,
    captured_emit: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the fallback must not remove the real subject.

    ⚠️ This one passes both before and after, by design: it is the control that
    proves the change above narrowed the field to the truth rather than simply
    blanking it. It is not evidence of the fix on its own.
    """
    import routers.tts_routes as tts_routes

    monkeypatch.setattr(tts_routes, "get_subject_id_from_context", lambda: _SUBJECT)

    await _drive(stubbed_app, licence={"active": True, "tenant_id": _TENANT})

    assert captured_emit, "the route emitted no cost event -- the test proves nothing"
    assert captured_emit[0]["subject_id"] == _SUBJECT


# --------------------------------------------------------------------------
# metadata_extra (billing's allowlist -- w2v-common 0.69.0, #168)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage_is_stamped_from_the_correlation_context(
    stubbed_app: Any,
    captured_emit: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared stage must reach billing's metadata allowlist.

    FAILS on origin/main with ``no metadata_extra kwarg``: the kwarg did not
    exist in w2v_common below 0.69.0 and this route passed nothing.
    """
    import routers.tts_routes as tts_routes

    monkeypatch.setattr(tts_routes, "get_stage_from_context", lambda: "narrate")

    await _drive(stubbed_app, licence={"active": True, "tenant_id": _TENANT})

    assert captured_emit, "the route emitted no cost event -- the test proves nothing"
    kwargs = captured_emit[0]
    assert "metadata_extra" in kwargs, "cost event was emitted with NO metadata_extra kwarg"
    assert kwargs["metadata_extra"]["stage"] == "narrate"


@pytest.mark.asyncio
async def test_an_undeclared_stage_is_skipped_not_emitted_empty(
    stubbed_app: Any,
    captured_emit: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No stage on the wire -> ``None``, which cost_bus SKIPS.

    ``""`` would be marshalled as a present-but-empty field. ``None`` is the
    only spelling the emitter treats as "not provided" (``if value is None:
    continue``), and it is what the receiver's ``if c.get(k)`` needs to leave
    the column alone.
    """
    import routers.tts_routes as tts_routes

    monkeypatch.setattr(tts_routes, "get_stage_from_context", lambda: "")

    await _drive(stubbed_app, licence={"active": True, "tenant_id": _TENANT})

    assert captured_emit, "the route emitted no cost event -- the test proves nothing"
    assert captured_emit[0]["metadata_extra"]["stage"] is None


@pytest.mark.asyncio
async def test_no_allowlist_field_is_invented_to_fill_a_column(
    stubbed_app: Any, captured_emit: list[dict[str, Any]]
) -> None:
    """The five fields this service cannot know must be ABSENT, not defaulted.

    A value invented to fill a column reads 100% green on every coverage metric
    forever while being wrong on every row. ``fallback_used`` is the sharpest
    case: ``TTSService.synthesize`` resolves exactly one provider and raises
    rather than trying another, so even a constant ``False`` would assert a
    mechanism that does not exist.
    """
    await _drive(stubbed_app, licence={"active": True, "tenant_id": _TENANT})

    assert captured_emit, "the route emitted no cost event -- the test proves nothing"
    metadata = captured_emit[0]["metadata_extra"]
    for field in ("schema_id", "job_id", "fallback_used", "worker_name", "caller_client_id"):
        assert field not in metadata, f"{field} was invented; this service cannot know it"


@pytest.mark.asyncio
async def test_the_wire_fields_really_carry_stage_and_app_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing a kwarg is not the same as billing receiving a field.

    Every other test here captures what the ROUTE handed the emitter, with
    ``schedule_cost_event`` mocked out. That proves the call site and nothing
    about the library: if ``metadata_extra`` were misspelled, dropped by the
    allowlist, or absent from the installed w2v_common, those tests would still
    be green. So this one drives the REAL ``w2v_common.metering.emit_cost_event``
    against a stub redis and asserts on the dict that would actually be XADDed.
    """
    import uuid as _uuid

    from w2v_common.metering import emit_cost_event

    # The REAL routing slugs, imported rather than retyped: a literal
    # "w2v-tts" here trips detect-secrets' Secret Keyword rule on the
    # `service_key=` assignment (it is a routing slug, not a credential -- see
    # the pragma on the same constant in services/metering.py), and a test that
    # hardcodes them could drift away from what the route actually emits.
    from services.metering import COST_SOURCE, SERVICE_KEY

    published: list[dict[str, str]] = []

    class _StubRedis:
        async def xadd(self, _stream: str, fields: dict[str, str], **_kw: Any) -> str:
            published.append(dict(fields))
            return "1-1"

    monkeypatch.setenv("METERING_STREAM", "billing:costs:test")

    await emit_cost_event(
        _StubRedis(),  # type: ignore[arg-type]
        service_key=SERVICE_KEY,
        cost_source=COST_SOURCE,
        tenant_id=_uuid.UUID(_TENANT),
        provider="kokoro",
        model="fake",
        input_tokens=5,
        output_tokens=0,
        latency_ms=1,
        license_id="",
        subject_id="",
        app_id=_APP_ID,
        metadata_extra={"stage": "narrate"},
    )

    assert published, "emit_cost_event published nothing -- the test proves nothing"
    fields = published[0]
    assert fields["app_id"] == _APP_ID
    assert fields["stage"] == "narrate", (
        "metadata_extra did not survive marshalling -- the installed w2v_common "
        "is below 0.69.0, or the key is outside billing's allowlist"
    )
    # The absence half, on the wire this time.
    for field in ("schema_id", "job_id", "fallback_used", "worker_name", "caller_client_id"):
        assert field not in fields
