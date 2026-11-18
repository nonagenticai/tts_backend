"""The entitlement gate on the tenant-scoped router actually refuses an
unlicensed tenant.

Until now this service authenticated the licence token for tenant IDENTITY and
stopped there: ``AuthMiddleware`` answers "who are you", never "may you". A
tenant whose licence was ``expired`` or ``revoked`` still presented a
syntactically valid token, so ``/api/tts/synthesize`` served them — and billed
them (1,741 ``w2v-tts`` rows in ``billing_cost_events``). ``LICENSE_GATE`` is
mounted router-wide on ``src/routers/tts_routes.py`` to close that.

Every test here drives a REAL HTTP request through the REAL router, because the
interesting question is not whether ``require_license`` works — w2v_common owns
that — but whether THIS service's routes actually carry it. This FastAPI build
includes routers lazily (``_IncludedRouter``), so a parent router's
``dependencies=`` are applied when a request is matched, not baked into the
route objects at import time: static introspection of ``route.dependant`` would
report these routes as ungated, a *false negative*. A request is the only honest
probe.

The paired 200s (entitled, flag-off) are what stop this file from passing
vacuously: if the gate were mounted wrongly, or the fixtures broken, those cases
would not be 200s either.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth import clear_license_status_cache
from w2v_common.auth.licenses_read_client import LicensesClientError

from auth.dependencies import get_current_user
from routers.tts_routes import router as tts_router
from tests.conftest import TEST_TENANT_ID

SUBJECT = "user-under-test"

#: The cheapest genuinely tenant-scoped route on the gated router: it reads only
#: ``request.app.state.tts_service``, which is trivial to stub, and does no
#: synthesis, no Redis and no cost emit. So a non-200 here can only be the gate —
#: which is the whole point.
GATED_PATH = "/api/tts/voices"

#: The expensive one. Asserted too, because it is the route that is BILLED: if
#: only ``/voices`` were gated the revenue leak would still be open.
GATED_SYNTH_PATH = "/api/tts/synthesize"

#: A probe. Must answer 200 for a revoked tenant: a 402/503 on a liveness probe
#: flaps the launchd unit, and an unlicensed tenant cannot be billed by killing
#: our own service.
PROBE_PATH = "/health/live"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Record:
    """A licences wire record. Only ``model_dump`` is on the gate's path."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    """An ACTIVE licence detail (the 200 from ``GET /licenses/active``)."""

    conflict = False


class _FakeLicenses:
    """Stands in for ``LicensesReader``.

    ``active`` None models the authoritative 404 ("no active licence"), which
    sends the gate to the subject inventory to refine none/expired/revoked.
    ``error`` models an upstream failure, which must 503, never allow.
    """

    def __init__(
        self,
        *,
        active: _Detail | None = None,
        inventory: list[_Record] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._error = error

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> _Detail | None:
        if self._error is not None:
            raise self._error
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[_Record]:
        if self._error is not None:
            raise self._error
        return list(self._inventory)


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=TEST_TENANT_ID, status="revoked")]


def _expired() -> list[_Record]:
    return [
        _Record(
            app={"key": "w2v"},
            tenant_id=TEST_TENANT_ID,
            status="expired",
            expires_at="2020-01-01T00:00:00+00:00",
        )
    ]


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=TEST_TENANT_ID,
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


class _FakeProvider:
    name = "kokoro"
    voices = ["af_heart"]
    default_voice = "af_heart"


class _FakeTTSService:
    """Enough of ``TTSService`` for ``/voices`` to answer 200 without a model."""

    default_provider = "kokoro"
    providers = {"kokoro": _FakeProvider()}

    def loaded_models(self) -> list[str]:
        return ["kokoro"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts() -> Any:
    """The gate caches 60s per (tenant, app, subject). Two tests in one process
    would otherwise share a verdict and the second would assert nothing."""
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


@pytest.fixture
def gated_app() -> FastAPI:
    """The REAL router, with identity stamped the way AuthMiddleware stamps it.

    ``ENABLE_AUTH=false`` in the test env means the real middleware is not
    installed, so nothing would populate ``request.state.tenant_id`` and the gate
    would refuse everything for the trivial reason that there is no tenant — a
    test that passes for the wrong reason. Stamping it here reproduces production
    identity and forces the assertions onto the licence verdict itself.
    """
    application = FastAPI()

    class _StampIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request: Any, call_next: Any) -> Any:
            request.state.tenant_id = TEST_TENANT_ID
            request.state.subject_id = SUBJECT
            return await call_next(request)

    application.add_middleware(_StampIdentity)
    application.include_router(tts_router)
    application.state.tts_service = _FakeTTSService()

    from tests.conftest import _authenticated_user

    application.dependency_overrides[get_current_user] = _authenticated_user
    return application


async def _get(app: FastAPI, path: str) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path)


async def _post(app: FastAPI, path: str, json: dict[str, Any]) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(path, json=json)


def _licenses(monkeypatch: pytest.MonkeyPatch, fake: _FakeLicenses) -> None:
    """Patch the client the gate resolves through, in the module that calls it."""
    monkeypatch.setattr("w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake)


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inventory", "code"),
    [
        ([], "license_none"),
        (_revoked(), "license_revoked"),
        (_expired(), "license_expired"),
    ],
    ids=["none", "revoked", "expired"],
)
async def test_armed_gate_402s_an_unentitled_tenant(
    gated_app: FastAPI, monkeypatch: pytest.MonkeyPatch, inventory: list[_Record], code: str
) -> None:
    """402 with the state-specific code — the tenant's licence is the problem."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    response = await _get(gated_app, GATED_PATH)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == code


async def test_armed_gate_402s_the_billed_synthesize_route(gated_app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """The route that produces `w2v-tts` cost events is refused BEFORE its body
    runs — the gate is a router dependency, so it precedes request-model
    validation of a payload that would otherwise be synthesised and billed."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _post(gated_app, GATED_SYNTH_PATH, {"text": "hello"})

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == "license_revoked"


async def test_armed_gate_503s_when_entitlement_cannot_be_established(
    gated_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upstream failure is NOT an allow, and NOT a 402 either: we do not know."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    response = await _get(gated_app, GATED_PATH)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable"
    assert response.headers["Retry-After"]


# ---------------------------------------------------------------------------
# Armed: an entitled tenant is served (the non-vacuity pair)
# ---------------------------------------------------------------------------


async def test_armed_gate_serves_an_entitled_tenant(gated_app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = await _get(gated_app, GATED_PATH)

    assert response.status_code == 200, response.text
    assert response.json()["default_provider"] == "kokoro"


# ---------------------------------------------------------------------------
# Dormant on merge
# ---------------------------------------------------------------------------


async def test_gate_is_dormant_while_the_flag_is_unset(gated_app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Merging this changes NOTHING until the flag is armed in this service's
    env. A revoked tenant is still served — deliberately — so the rollout is a
    flag flip, not a deploy."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(gated_app, GATED_PATH)

    assert response.status_code == 200, response.text


async def test_gate_is_dormant_when_the_flag_is_explicitly_false(
    gated_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(gated_app, GATED_PATH)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Probes are never gated
# ---------------------------------------------------------------------------


async def test_the_real_apps_probe_is_served_for_a_revoked_tenant(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Against the REAL ``main.app``: a gated probe path flaps the launchd unit,
    so ``/health/live`` must stay 200 with the gate armed and a revoked tenant."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await client.get(PROBE_PATH)

    assert response.status_code == 200, response.text


async def test_the_services_own_probe_paths_are_exempt_by_path() -> None:
    """Belt and braces: even if a future refactor mounted the gate on the health
    routes, w2v_common would still refuse to enforce on these paths. This is also
    why this service hand-rolls NO exemption list of its own."""
    from w2v_common.auth import is_license_exempt_path

    for path in ("/health", "/health/live", "/health/ready", "/healthz", "/readyz"):
        assert is_license_exempt_path(path), path
    assert not is_license_exempt_path(GATED_PATH)
    assert not is_license_exempt_path(GATED_SYNTH_PATH)
