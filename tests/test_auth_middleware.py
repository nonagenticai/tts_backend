"""Wiring tests for the consolidated auth layer.

Auth is now w2v_common's ``AuthMiddleware`` (re-exported from ``auth.middleware``).
The introspection internals — including the single-flight lock + 10s timeout that
fixed the 2026-06-27 narration 401 — are owned and unit-tested in w2v_common.
These tests only assert this service's wiring: public routes are open, protected
routes require a valid license token, and a valid token populates request state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import w2v_common.auth.middleware as w2v_auth
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from w2v_common.auth.identity_context import get_subject_id_from_context

from auth.middleware import AuthMiddleware


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/echo")
    async def _echo(request: Request) -> JSONResponse:
        # w2v_common carries identity on request.state.user (canonical) +
        # flat request.state.tenant_id; the real subject is in the identity
        # ContextVar (w2v_common maps user.sub->tenant_id), which is exactly
        # how tts_routes sources subject_id for metering.
        return JSONResponse(
            {
                "tenant_id": getattr(request.state, "tenant_id", None),
                "subject_id": get_subject_id_from_context() or None,
            }
        )

    return app


@pytest.fixture(autouse=True)
def _clear_license_cache() -> None:
    # Reset w2v_common's module-level introspection caches + single-flight locks.
    w2v_auth._license_cache = None
    w2v_auth._license_neg_cache = None
    w2v_auth._introspect_locks.clear()


@pytest.fixture
async def auth_client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_health_is_public(auth_client: httpx.AsyncClient) -> None:
    resp = await auth_client.get("/health")
    assert resp.status_code == 200


async def test_protected_route_without_token_is_rejected(
    auth_client: httpx.AsyncClient,
) -> None:
    # Fail-closed: w2v_common does not mint a synthetic identity for an
    # anonymous caller (trust_incluster_identity is off by default).
    resp = await auth_client.get("/api/echo")
    assert resp.status_code == 401


async def test_protected_route_with_valid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake(token: str) -> dict[str, Any]:
        assert token == "abc.def.ghi"
        return {
            "active": True,
            # A real UUID: from w2v-common 0.67.10 the middleware REFUSES a slug tenant
            # with 401 ("an empty or slug tenant would bypass isolation"), so a slug
            # fixture here would be testing a request that can no longer exist.
            "tenant_id": "11111111-1111-4111-8111-111111111111",
            "subject_id": "subject-uuid-9",
            "license_id": "lic-77",
            "roles": ["user"],
        }

    monkeypatch.setattr(w2v_auth, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/echo", headers={"X-Tenant-Id": "Bearer abc.def.ghi"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tenant_id"] == "11111111-1111-4111-8111-111111111111"
    assert body["subject_id"] == "subject-uuid-9"


async def test_protected_route_with_invalid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake(_token: str) -> dict[str, Any]:
        return {"active": False}

    monkeypatch.setattr(w2v_auth, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/echo", headers={"X-Tenant-Id": "Bearer dud-token"})

    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"
