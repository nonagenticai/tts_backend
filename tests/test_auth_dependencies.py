"""Contract tests for the auth dependencies.

Product-owner ruling (2026-08-26): a ``tenant_id`` may NEVER hold a sentinel.
All traffic carries a real registered tenant forwarded from the requestor; an
absent tenant is an ERROR, never a substituted default.

These tests replace the two that pinned the OPPOSITE contract
(``test_default_tenant_id_falls_back_to_nil_sentinel`` and
``test_anonymous_user_attributes_to_nil_sentinel``).
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from w2v_common.tenant import MissingTenantError

import auth.dependencies as deps
from auth.models import User

NIL = "00000000-0000-0000-0000-000000000000"


def test_no_sentinel_constant_is_reachable_from_the_package() -> None:
    """The substitution machinery must be GONE, not merely unused."""
    assert not hasattr(deps, "UNATTRIBUTED_TENANT_ID")
    assert not hasattr(deps, "DEFAULT_TENANT_ID")
    assert not hasattr(deps, "_ANONYMOUS_USER")


def test_require_tenant_accepts_a_real_tenant() -> None:
    user = User(sub="s", preferred_username="u", tenant_id="11111111-1111-4111-8111-111111111111")
    assert deps.require_tenant(user) == uuid.UUID("11111111-1111-4111-8111-111111111111")


@pytest.mark.parametrize("slug", ["n8n-dev", "acme-tenant", "anonymous"])
def test_a_non_uuid_realm_slug_is_refused_at_the_MODEL_boundary(slug: str) -> None:
    """This test'"'"'s contract INVERTED with w2v-common 0.67.10, deliberately.

    It used to assert a Keycloak realm slug was accepted. ``User.tenant_id`` is
    now typed ``uuid.UUID | None``, so pydantic refuses a slug at construction —
    earlier than this guard could, and in one place for the whole fleet. Kept
    (rather than deleted) because "what happens to a slug" is exactly the
    question the old test answered, and the answer changed.
    """
    with pytest.raises(ValidationError):
        User(sub="s", preferred_username="u", tenant_id=slug)


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_tenant_is_refused_at_the_MODEL_boundary(blank: str) -> None:
    """Blanks never reach the guard any more — pydantic rejects them first."""
    with pytest.raises(ValidationError):
        User(sub="", preferred_username="u", tenant_id=blank)


def test_require_tenant_rejects_an_absent_tenant() -> None:
    """``None`` is the one falsy tenant the model still admits; the guard 401s it.

    Note the mechanism: ``effective_tenant_id`` RAISES rather than returning a
    falsy value, so the guard must catch — the old ``(… or "").strip()`` would
    now raise ``AttributeError`` on a ``uuid.UUID`` and 500 instead of 401.
    """
    user = User(sub="", preferred_username="u", tenant_id=None)
    with pytest.raises(HTTPException) as exc:
        deps.require_tenant(user)
    assert exc.value.status_code == 401


@pytest.mark.parametrize(
    "nil_spelling",
    [
        NIL,
        NIL.upper(),
        "00000000000000000000000000000000",
        "{00000000-0000-0000-0000-000000000000}",
        "urn:uuid:00000000-0000-0000-0000-000000000000",
    ],
)
def test_require_tenant_rejects_the_nil_uuid_itself(nil_spelling: str) -> None:
    """The nil UUID must be refused in EVERY spelling, not just the canonical one.

    A ``None``-only guard leaves the hole open: w2v_common's ``AuthMiddleware``
    substitutes ``DEFAULT_TENANT_ID`` (itself defaulting to the nil UUID) onto
    the ``User`` it builds on the legacy trusted-network path, so a real request
    can arrive already carrying the sentinel. All of these cast to the same nil
    value and would all reach billing as __unattributed__.
    """
    user = User(sub=nil_spelling, preferred_username="u", tenant_id=nil_spelling)
    with pytest.raises(HTTPException) as exc:
        deps.require_tenant(user)
    assert exc.value.status_code == 401


def test_effective_tenant_id_no_longer_falls_back_to_sub() -> None:
    """The sub-fallback is GONE upstream (#127), and that is the point.

    ``effective_tenant_id`` used to be ``self.tenant_id or self.sub`` — a
    SELF-NAMED tenant, i.e. a sentinel by another name: it minted a tenant that
    was never registered, and because ``sub`` is free-form it could not even be
    a UUID. It now raises. The guard still ends at 401 either way.
    """
    user = User(sub=NIL, preferred_username="u", tenant_id=None)
    with pytest.raises(MissingTenantError):
        _ = user.effective_tenant_id
    with pytest.raises(HTTPException) as exc:
        deps.require_tenant(user)
    assert exc.value.status_code == 401


async def test_get_current_user_refuses_an_unauthenticated_caller() -> None:
    """There is no anonymous principal: with no identity on state, it is 401.

    The deleted ``_ANONYMOUS_USER`` did not merely default a value — it
    manufactured an IDENTITY whose tenant was the sentinel and which self-granted
    ``roles=["admin"]``.
    """

    class _Req:
        state = type("S", (), {})()

    with pytest.raises(HTTPException) as exc:
        await deps.get_current_user(_Req())  # ty: ignore[invalid-argument-type]
    assert exc.value.status_code == 401


async def test_get_current_user_refuses_a_user_carrying_the_sentinel() -> None:
    """The upstream-injected nil UUID must not survive the dependency."""

    class _Req:
        state = type("S", (), {"user": User(sub=NIL, preferred_username="anonymous", tenant_id=NIL)})()

    with pytest.raises(HTTPException) as exc:
        await deps.get_current_user(_Req())  # ty: ignore[invalid-argument-type]
    assert exc.value.status_code == 401
