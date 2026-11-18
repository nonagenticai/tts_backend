"""FastAPI dependencies for authentication.

Identity is populated on ``request.state.user`` by w2v_common's ``AuthMiddleware``
(re-exported from ``auth.middleware``); these helpers read it back.

Product-owner ruling (2026-08-26): a ``tenant_id`` may NEVER hold a sentinel.
All traffic on every message carries a real registered tenant forwarded from the
requestor. An absent tenant is an ERROR, never a substituted default.

This module used to declare ``UNATTRIBUTED_TENANT_ID`` (the nil UUID) and
substitute it via ``DEFAULT_TENANT_ID`` into a synthetic ``_ANONYMOUS_USER``.
That was the most dangerous spelling of the bug: the nil UUID is a VALID uuid,
so it passes every uuid cast and dirty-row check, and ``billing-service`` then
buckets it as unattributed and NEVER CHARGES IT. Real usage, silently unbilled.
Both the constants and the anonymous principal are gone; no sentinel is
reachable from this package.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Coroutine
from typing import Any, cast

from fastapi import Depends, HTTPException, Request, status
from w2v_common.tenant import MissingTenantError

from .models import User

# Kill switch for the auth MIDDLEWARE only. It no longer manufactures an
# identity: with auth disabled nothing populates ``request.state.user``, so
# every protected route refuses. It is not a way to become a tenant.
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Authentication required",
    headers={"WWW-Authenticate": "Bearer"},
)

_MISSING_TENANT = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Authenticated identity carries no registered tenant",
    headers={"WWW-Authenticate": "Bearer"},
)


def require_tenant(user: User) -> uuid.UUID:
    """Return the caller's real tenant id, or raise 401.

    Two refusals, both load bearing:

    * **No tenant at all.** From w2v-common 0.67.10, ``effective_tenant_id``
      RAISES ``MissingTenantError`` instead of returning a falsy value — the
      old ``self.tenant_id or self.sub`` fallback was a sentinel by another
      name. So the absence is caught as an exception here, not as a blank
      string; the previous ``(… or "").strip()`` spelling would now raise
      ``AttributeError`` on the ``uuid.UUID`` it returns, i.e. a 500 on every
      authenticated request from the guard whose whole job is a clean refusal.
    * **The nil UUID.** Still by value, and still not merely "un-substituted
      here": the nil UUID is a VALID uuid, so it survives every cast, and
      billing then buckets it as unattributed and NEVER CHARGES it. Spellings
      collapse for free now that the value arrives parsed — braced,
      urn-prefixed, upper-case and dash-less all become the same ``UUID(int=0)``,
      so the old string-shaped ``_is_nil_uuid`` helper is no longer needed.

    Non-UUID slugs need no branch either: ``User.tenant_id`` is typed
    ``uuid.UUID | None``, so pydantic rejects them at construction — the
    boundary, which is a better place than this guard.
    """
    try:
        tenant_id = user.effective_tenant_id
    except MissingTenantError:
        raise _MISSING_TENANT from None
    if tenant_id.int == 0:
        raise _MISSING_TENANT
    return tenant_id


async def get_current_user(request: Request) -> User:
    """Return the authenticated user carrying a real tenant, or raise 401."""
    user = getattr(request.state, "user", None)
    if not isinstance(user, User):
        raise _UNAUTHENTICATED

    # Fail loud rather than substitute. Every protected route depends on this,
    # and the metering emitter reads ``user.effective_tenant_id``, so a request
    # that cannot name its tenant never reaches the cost bus at all.
    require_tenant(user)
    return user


def resolve_app_id(request: Request) -> str:
    """Return the app the END USER is in, or ``""``.

    ``billing_cost_events.app_id`` is a **uuid** naming the CALLER's application
    — billing's ``docs/EMITTER_GUIDE.md`` defines it as "UUID of the app the user
    is in". It is emphatically NOT this service describing itself: ``service_key``
    already identifies the callee (``w2v-tts``), and if ``app_id`` repeated that,
    every distinct origin would collapse into one bucket while reading as 100%
    coverage — a defaulted identity is green on every attribution metric forever
    and wrong on every row.

    ONE rung, then the honest empty:

    1. ``request.state.license["app"]["id"]`` — the app on the licence
       introspection ``AuthMiddleware`` already performed for this request
       (``w2v_common.auth.middleware`` sets ``request.state.license = result``
       on the bearer-licence lane). Vouched by the licence service; a caller
       cannot lie about it, and it costs no extra I/O. The shape is the
       licences service's own: ``license-authz`` reads ``data["app"]["id"]``
       out of the same introspection body.
    2. ``""`` -> ``billing`` stores NULL (``cost_flusher._parse_uuid("")`` is
       ``None``).

    ⛔ There is deliberately NO ``X-App-Id`` rung here, and that is a departure
    from the kag-graph reference this function otherwise copies. Two measured
    reasons:

    * **Nothing sends it.** The only caller of ``/api/tts/synthesize`` in the
      estate is ``modality_gateway``, whose ``_FORWARD_REQUEST_HEADERS``
      (``src/services/tts_proxy.py``) forwards ``x-tenant-id``, ``x-license-id``,
      ``x-subject-id``, ``x-effective-tenant-id``, ``x-request-id``, ``accept``
      and ``content-type`` — and no ``x-app-id``. A rung no caller can reach
      buys zero attribution.
    * **A wrong value would be permanent.** ``billing_service``'s
      ``cost_flusher`` never overwrites an app_id the caller genuinely presented
      (``if row["app_id"] is not None: continue``), so an unverified header
      from any in-cluster hop would pin a misattribution that no later
      resolution can repair. An empty column, by contrast, lets billing DERIVE
      the app from the ``license_id`` this route already emits — the path that
      covered 85.6% of a 7-day window.

    ⛔ Do NOT "fix" an empty result by stamping this service's own registered
    ``licenses.app`` row. mlx-backend is the cautionary tale: 4,528 rows
    asserting an app that holds ZERO licences, unrepairable for the reason
    above.
    """
    licence = getattr(request.state, "license", None)
    if isinstance(licence, dict):
        app = licence.get("app")
        if isinstance(app, dict):
            app_id = app.get("id")
            # Guard the VALUE, not just the key: a JSON ``null`` would otherwise
            # stringify to the literal "None", which billing stores as real data.
            if app_id:
                return str(app_id).strip()
    return ""


def require_role(role: str) -> Callable[..., Coroutine[Any, Any, User]]:
    async def checker(user: User = Depends(get_current_user)) -> User:
        if role not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required",
            )
        return user

    return cast("Callable[..., Coroutine[Any, Any, User]]", checker)
