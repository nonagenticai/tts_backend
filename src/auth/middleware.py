"""License-token auth — consolidated onto w2v_common's shared middleware.

The vendored copy that used to live here (2s introspect timeout, no single-flight
lock) was the root of the 2026-06-27 narration-401 bug class: a concurrent
cold-cache burst of synth requests each stampeded the licenses endpoint until the
2s budget tripped → spurious 401. The canonical ``AuthMiddleware`` lives in
``w2v_common`` and, as of w2v-common 0.20.1, collapses that burst into one
introspection call (per-token single-flight) with a 10s timeout.

We re-export ``AuthMiddleware`` and ``PUBLIC_ENDPOINTS`` from w2v_common so
``main.py`` keeps its imports. ``PROTECTED_PREFIXES`` stays local — w2v_common
enforces protection via a public-denylist (everything not public is protected,
which for this service's routes — ``/api/tts/*`` and ``/me`` — is the same
outcome as the old allowlist); this set now only annotates the OpenAPI schema
(see ``main.custom_openapi``).
"""

from __future__ import annotations

from w2v_common.auth.middleware import PUBLIC_ENDPOINTS, AuthMiddleware

PROTECTED_PREFIXES: set[str] = {"/api", "/me"}

__all__ = ["AuthMiddleware", "PROTECTED_PREFIXES", "PUBLIC_ENDPOINTS"]
