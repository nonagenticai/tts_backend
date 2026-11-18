"""User model — re-exported from w2v_common (fleet-canonical identity).

The richer vendored model (realm_access/resource_access/has_role/…) was unused by
the routes, which only need ``sub``, ``preferred_username``, ``effective_tenant_id``
and ``roles``. Consolidating onto the shared model keeps the introspection->state
shape identical across the fleet so ``request.state.user`` is a single type.
"""

from __future__ import annotations

from w2v_common.auth.models import User

__all__ = ["User"]
