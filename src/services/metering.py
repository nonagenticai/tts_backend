"""Cost metering via the centralized w2v_common cost-bus helper.

This module is a thin shim over :mod:`w2v_common.metering` (the fleet-canonical
cost-event emitter, v0.25.0+). It keeps the TTS-specific routing constants and
re-exports the Redis lifecycle + scheduling helpers so existing call sites
(``src/main.py``, ``src/routers/tts_routes.py``) keep working unchanged.

Why the shim delegates instead of marshalling its own fields:
- The field shape (EMITTER_GUIDE.md), the canonical stream resolution and the
  never-raise fire-and-forget contract all live in w2v_common and are tested
  there once for the whole fleet. w2v_common USED TO coerce a falsy tenant_id
  to the nil-UUID sentinel; as of 0.67.0 (#127) it does not — the sentinel is
  gone upstream and a tenant that is absent or not a UUID now RAISES. Both
  layers therefore refuse rather than substitute: ``schedule_cost_event``
  parses the tenant synchronously in the caller's frame (outside its own
  never-raises swallow), and ``auth.dependencies.require_tenant`` 401s a
  missing or nil-UUID tenant before any route body runs, so nothing
  unattributed can reach the cost bus from here.
- ``schedule_cost_event`` is GC-safe: it holds a strong reference to the
  background task so it can't be garbage-collected before it runs. This replaces
  the old ``asyncio.create_task(emit_cost_event(...))`` pattern, where the task
  handle was dropped and could be GC'd mid-flight, silently losing the event.

Both TTS providers (Kokoro / Kitten) run locally on Apple Silicon, so
``cost_usd`` is always ``0.0`` — the emission is usage telemetry (character
count, duration, provider attribution per tenant), not billing.
"""

from __future__ import annotations

from w2v_common.metering import (
    Funding,
    close_metering_redis,
    cost_event_key,
    emit_cost_event,
    init_metering_redis,
    resolve_stream,
    schedule_cost_event,
)

# TTS cost-event routing slugs (not credentials).
COST_SOURCE = "w2v-tts"
SERVICE_KEY = "w2v-tts"  # pragma: allowlist secret -- routing slug, not a credential

#: WHO FUNDS a synthesis, per provider (billing ``funding`` taxonomy, G5). Both
#: engines this service can load run on platform hardware (mini2). There is no
#: vendor and no tenant-key path here, so neither ``platform_external`` nor
#: ``tenant_byo`` has an arm. Mirrors tts_service #133's table minus its
#: cloud providers.
_PROVIDER_FUNDING: dict[str, Funding] = {
    "kokoro": "local_gpu",
    "kitten": "local_gpu",
}

#: ``event_key`` call kinds. Distinct kinds keep the two id spaces apart: an
#: ``Idempotency-Key`` that happens to equal some other request's
#: ``X-Request-ID`` must never collapse into the same charge.
EVENT_KIND_IDEMPOTENCY = "synthesize-idem"
EVENT_KIND_REQUEST = "synthesize-req"


class UnknownFundingProviderError(ValueError):
    """A provider with no declared funding reached the cost bus.

    Fail loud: billing stores ``funding`` verbatim, so a guessed value is a
    silent mis-classification on every row. A new provider must be added to
    ``_PROVIDER_FUNDING`` deliberately.
    """


def funding_for_provider(provider: str) -> Funding:
    """The billing ``funding`` for a synthesis served by ``provider``."""
    try:
        return _PROVIDER_FUNDING[provider]
    except KeyError:
        raise UnknownFundingProviderError(
            f"tts provider {provider!r} has no declared funding (add it to _PROVIDER_FUNDING)"
        ) from None


def synthesis_event_key(
    *,
    tenant_id: object,
    idempotency_key: str | None,
    request_id: str | None,
) -> str | None:
    """Deterministic ``event_key`` for ONE ``/synthesize`` request.

    Precedence:

    1. ``Idempotency-Key`` header -- the caller's explicit statement that a
       retry is the same logical request.
    2. ``X-Request-ID`` -- inbound when the caller sent one (modality_gateway
       forwards it), else the uuid7 ``RequestIdMiddleware`` minted for this
       request. Either way it is fixed for the life of the request, so any
       re-emit of THIS synthesis (the cost bus's own XADD retry after a timeout
       that actually landed) yields the same key and billing dedupes it.

    NOT a hash of the request body: identical payloads are distinct calls
    (``cost_event_key`` docstring / billing ``derive_event_key``), so a content
    key would collapse two genuine syntheses of the same sentence into one
    charge.

    ``tenant_id`` is part of the key because both ids are caller-supplied and
    billing dedupes GLOBALLY: two tenants sending ``Idempotency-Key: 1`` must
    not collapse into one event.

    Returns ``None`` only when there is no id at all (unreachable behind
    ``RequestIdMiddleware``); the cost bus then mints a uuid7 and counts it on
    ``w2v_cost_event_key_source_total{source="minted"}`` -- visible, not silent.
    """
    idem = (idempotency_key or "").strip()
    if idem:
        return cost_event_key(SERVICE_KEY, EVENT_KIND_IDEMPOTENCY, tenant_id, idem)
    rid = (request_id or "").strip()
    if rid:
        return cost_event_key(SERVICE_KEY, EVENT_KIND_REQUEST, tenant_id, rid)
    return None


__all__ = [
    "COST_SOURCE",
    "EVENT_KIND_IDEMPOTENCY",
    "EVENT_KIND_REQUEST",
    "SERVICE_KEY",
    "UnknownFundingProviderError",
    "close_metering_redis",
    "emit_cost_event",
    "funding_for_provider",
    "init_metering_redis",
    "resolve_stream",
    "schedule_cost_event",
    "synthesis_event_key",
]
