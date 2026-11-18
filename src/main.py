"""TTS Backend — FastAPI entrypoint."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.exceptions import HTTPException as StarletteHTTPException
from w2v_common.admission import DisconnectProbeMiddleware, install_gpu_busy_handler

from auth import AuthMiddleware
from auth.dependencies import get_current_user
from auth.middleware import PROTECTED_PREFIXES, PUBLIC_ENDPOINTS
from auth.models import User
from middleware.logging import RequestLoggingMiddleware
from middleware.request_id import RequestIdMiddleware
from middleware.security_headers import SecurityHeadersMiddleware
from models import HealthResponse, PremasterHealth
from routers import tts_routes
from services.admission import build_gpu_gate
from services.metering import close_metering_redis, init_metering_redis
from services.post_processor import premaster_health, run_premaster_self_test
from services.tts_service import get_tts_service

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_NAME = os.getenv("SERVICE_NAME", "tts-backend")
VERSION = os.getenv("VERSION", "0.1.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
# Product-owner ruling (2026-08-26): a tenant_id may NEVER hold a sentinel. Every
# message carries a real registered tenant forwarded from the requestor; an absent
# tenant is an ERROR, never a substituted default. The nil-UUID `DEFAULT_TENANT_ID`
# substitution that used to live here is deleted — it cast cleanly as a uuid and so
# billing bucketed real usage as __unattributed__ and never charged it.
# Enforcement is `auth.dependencies.require_tenant` (401).
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false").lower() == "true"
docs_enabled = ENABLE_DOCS or ENVIRONMENT.lower() in {"development", "dev", "test"}
LEAK_EXCEPTION_DETAILS = os.getenv("LEAK_EXCEPTION_DETAILS", "").lower() == "true" or ENVIRONMENT.lower() in {
    "development",
    "dev",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    logger.info("Starting %s v%s (%s)", SERVICE_NAME, VERSION, ENVIRONMENT)
    svc = get_tts_service()
    try:
        await svc.initialize()
    except Exception:
        logger.exception("TTSService init partial failure — continuing")
    app.state.tts_service = svc
    # Built FRESH at boot so a malformed GPU_LANE_SYNTH_* value fails the start
    # loudly, and a restarted lifespan never inherits a gate a previous shutdown
    # closed.
    gate = build_gpu_gate()
    app.state.gpu_gate = gate
    app.state.metering_redis = await init_metering_redis(os.getenv("METERING_REDIS_URL", ""))
    # Run the REAL premaster ffmpeg invocations once, now, so a build that cannot
    # honour the argv fails loudly at start instead of silently on every clip
    # (PR #64). Never raises; a failure logs ERROR and degrades /health.
    await asyncio.to_thread(run_premaster_self_test)
    logger.info("%s ready, providers loaded: %s", SERVICE_NAME, svc.loaded_models() or "(none)")
    yield
    logger.info("Shutting down %s", SERVICE_NAME)
    # Stop admitting, give the in-flight synthesis a bounded chance to finish.
    if not await gate.shutdown(timeout=30.0):
        logger.warning("GPU gate shutdown timed out with synthesis still running")
    await close_metering_redis(getattr(app.state, "metering_redis", None))


app = FastAPI(
    title=SERVICE_NAME,
    description=(
        "Text-to-speech inference service for Apple Silicon (Kokoro + Kitten). "
        "Authentication is via license token in `X-Tenant-Id: Bearer <token>` "
        "with RFC1918/loopback bypass for in-cluster traffic."
    ),
    version=VERSION,
    lifespan=lifespan,
    redirect_slashes=False,
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    openapi_url="/openapi.json" if docs_enabled else None,
    swagger_ui_parameters={"persistAuthorization": True},
    generate_unique_id_function=lambda route: f"{route.tags[0] if route.tags else 'default'}_{route.name}",
    openapi_tags=[
        {"name": "Health", "description": "Liveness, readiness, dependency probes."},
        {"name": "TTS", "description": "Text-to-speech synthesis operations."},
        {"name": "Identity", "description": "Current authenticated identity context."},
    ],
)


# --- CORS ---
_cors_raw = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]
if ENVIRONMENT == "production" and not CORS_ORIGINS:
    raise RuntimeError(
        "CORS_ORIGINS env var is required in production. Set it on the launchd service env (see gitops-production)."
    )

# Middleware: RequestId → Auth → SecurityHeaders → RequestLogging → CORS (reverse-registered)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Processing-Time"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

if ENABLE_AUTH:
    app.add_middleware(AuthMiddleware)
    logger.info("Authentication middleware enabled")
else:
    logger.warning("Authentication middleware DISABLED (ENABLE_AUTH=false)")

app.add_middleware(RequestIdMiddleware)
# OUTERMOST, and it must stay the LAST add_middleware call. The three
# BaseHTTPMiddleware layers above hide the client's http.disconnect from
# `request.is_disconnected()`, so without this probe the `synth` lane's
# client-gone skip never fires and an abandoned queued job still synthesises.
app.add_middleware(DisconnectProbeMiddleware)

# GpuBusy -> 503 + Retry-After {"error":"gpu_busy",...}; ClientGone -> bodyless 499.
install_gpu_busy_handler(app)


def _install_metrics_routing_compat() -> None:
    """Make prometheus-fastapi-instrumentator tolerate fastapi>=0.137 routing.

    fastapi 0.137 wraps each `include_router(...)` result in an `_IncludedRouter`
    (a `BaseRoute` subclass that has no `.path`). prometheus-fastapi-instrumentator
    resolves a request's metric label by iterating `app.routes` and reading
    `route.path` on the first FULL match, which raises
    `AttributeError: '_IncludedRouter' object has no attribute 'path'` for every
    request that lands on an included router. Flatten those wrappers to their
    effective routes before the library's resolver runs. Behavior-preserving: the
    metric `handler` label remains the real (prefixed) route path, exactly as
    before the fastapi change.
    """
    global _METRICS_ROUTING_COMPAT_INSTALLED
    if _METRICS_ROUTING_COMPAT_INSTALLED:
        return
    try:
        import prometheus_fastapi_instrumentator.routing as _pfir
        from fastapi.routing import _IncludedRouter
    except ImportError, AttributeError:  # pragma: no cover - defensive; never break startup
        return

    orig_get_route_name = _pfir._get_route_name

    def _flatten(routes: Any) -> list[Any]:
        flat: list[Any] = []
        for route in routes:
            if isinstance(route, _IncludedRouter):
                for ctx in route.effective_route_contexts():
                    starlette_route = getattr(ctx, "starlette_route", None)
                    flat.append(starlette_route if starlette_route is not None else ctx.original_route)
            else:
                flat.append(route)
        return flat

    def _patched(scope: Any, routes: Any, *args: Any, **kwargs: Any) -> Any:
        # Forward whatever the installed library passes: 8.0.x calls this with a
        # third `route_name` argument, 8.1.0 dropped it. Adapting here keeps the
        # patch correct across both rather than pinning us to one signature.
        return orig_get_route_name(scope, _flatten(routes), *args, **kwargs)

    setattr(_pfir, "_get_route_name", _patched)  # noqa: B010 - dynamic library attr, keeps ty's typed-attr check satisfied
    _METRICS_ROUTING_COMPAT_INSTALLED = True


_METRICS_ROUTING_COMPAT_INSTALLED = False
_install_metrics_routing_compat()

Instrumentator(excluded_handlers=["/metrics", "/health.*"]).instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False, tags=["monitoring"]
)


# --- Error envelope helpers ---
def _error_envelope(
    code: str,
    message: str,
    http_status: int,
    *,
    request_id: str,
    details: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": code,
        "message": message,
        "status": http_status,
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if details is not None:
        payload["details"] = details
    return {"error": payload}


def _http_exception_to_response(exc: StarletteHTTPException, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_envelope(f"HTTP_{exc.status_code}", str(exc.detail), exc.status_code, request_id=request_id),
        headers=exc.headers,
    )


def _find_http_exception(exc: BaseException) -> HTTPException | None:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            found = _find_http_exception(sub)
            if found is not None:
                return found
    return None


def _request_id_from(request: Request) -> str:
    return request.state.request_id if hasattr(request.state, "request_id") else str(uuid.uuid7())


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _http_exception_to_response(exc, _request_id_from(request))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = _request_id_from(request)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_envelope(
            "VALIDATION_ERROR",
            "Request validation failed",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            request_id=request_id,
            details=str(exc.errors()) if LEAK_EXCEPTION_DETAILS else None,
        ),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id: str = _request_id_from(request)

    http_exc = _find_http_exception(exc)
    if http_exc is not None:
        return _http_exception_to_response(http_exc, request_id)

    logger.exception("unhandled_error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_envelope(
            "INTERNAL_SERVER_ERROR",
            "An unexpected error occurred",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=request_id,
            details=str(exc) if LEAK_EXCEPTION_DETAILS else None,
        ),
    )


# --- OpenAPI schema with per-path security ---
def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    components = openapi_schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes.setdefault(
        "BearerAuth",
        {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "License",
            "description": (
                "License token. Transport: `X-Tenant-Id: Bearer <token>` "
                "(not the Authorization header — that is reserved for caller-side auth)."
            ),
        },
    )

    parameters = components.setdefault("parameters", {})
    parameters.setdefault(
        "TenantIdHeader",
        {
            "name": "X-Tenant-Id",
            "in": "header",
            "required": True,
            "description": (
                "Registered tenant identifier, forwarded from the requestor. Required — "
                "there is no default and no sentinel; a request that cannot name its "
                "tenant is refused with 401."
            ),
            "schema": {"type": "string"},
        },
    )

    for path, path_item in openapi_schema.get("paths", {}).items():
        normalized_path = path.rstrip("/") or "/"
        requires_auth = (
            any(normalized_path == prefix or normalized_path.startswith(prefix + "/") for prefix in PROTECTED_PREFIXES)
            and normalized_path not in PUBLIC_ENDPOINTS
        )

        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue

            if requires_auth:
                operation.setdefault("security", [{"BearerAuth": []}])
            else:
                operation.pop("security", None)

            op_params = operation.setdefault("parameters", [])
            has_tenant = any(p.get("name", "").lower() == "x-tenant-id" and p.get("in") == "header" for p in op_params)
            if not has_tenant:
                op_params.append({"$ref": "#/components/parameters/TenantIdHeader"})

    app.openapi_schema = openapi_schema
    return app.openapi_schema


object.__setattr__(app, "openapi", custom_openapi)


# --- Routers ---
app.include_router(tts_routes.router)


# --- Service banner + health endpoints ---
@app.get(
    "/",
    operation_id="service_root",
    tags=["Health"],
    summary="Service banner",
)
async def root() -> dict[str, str]:
    """Root endpoint (public)."""
    return {"service": SERVICE_NAME, "version": VERSION, "docs": "/docs", "health": "/health"}


def _models_loaded() -> list[str]:
    svc = getattr(app.state, "tts_service", None)
    return svc.loaded_models() if svc else []


@app.get(
    "/health",
    response_model=HealthResponse,
    operation_id="health_check",
    tags=["Health"],
    summary="Liveness + identity probe",
)
async def health() -> HealthResponse:
    # Derived from what is actually loaded (same rule as /health/ready), not
    # hardcoded — this endpoint used to report "healthy" even with zero
    # providers loaded, which is exactly the state that should NOT read as
    # healthy on a no-CI, manually-deployed live path.
    return _health_response()


def _health_response() -> HealthResponse:
    loaded = _models_loaded()
    svc = getattr(app.state, "tts_service", None)
    providers = getattr(svc, "providers", {})
    engines = {name: getattr(provider, "engine", name) for name, provider in providers.items()}
    kokoro = providers.get("kokoro")
    premaster = PremasterHealth(**premaster_health())
    # A failed premaster self-test does not stop synthesis (the premaster fails
    # open), so it is `degraded`, not an error code: still 200, still serving.
    ok = bool(loaded) and premaster.self_test != "failed"
    return HealthResponse(
        status="healthy" if ok else "degraded",
        service=SERVICE_NAME,
        version=VERSION,
        device="unified",
        models_loaded=loaded,
        provider_engines=engines,
        word_timings_available=bool(kokoro and getattr(kokoro, "emits_word_timings", lambda: False)()),
        premaster=premaster,
    )


@app.get(
    "/health/ready",
    response_model=HealthResponse,
    operation_id="health_ready",
    tags=["Health"],
    summary="Readiness probe — reports degraded if no provider loaded",
)
async def ready() -> HealthResponse:
    return _health_response()


@app.get(
    "/health/live",
    response_model=HealthResponse,
    operation_id="health_live",
    tags=["Health"],
    summary="Liveness probe",
)
async def live() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=VERSION,
        device="unified",
        models_loaded=[],
    )


@app.get(
    "/me",
    operation_id="get_current_identity",
    tags=["Identity"],
    summary="Current authenticated identity",
)
async def me(user: User = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "user_id": user.sub,
        "username": user.preferred_username,
        "tenant_id": user.effective_tenant_id,
        "roles": user.roles,
    }
