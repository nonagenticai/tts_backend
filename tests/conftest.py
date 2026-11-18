"""Shared fixtures + module-level test environment for tts_backend."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from typing import Any

import pytest

# --- Env setup (must happen BEFORE `main` is imported) ---
os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")
os.environ.setdefault("LICENSE_INTROSPECT_URL", "http://license-introspect.test/license/introspect")

# The GitHub Actions launchd-service runner (mini1/mini2/m1/m4) inherits a
# bare launchd PATH (/usr/bin:/bin:/usr/sbin:/sbin) — it never sources a login
# shell, so Homebrew's ffmpeg is on the box but invisible to the job. This is
# additive only (existing PATH wins via append-if-missing) so a dev shell with
# its own PATH is untouched; it exists solely so `shutil.which("ffmpeg")` and
# the bare `subprocess.run(["ffmpeg", ...])` calls in post_processor.py can
# find the real binary instead of the CI guard below flagging it missing.
for _brew_dir in ("/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin"):
    if _brew_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _brew_dir


# --- Stub Apple-Silicon-only deps ---
sys.modules.setdefault("kokoro_onnx", type(sys)("kokoro_onnx"))
sys.modules.setdefault("kittentts", type(sys)("kittentts"))
try:
    import soundfile  # noqa: F401
except ImportError:
    sys.modules["soundfile"] = type(sys)("soundfile")


# A real, registered-shaped tenant for route tests. Per the 2026-08-26
# product-owner ruling there is no anonymous principal and no sentinel: route
# tests must authenticate as a REAL tenant, exactly as production traffic does.
# (The old `ENABLE_AUTH=false -> anonymous admin carrying the nil UUID` path is
# deleted; `ENABLE_AUTH=false` now only skips the middleware, it grants nothing.)
TEST_TENANT_ID = "11111111-1111-4111-8111-111111111111"


def _authenticated_user() -> Any:
    from auth.models import User

    return User(
        sub=TEST_TENANT_ID,
        preferred_username="test-subject",
        tenant_id=TEST_TENANT_ID,
        roles=["admin"],
    )


@pytest.fixture(autouse=True)
def _fresh_gpu_gate() -> Any:
    """Every test starts with no GPU gate on the shared app.

    ``main.app`` is one module-level object for the whole session. A test that
    runs the lifespan leaves a SHUT-DOWN gate on ``app.state`` (correct in
    production: a closed gate must stay closed), and an asyncio semaphore binds
    to the first event loop that contends for it. Either would leak into every
    later test, so the gate is reset here and recreated lazily on first use.
    """
    from main import app

    app.state.gpu_gate = None
    yield
    app.state.gpu_gate = None


@pytest.fixture
def authenticated_app() -> Any:
    """The real app with `get_current_user` satisfied by a REAL tenant."""
    from auth.dependencies import get_current_user
    from main import app

    app.dependency_overrides[get_current_user] = _authenticated_user
    try:
        yield app
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def client(authenticated_app: Any) -> AsyncIterator[Any]:
    import httpx

    transport = httpx.ASGITransport(app=authenticated_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def test_client(authenticated_app: Any) -> Any:
    from fastapi.testclient import TestClient

    return TestClient(authenticated_app)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """In CI, a missing ffmpeg is a FAILURE, never a silent skip.

    Every real-ffmpeg test is `skipif(..., reason="ffmpeg not on PATH")` so a dev box
    without ffmpeg can still run the suite. On a CI runner that would turn the whole
    premaster/limiter surface into green skips — the exact blind spot behind PR #64.
    """
    import shutil

    if not (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) or shutil.which("ffmpeg"):
        return
    needing = [
        item.nodeid
        for item in items
        for mark in item.iter_markers("skipif")
        if mark.kwargs.get("reason") == "ffmpeg not on PATH"
    ]
    if needing:
        raise pytest.UsageError(
            f"CI run without ffmpeg on PATH: {len(needing)} premaster tests would silently skip (e.g. {needing[0]})"
        )
