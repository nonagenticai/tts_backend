"""Health-endpoint smoke tests (no Apple-Silicon model load required)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(test_client: TestClient) -> None:
    r = test_client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "tts-backend"


def test_health_reports_degraded_with_no_provider_loaded(test_client: TestClient) -> None:
    """`/health` used to hardcode status="healthy" regardless of models_loaded —

    reporting healthy even with zero providers loaded, exactly the state a
    no-CI, manually-deployed live path most needs to surface (A/V quality
    wave 8, plan item 2e). It must now derive status the same way
    `/health/ready` already does.
    """
    r = test_client.get("/health")
    body = r.json()
    assert body["models_loaded"] == []
    assert body["status"] == "degraded"


def test_live_ok(test_client: TestClient) -> None:
    r = test_client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_root_info(test_client: TestClient) -> None:
    r = test_client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "tts-backend"


def test_synthesize_503_when_no_provider(test_client: TestClient) -> None:
    r = test_client.post("/api/tts/synthesize", json={"text": "hello"})
    assert r.status_code == 503


def test_metrics_endpoint_exposes_prometheus_format(test_client: TestClient) -> None:
    test_client.get("/health")
    r = test_client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "# HELP" in body
    assert "# TYPE" in body
    assert "http_request_duration_seconds" in body


def test_dropped_shadowing_health_route_404s(test_client: TestClient) -> None:
    """`/api/tts/health` was dropped; it must 404 now."""
    r = test_client.get("/api/tts/health")
    assert r.status_code == 404
