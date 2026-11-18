"""Realm-per-app P7: the outbound S2S token comes from ``secure-apps``.

tts-backend reads licences on ``licenses.test`` with a ``w2v-backend``
client-credentials JWT minted by ``w2v_common.auth.licenses_read_client``.
Realm ``portal`` becomes staff-only, so gitops sets
``KEYCLOAK_SERVICE_REALM=secure-apps`` while ``KEYCLOAK_REALM`` stays ``portal``
(it is ALSO the inbound validator's primary issuer realm). These tests pin
that split against the INSTALLED w2v_common: below 0.115.0 the env is inert
and the first test fails.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from w2v_common.auth import licenses_read_client as lrc

_KC = "https://auth.nonagentic.ai"


@pytest.fixture(autouse=True)
def _fresh_minter() -> Generator[None]:
    lrc.reset_service_token_minter()
    yield
    lrc.reset_service_token_minter()


@pytest.fixture
def _sa_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEYCLOAK_URL", _KC)
    monkeypatch.setenv("KEYCLOAK_REALM", "portal")
    monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "w2v-backend")
    monkeypatch.setenv("KEYCLOAK_CLIENT_SECRET", "test-not-a-secret")  # pragma: allowlist secret


@pytest.mark.usefixtures("_sa_env")
def test_service_realm_secure_apps_mints_from_secure_apps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEYCLOAK_SERVICE_REALM", "secure-apps")
    minter = lrc._get_service_token_minter()
    assert minter is not None
    assert minter._token_url == f"{_KC}/realms/secure-apps/protocol/openid-connect/token"


@pytest.mark.usefixtures("_sa_env")
def test_unset_service_realm_keeps_portal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: without the env the token still comes from KEYCLOAK_REALM."""
    monkeypatch.delenv("KEYCLOAK_SERVICE_REALM", raising=False)
    minter = lrc._get_service_token_minter()
    assert minter is not None
    assert minter._token_url == f"{_KC}/realms/portal/protocol/openid-connect/token"
