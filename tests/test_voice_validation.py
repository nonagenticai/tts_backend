"""Voice-validation regression tests.

kokoro-onnx's ``create()`` raises a bare ``AssertionError`` on an unknown
voice. Before the fix this surfaced as a 500. The provider now validates the
voice against its loaded registry and raises ``ValueError`` → the route maps it
to a 400. Valid / omitted voices still synthesize.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.tts_service import KokoroProvider, SynthesisResult


class _FakeKokoro:
    def create(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        return np.zeros(2400, dtype=np.float32), 24000


def _loaded_kokoro() -> KokoroProvider:
    prov = KokoroProvider()
    prov._kokoro = _FakeKokoro()
    prov.voices = ["af_heart", "am_adam", "bf_emma"]
    return prov


def test_provider_unknown_voice_raises_valueerror() -> None:
    prov = _loaded_kokoro()
    with pytest.raises(ValueError, match="Unknown voice 'nope'"):
        prov.synthesize("hello", voice="nope", speed=1.0)


def test_provider_valid_voice_synthesizes() -> None:
    prov = _loaded_kokoro()
    result = prov.synthesize("hello", voice="am_adam", speed=1.0)
    assert isinstance(result, SynthesisResult)
    assert result.voice == "am_adam"


def test_provider_omitted_voice_uses_default() -> None:
    prov = _loaded_kokoro()
    prov.default_voice = "af_heart"
    result = prov.synthesize("hello", voice=None, speed=1.0)
    assert result.voice == "af_heart"


# --- Route-level: ValueError → 400, success → 200 ---


class _FakeService:
    default_provider = "kokoro"

    def __init__(self) -> None:
        prov = _loaded_kokoro()
        self.providers = {"kokoro": prov}

    def loaded_models(self) -> list[str]:
        return ["kokoro"]

    def synthesize(
        self, text: str, provider: Any, voice: str | None, speed: float, ssml: bool | None = None
    ) -> SynthesisResult:
        prov = self.providers["kokoro"]
        return prov.synthesize(text, voice, speed, ssml)


@pytest.fixture
def client_with_service(test_client: TestClient) -> TestClient:
    from main import app

    app.state.tts_service = _FakeService()
    return test_client


def test_route_unknown_voice_returns_400(client_with_service: TestClient) -> None:
    r = client_with_service.post("/api/tts/synthesize", json={"text": "hi", "voice": "nope"})
    assert r.status_code == 400
    assert "nope" in r.json()["error"]["message"]


def test_route_valid_voice_returns_200(client_with_service: TestClient) -> None:
    r = client_with_service.post("/api/tts/synthesize", json={"text": "hi", "voice": "am_adam"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"


def test_route_omitted_voice_returns_200(client_with_service: TestClient) -> None:
    r = client_with_service.post("/api/tts/synthesize", json={"text": "hi"})
    assert r.status_code == 200


def test_voices_endpoint_lists_real_ids(client_with_service: TestClient) -> None:
    r = client_with_service.get("/api/tts/voices")
    assert r.status_code == 200
    ids = {v["voice_id"] for v in r.json()["voices"]}
    assert {"af_heart", "am_adam", "bf_emma"} <= ids
