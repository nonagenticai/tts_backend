"""Route-level test: /api/tts/synthesize applies the pre-master + emits headers.

The `client` fixture authenticates as a REAL tenant via a `get_current_user`
override (there is no anonymous principal and no nil-UUID default any more), so
the route is reachable without standing up license introspection. We stub app.state.tts_service with a fake
provider that yields a real sine PCM (the conftest stubs kokoro_onnx/kittentts,
so the real TTSService would 503 with no loaded providers), then assert the
response carries the loudness headers and that master_lufs steers the target.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from services.tts_service import SynthesisResult

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")

_SR = 24000


def _sine_pcm(amplitude: float = 0.5, seconds: float = 3.0, sr: int = _SR, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


class _FakeService:
    """Minimal stand-in for TTSService that yields fixed sine PCM."""

    def __init__(self, pcm: np.ndarray, sr: int = _SR) -> None:
        self._pcm = pcm
        self._sr = sr

    def loaded_models(self) -> list[str]:
        return ["kokoro"]

    def synthesize(
        self,
        text: str,
        provider: str | None,
        voice: str | None,
        speed: float,
        ssml: bool | None = None,
    ) -> SynthesisResult:
        return SynthesisResult(
            audio_pcm=self._pcm,
            sample_rate=self._sr,
            duration_sec=float(len(self._pcm) / self._sr),
            provider="kokoro",
            voice=voice or "af_heart",
            model="fake",
        )


@pytest.fixture
def synth_client(test_client):  # noqa: ANN001, ANN201 — uses the conftest TestClient fixture
    test_client.app.state.tts_service = _FakeService(_sine_pcm(0.5))
    return test_client


def test_synthesize_emits_loudness_headers_near_default_target(synth_client) -> None:  # noqa: ANN001
    resp = synth_client.post("/api/tts/synthesize", json={"text": "Hello world."})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "audio/wav"
    assert "x-tts-lufs" in resp.headers
    lufs = float(resp.headers["x-tts-lufs"])
    assert abs(lufs - (-18.0)) <= 1.5, lufs
    tp = float(resp.headers["x-tts-truepeak-dbtp"])
    assert tp <= -1.5 + 0.1, tp


def test_master_lufs_shifts_measured_target(synth_client) -> None:  # noqa: ANN001
    resp = synth_client.post("/api/tts/synthesize", json={"text": "Hello world.", "master_lufs": -16})
    assert resp.status_code == 200, resp.text
    lufs = float(resp.headers["x-tts-lufs"])
    assert abs(lufs - (-16.0)) <= 1.5, lufs


# The orchestrator logged "tts synthesize response has no X-TTS-LUFS header" 66x/24h
# while the audio was in fact normalised — the post-premaster re-measurement was
# failing silently. Both response branches must carry the header, or that log
# line comes back.
@pytest.mark.parametrize(
    "accept,expected_content_type",
    [
        ("audio/wav", "audio/wav"),  # wav branch (tts_routes.py, final Response)
        ("application/json", "application/json"),  # JSON branch (SynthesizeResponse)
    ],
    ids=["wav_branch", "json_branch"],
)
def test_lufs_header_present_on_both_response_branches(synth_client, accept, expected_content_type) -> None:  # noqa: ANN001
    resp = synth_client.post(
        "/api/tts/synthesize",
        json={"text": "Hello world."},
        headers={"accept": accept},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith(expected_content_type)
    assert "x-tts-lufs" in resp.headers, dict(resp.headers)
    assert abs(float(resp.headers["x-tts-lufs"]) - (-18.0)) <= 1.5
    assert "x-tts-truepeak-dbtp" in resp.headers, dict(resp.headers)
    assert float(resp.headers["x-tts-truepeak-dbtp"]) <= -1.5 + 0.1


class TestTheServiceRefusesWhatItCannotDo:
    """Two 200s that contradicted the request, now answered honestly."""

    def test_a_sample_rate_the_engine_cannot_produce_is_refused_not_restamped(self, synth_client):
        """`soundfile.write` does not resample — it re-stamps the header.

        A 48 kHz request against the 24 kHz engine used to return chipmunk audio
        (double speed, half length) as a metered 200, with `duration_sec` computed
        from the sample count at the ENGINE rate so the invoice was wrong too.
        """
        resp = synth_client.post(
            "/api/tts/synthesize",
            json={"text": "Hello world.", "sample_rate": 48000},
        )
        assert resp.status_code == 400
        # The app wraps every error in `{"error": {...}}` (main._error_envelope), so a
        # bare top-level `detail` has not existed for some time. This assertion still
        # read `resp.json()["detail"]` and raised KeyError — it never reached the string
        # check it exists to make. It went unnoticed because this whole module is gated
        # on `shutil.which("ffmpeg")`, which the macOS CI runner does not have: the
        # module SKIPS in CI and only fails on a dev box. The route itself was always
        # correct (the 400 above passed); only the assertion was stale.
        message = resp.json()["error"]["message"]
        assert "does not" in message or "not supported" in message, message

    def test_the_engines_own_rate_is_still_accepted(self, synth_client):
        """Control: the guard must reject only what the engine cannot serve."""
        resp = synth_client.post(
            "/api/tts/synthesize",
            json={"text": "Hello world.", "sample_rate": _SR},
        )
        assert resp.status_code == 200


def test_the_premaster_receives_unquantized_float_audio(synth_client, monkeypatch) -> None:  # noqa: ANN001
    """The route must hand FLOAT to the premaster, and PCM_16 must come out.

    The old order was `encode_wav` (PCM_16) → `premaster` (loudnorm gain, then a
    second undithered `pcm_s16le`), so a quiet clip was truncated to 16 bits and
    only THEN boosted — lifting the quantization floor with the signal. Assert the
    wiring directly: whatever the route passes in must be a 32-bit float WAV.
    """
    import io

    import soundfile as sf

    import routers.tts_routes as routes

    seen: dict[str, str] = {}
    real = routes.premaster_float_wav

    def _spy(wav_bytes: bytes, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        seen["in_subtype"] = sf.info(io.BytesIO(wav_bytes)).subtype
        return real(wav_bytes, *args, **kwargs)

    monkeypatch.setattr(routes, "premaster_float_wav", _spy)
    resp = synth_client.post("/api/tts/synthesize", json={"text": "Hello world."})
    assert resp.status_code == 200, resp.text
    assert seen["in_subtype"] == "FLOAT", seen
    assert sf.info(io.BytesIO(resp.content)).subtype == "PCM_16"


class TestADegradedClipSaysSoToTheCaller:
    """A fail-open returns UN-NORMALISED audio with a 200 and no X-TTS-LUFS — which
    is exactly what an unparseable measurement also looks like. The orchestrator
    could not distinguish "normalised, measurement lost" from "never normalised".
    Now it can, on BOTH response branches.
    """

    @pytest.mark.parametrize(
        ("accept", "expected_content_type"),
        [("audio/wav", "audio/wav"), ("application/json", "application/json")],
    )
    def test_the_header_names_the_reason_on_both_branches(
        self,
        synth_client,  # noqa: ANN001
        monkeypatch: pytest.MonkeyPatch,
        accept: str,
        expected_content_type: str,
    ) -> None:
        import services.post_processor as pp

        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", False)
        resp = synth_client.post(
            "/api/tts/synthesize",
            json={"text": "Hello there."},
            headers={"accept": accept},
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith(expected_content_type)
        assert resp.headers.get("X-TTS-Premaster-Degraded") == "TTS_PREMASTER_ENABLED is off"
        # And the fail-open still honours the format contract.
        assert "X-TTS-LUFS" not in resp.headers

    def test_a_healthy_clip_carries_no_degraded_header(self, synth_client) -> None:  # noqa: ANN001
        """The discriminator half: a header that is always present signals nothing."""
        resp = synth_client.post("/api/tts/synthesize", json={"text": "Hello there."})
        assert resp.status_code == 200, resp.text
        assert "X-TTS-Premaster-Degraded" not in resp.headers
        assert "X-TTS-LUFS" in resp.headers
