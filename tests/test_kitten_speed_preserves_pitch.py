"""Kitten speed must change TEMPO, not PITCH.

The old path resampled with np.interp, so speed=0.5 halved every frequency (an
octave down). A 440 Hz tone through speed 0.5 must still peak at ~440 Hz and last
~2x as long.
"""

import shutil

import numpy as np
import pytest

from services.tts_service import KittenProvider

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

_SR = 24000
_HZ = 440.0


class _TonePipeline:
    def generate(self, text, voice=None):
        t = np.arange(_SR) / _SR  # 1.0 s
        return (0.5 * np.sin(2 * np.pi * _HZ * t)).astype(np.float32)


def _dominant_hz(audio: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return float(np.fft.rfftfreq(len(audio), 1 / sr)[int(np.argmax(spectrum))])


@pytest.mark.parametrize("speed", [0.5, 2.0])
def test_kitten_speed_preserves_pitch(speed):
    p = KittenProvider()
    p._pipeline = _TonePipeline()
    result = p.synthesize("hello", voice=None, speed=speed)
    assert _dominant_hz(result.audio_pcm, result.sample_rate) == pytest.approx(_HZ, rel=0.03)
    assert result.duration_sec == pytest.approx(1.0 / speed, rel=0.05)


def test_kitten_speed_timeout_raises_not_resamples(monkeypatch):
    import subprocess

    from services import tts_service

    def _hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=kwargs.get("timeout"))

    monkeypatch.setattr(tts_service.subprocess, "run", _hang)
    p = KittenProvider()
    p._pipeline = _TonePipeline()
    with pytest.raises(RuntimeError, match="timed out"):
        p.synthesize("hello", voice=None, speed=0.5)
