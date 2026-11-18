"""Pass-2 must actually run LINEAR loudnorm, not silently fall back to dynamic.

Live 2026-09-17 on mini2: `/health` counted `pass-2 normalization_type not linear`
on every clip. Measured on mini2's ffmpeg 8.1.1 with three real kokoro clips:
input ~-25.5 LUFS, LRA ~1.5 LU (well under loudnorm's default 7 — NOT the breaker),
true peak -3..-7 dBTP, so the +7.3..+7.8 dB linear gain would land peaks at
+0.4..+4.9 dBTP > -1.5. True-peak headroom broke linear on 3 of 3. The fix
lookahead-limits the peaks to the ceiling the linear gain can afford.
"""

from __future__ import annotations

import io
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import services.post_processor as pp

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
_SR = 24000

# clip0 as measured on mini2 (pass-1, production filter).
_CLIP0 = {"input_i": "-25.26", "input_tp": "-3.15", "input_lra": "1.50", "input_thresh": "-35.62"}


@pytest.fixture(autouse=True)
def _clean_latches() -> Any:
    pp._reset_log_latches()
    yield
    pp._reset_log_latches()


class TestPeakCeilingDerivation:
    def test_real_mini2_clip_needs_a_ceiling_of_tp_minus_offset_minus_margin(self) -> None:
        ceiling = pp._linear_peak_ceiling_db(_CLIP0, -18.0, -1.5)
        assert ceiling is not None
        # offset = -18 - -25.26 = 7.26 ; ceiling = -1.5 - 7.26 - 0.3
        assert ceiling == pytest.approx(-9.06, abs=1e-9)

    def test_headroom_already_available_adds_no_limiter(self) -> None:
        quiet_peaks = {**_CLIP0, "input_tp": "-12.00"}  # -12 + 7.26 = -4.74 <= -1.5
        assert pp._linear_peak_ceiling_db(quiet_peaks, -18.0, -1.5) is None

    def test_exactly_at_the_ceiling_adds_no_limiter(self) -> None:
        at = {**_CLIP0, "input_i": "-25.00", "input_tp": "-8.50"}  # -8.5 + 7.0 == -1.5 exactly in binary
        assert pp._linear_peak_ceiling_db(at, -18.0, -1.5) is None

    def test_ceiling_below_alimiter_range_is_left_to_dynamic(self) -> None:
        # offset 30 dB => ceiling -31.8 dBFS, below alimiter's 0.0625 (-24.08 dB) floor.
        very_quiet = {**_CLIP0, "input_i": "-48.00", "input_tp": "-10.00"}
        assert pp._linear_peak_ceiling_db(very_quiet, -18.0, -1.5) is None

    def test_non_finite_measurement_adds_no_limiter(self) -> None:
        assert pp._linear_peak_ceiling_db({**_CLIP0, "input_i": "-inf"}, -18.0, -1.5) is None


class TestPass2FilterString:
    def test_breaching_clip_gets_the_compensated_limiter_before_loudnorm(self) -> None:
        af = pp._pass2_filter(_CLIP0, -18.0, -1.5)
        assert af.startswith("aresample=192000,apad=pad_len=960,alimiter=limit=")
        assert ":level=false," in af  # auto-level would undo the gain staging
        assert "latency" not in af  # absent on ffmpeg 4.4 (dev/CI) — portability
        assert "atrim=start_sample=960,asetpts=N/SR/TB,loudnorm=" in af
        match = re.search(r"alimiter=limit=([0-9.]+)", af)
        assert match is not None
        limit = float(match.group(1))
        assert 20 * math.log10(limit) == pytest.approx(-9.06, abs=1e-3)
        # loudnorm is told the LIMITED peak, so its linear-eligibility check passes.
        assert ":measured_TP=-9.06:" in af
        assert ":measured_I=-25.26:" in af
        assert af.endswith(":linear=true:print_format=json")

    def test_clip_with_headroom_keeps_the_plain_linear_loudnorm(self) -> None:
        m = {**_CLIP0, "input_tp": "-12.00"}
        assert pp._pass2_filter(m, -18.0, -1.5) == (
            "loudnorm=I=-18.0:TP=-1.5:measured_I=-25.26:measured_TP=-12.00"
            ":measured_LRA=1.50:measured_thresh=-35.62:linear=true:print_format=json"
        )


def _peaky_speechlike_wav(seconds: float = 4.0) -> tuple[bytes, np.ndarray]:
    """Quiet noise phrases with silent gaps and short loud plosive-like bursts (FLOAT)."""
    rng = np.random.default_rng(7)
    n = int(_SR * seconds)
    pcm = np.zeros(n, dtype=np.float32)
    for start in np.arange(0.0, seconds, 0.8):
        a, b = int(start * _SR), int((start + 0.45) * _SR)
        pcm[a:b] = 0.05 * rng.standard_normal(b - a).astype(np.float32)
        burst = int(0.0015 * _SR)
        pcm[a + 200 : a + 200 + burst] += 0.35 * np.sin(2 * np.pi * 900 * np.arange(burst) / _SR).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, pcm, _SR, format="WAV", subtype="FLOAT")
    return buf.getvalue(), pcm


class TestRealFfmpegRunsLinear:
    @_needs_ffmpeg
    def test_peaky_clip_is_normalised_linearly_on_target_and_unshifted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        wav, src = _peaky_speechlike_wav()
        seen: list[dict[str, Any] | None] = []
        real_check = pp._check_normalization_type
        monkeypatch.setattr(pp, "_check_normalization_type", lambda post: seen.append(post) or real_check(post))

        # Precondition: this clip DOES breach TP under a plain linear gain.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "in.wav"
            p.write_bytes(wav)
            measured = pp._measure_json(p, -18.0, -1.5)
        assert measured is not None
        assert pp._linear_peak_ceiling_db(measured, -18.0, -1.5) is not None

        out, metrics = pp.premaster(wav)
        assert metrics.degraded_reason is None
        assert seen and seen[0] is not None
        assert seen[0]["normalization_type"] == "linear"
        assert metrics.integrated_lufs is not None
        # ±0.5, the same bar the real mini2 clips met (delivered -18.20/-18.22/-18.11).
        # Limiting removes the transients' share of loudness, which pass-1's measured_I
        # cannot see, so a clip whose bursts carry MORE loudness than real plosives
        # (0.7-amplitude bursts landed -18.6) is not a fair speech proxy; 0.35 is.
        assert abs(metrics.integrated_lufs + 18.0) <= 0.5
        assert metrics.true_peak_dbtp is not None
        assert metrics.true_peak_dbtp <= -1.5

        dst, _ = sf.read(io.BytesIO(out), dtype="float32", always_2d=False)
        assert len(dst) == len(src)  # frame count preserved (word timings)
        # No lookahead delay leaked through: cross-correlation peaks at lag 0.
        size = 2 * len(src)
        xc = np.fft.irfft(np.fft.rfft(dst, size) * np.conj(np.fft.rfft(src, size)), size)
        window = np.concatenate([xc[-200:], xc[:200]])
        assert int(np.argmax(window)) - 200 == 0

    @_needs_ffmpeg
    def test_limiter_graph_uses_only_builtin_filters(self) -> None:
        """Every filter in the graph must exist on the running ffmpeg (no optional lib)."""
        res = subprocess.run(  # noqa: S603
            ["ffmpeg", "-hide_banner", "-filters"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        available = {line.split()[1] for line in res.stdout.splitlines() if len(line.split()) > 2}
        names = {seg.split("=", 1)[0] for seg in pp._pass2_filter(_CLIP0, -18.0, -1.5).split(",")}
        assert names <= available, names - available


class TestSelfTestExercisesTheLimiterGraph:
    """/health's self-test must run the headroom-limiter graph, not only plain loudnorm.

    A build missing apad/alimiter/atrim/asetpts must fail the self-test loudly, the
    lesson of PR #64 (soxr): a self-test that never touches the risky argv passes on
    a build that fails open on every real clip.
    """

    def test_self_test_clip_breaches_tp_under_plain_linear_gain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "st.wav"
            p.write_bytes(pp._self_test_clip())
            if not _HAS_FFMPEG:
                pytest.skip("ffmpeg not on PATH")
            measured = pp._measure_json(p, -18.0, -1.5)
        assert measured is not None
        assert float(measured["input_lra"]) > 0.0  # LRA 0 makes loudnorm refuse linear
        assert pp._linear_peak_ceiling_db(measured, -18.0, -1.5) is not None

    @_needs_ffmpeg
    def test_self_test_passes_on_the_limiter_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        diags: list[dict[str, Any]] = []
        real = pp.premaster

        def _spy(*args: Any, **kwargs: Any) -> Any:
            result = real(*args, **kwargs)
            if kwargs.get("_diag") is not None:
                diags.append(dict(kwargs["_diag"]))
            return result

        monkeypatch.setattr(pp, "premaster", _spy)
        status = pp.run_premaster_self_test()
        assert status["self_test"] == "passed", status
        assert diags == [{"limited": True, "normalization_type": "linear"}]
        assert abs(status["integrated_lufs"] + 18.0) <= 0.5

    @_needs_ffmpeg
    def test_self_test_fails_when_the_limiter_graph_is_not_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _plain(measured: dict[str, Any], target_lufs: float, true_peak: float) -> str:
            return (
                f"loudnorm=I={target_lufs}:TP={true_peak}:measured_I={measured['input_i']}"
                f":measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}"
                f":measured_thresh={measured['input_thresh']}:linear=true:print_format=json"
            )

        monkeypatch.setattr(pp, "_pass2_filter", _plain)
        status = pp.run_premaster_self_test()
        assert status["self_test"] == "failed"
        assert "headroom limiter not exercised" in status["reasons"]
        assert "pass-2 not linear" in status["reasons"]

    @_needs_ffmpeg
    def test_self_test_fails_when_a_limiter_filter_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = pp._pass2_filter
        monkeypatch.setattr(
            pp, "_pass2_filter", lambda *a: real(*a).replace("alimiter=", "alimiter_not_in_this_build=")
        )
        status = pp.run_premaster_self_test()
        assert status["self_test"] == "failed"
        assert status["reasons"]
