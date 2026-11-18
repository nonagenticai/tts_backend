"""Tests for the per-clip -18 LUFS pre-master (services/post_processor.py).

ffmpeg is present on the deploy hosts + the macOS CI runner, so the real
two-pass loudnorm path is exercised directly (guarded with a skipif for
portability). The fail-open paths (kill-switch off, ffmpeg absent) run
everywhere — they must never shell out.
"""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import services.post_processor as pp
from services.post_processor import AudioMetrics, measure_audio, premaster

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_log_latches() -> None:
    """WARNINGs are latched per-reason for the PROCESS, so a test that asserts on a
    log line only sees it if no earlier test already burned that reason."""
    pp._reset_log_latches()


def _sine_wav(amplitude: float, seconds: float = 3.0, sr: int = _SR, freq: float = 440.0) -> bytes:
    """Render a mono sine WAV (PCM_16) at the given peak amplitude (0..1)."""
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    pcm = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _integrated_lufs(wav_bytes: bytes) -> float | None:
    return measure_audio(wav_bytes).integrated_lufs


class TestMeasureAudio:
    @_needs_ffmpeg
    def test_measures_a_sine(self) -> None:
        m = measure_audio(_sine_wav(0.5))
        assert m.integrated_lufs is not None
        assert m.true_peak_dbtp is not None
        assert m.duration_seconds is not None
        # A -6 dBFS sine is loud; sanity-bound the reading well away from silence.
        assert -20.0 < m.integrated_lufs < 0.0

    def test_garbage_bytes_fail_open_to_none(self) -> None:
        """Non-audio input must not raise; loudness fields degrade to None."""
        m = measure_audio(b"not a wav file")
        assert isinstance(m, AudioMetrics)
        assert m.integrated_lufs is None
        assert m.true_peak_dbtp is None

    def test_ffmpeg_binary_absent_fails_open_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing ffmpeg binary (FileNotFoundError from subprocess.run) must
        not raise — measure_audio()'s docstring promises loudness fields stay
        None on any ffmpeg failure. Deterministic regardless of whether ffmpeg
        is installed on the test host."""

        def _raise(*_args: object, **_kwargs: object) -> object:
            raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

        monkeypatch.setattr(pp.subprocess, "run", _raise)
        m = measure_audio(_sine_wav(0.5))
        assert isinstance(m, AudioMetrics)
        assert m.integrated_lufs is None
        assert m.true_peak_dbtp is None

    def test_ffmpeg_timeout_fails_open_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An ffmpeg hang (subprocess.TimeoutExpired) must also fail open."""

        def _raise(*_args: object, **_kwargs: object) -> object:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=60)

        monkeypatch.setattr(pp.subprocess, "run", _raise)
        m = measure_audio(_sine_wav(0.5))
        assert isinstance(m, AudioMetrics)
        assert m.integrated_lufs is None


class TestPremasterNormalizes:
    @_needs_ffmpeg
    @pytest.mark.parametrize("amplitude", [0.5, 0.0316])  # ~-6 dBFS (loud) and ~-30 dBFS (quiet)
    def test_lands_near_minus_18_with_true_peak_ceiling(self, amplitude: float) -> None:
        out, metrics = premaster(_sine_wav(amplitude))
        assert out != _sine_wav(amplitude) or True  # bytes change; loudness is the real assertion
        assert metrics.integrated_lufs is not None
        assert abs(metrics.integrated_lufs - (-18.0)) <= 1.5, metrics.integrated_lufs
        assert metrics.true_peak_dbtp is not None
        assert metrics.true_peak_dbtp <= -1.5 + 0.1, metrics.true_peak_dbtp  # small TP tolerance

    @_needs_ffmpeg
    def test_custom_target_shifts_loudness(self) -> None:
        _out, metrics = premaster(_sine_wav(0.5), target_lufs=-16.0)
        assert metrics.integrated_lufs is not None
        assert abs(metrics.integrated_lufs - (-16.0)) <= 1.5, metrics.integrated_lufs


class TestPremasterFailOpen:
    def test_kill_switch_off_returns_input_bytes_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", False)
        src = _sine_wav(0.5)
        out, metrics = premaster(src)
        assert out == src  # byte-identical passthrough
        # Measurement fields empty → route omits the LUFS headers; `degraded_reason`
        # set → route emits X-TTS-Premaster-Degraded so the caller can tell an
        # un-normalised clip from a clip whose measurement merely went missing.
        assert metrics == AudioMetrics(degraded_reason="TTS_PREMASTER_ENABLED is off")

    def test_ffmpeg_absent_returns_input_bytes_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
        monkeypatch.setattr(pp.shutil, "which", lambda _name: None)
        src = _sine_wav(0.5)
        out, metrics = premaster(src)
        assert out == src
        assert metrics.integrated_lufs is None
        assert metrics.true_peak_dbtp is None
        # Name the rung, not just "something degraded" — this is the reason the
        # route puts in X-TTS-Premaster-Degraded.
        assert metrics == AudioMetrics(degraded_reason="ffmpeg not found on PATH")

    def test_pass1_measurement_failure_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If pass-1 measurement can't parse, return input unchanged (no pass-2)."""
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
        # The fail-open ladder is ordered: kill-switch → ffmpeg-on-PATH → pass-1.
        # Without this stub the injected pass-1 failure is unreachable on a host
        # with no ffmpeg (the CI runner), and the EARLIER rung answers instead —
        # correctly, which is why it read as a production bug and is not one.
        monkeypatch.setattr(pp.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(pp, "_measure_json", lambda *_a, **_k: None)
        src = _sine_wav(0.5)
        out, metrics = premaster(src)
        assert out == src
        assert metrics == AudioMetrics(degraded_reason="pass-1 loudnorm measurement failed")


class TestPremasterMeasuresFromPassTwo:
    """The post-normalisation LUFS must come out of pass-2's OWN json block.

    Regression guard for the live defect: the third ffmpeg run that re-measured
    the output had five silent `return None` paths, so `premaster` could return
    correctly-normalised bytes with `integrated_lufs=None`, and the route then
    dropped `X-TTS-LUFS` with nothing logged anywhere.
    """

    @_needs_ffmpeg
    def test_runs_exactly_two_ffmpeg_processes_and_still_reports_lufs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_run = pp.subprocess.run
        calls: list[list[str]] = []

        def _counting_run(cmd, *args, **kwargs):  # noqa: ANN001, ANN202
            calls.append(list(cmd))
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(pp.subprocess, "run", _counting_run)
        _out, metrics = premaster(_sine_wav(0.5))

        # The header depends on this being non-None — that IS the bug.
        assert metrics.integrated_lufs is not None
        assert metrics.true_peak_dbtp is not None
        assert metrics.duration_seconds is not None
        # Two passes only: the third re-measurement run is gone.
        assert len(calls) == 2, [c[:6] for c in calls]
        # ...and pass-2 is the one that writes the output file (not `-f null`).
        assert "null" not in calls[1]
        assert any("linear=true" in tok for tok in calls[1])
        assert any("print_format=json" in tok for tok in calls[1])

    @_needs_ffmpeg
    def test_pass2_json_matches_an_independent_remeasure(self) -> None:
        """Pass-2's `output_i` must agree with actually re-measuring the bytes."""
        out, metrics = premaster(_sine_wav(0.5))
        independent = _integrated_lufs(out)
        assert metrics.integrated_lufs is not None
        assert independent is not None
        assert abs(metrics.integrated_lufs - independent) <= 0.5, (metrics.integrated_lufs, independent)

    @_needs_ffmpeg
    def test_unparseable_pass2_json_degrades_to_remeasure_and_logs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If pass-2's stderr carries no JSON we must fall back to the old
        re-measure and SAY SO — never silently return metrics-free audio."""
        real_run = pp.subprocess.run
        seen = 0

        def _strip_json_from_pass2(cmd, *args, **kwargs):  # noqa: ANN001, ANN202
            nonlocal seen
            result = real_run(cmd, *args, **kwargs)
            seen += 1
            if seen == 2:  # pass-2: blank out the json block it printed
                result.stderr = "loudnorm ran but printed no measurements\n"
            return result

        monkeypatch.setattr(pp.subprocess, "run", _strip_json_from_pass2)
        with caplog.at_level("WARNING", logger=pp.logger.name):
            _out, metrics = premaster(_sine_wav(0.5))

        # Degrade, don't regress: the fallback re-measure still yields a value.
        assert metrics.integrated_lufs is not None, "fallback re-measure must still produce a LUFS"
        assert abs(metrics.integrated_lufs - (-18.0)) <= 1.5, metrics.integrated_lufs
        # And the degradation is audible in the log.
        messages = " | ".join(r.getMessage() for r in caplog.records)
        assert "no JSON block" in messages, messages
        assert "re-measuring the output clip" in messages, messages


class TestMeasureJsonLogsEveryFailurePath:
    """All five `return None` paths in `_measure_json` must log at WARNING.

    They were silent, which is why the missing header had no explanation.
    """

    def _run(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fake_run) -> str:  # noqa: ANN001
        monkeypatch.setattr(pp.subprocess, "run", fake_run)
        with caplog.at_level("WARNING", logger=pp.logger.name):
            assert measure_audio(_sine_wav(0.5)).integrated_lufs is None
        return " | ".join(r.getMessage() for r in caplog.records)

    def test_ffmpeg_not_runnable_logs(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        def _raise(*_a: object, **_k: object) -> object:
            raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

        assert "could not be run" in self._run(monkeypatch, caplog, _raise)

    def test_timeout_logs(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        def _raise(*_a: object, **_k: object) -> object:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=60)

        assert "could not be run" in self._run(monkeypatch, caplog, _raise)

    def test_nonzero_exit_logs(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        def _fail(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=["ffmpeg"], returncode=1, stdout="", stderr="boom")

        assert "exited 1" in self._run(monkeypatch, caplog, _fail)

    def test_no_json_block_logs(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        def _quiet(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout="", stderr="no braces here")

        assert "no JSON block" in self._run(monkeypatch, caplog, _quiet)

    def test_malformed_json_logs(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        def _garbage(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout="", stderr="{not json,}")

        assert "malformed JSON" in self._run(monkeypatch, caplog, _garbage)

    def test_non_object_json_logs(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """The `not isinstance(parsed, dict)` guard is defensive — the slice always
        starts with `{`, so json.loads can only yield a dict or raise. Drive it
        directly rather than pretending ffmpeg can produce it."""
        monkeypatch.setattr(pp.json, "loads", lambda _s: [1, 2, 3])
        with caplog.at_level("WARNING", logger=pp.logger.name):
            assert pp._parse_loudnorm_stderr('{"input_i": "-18.0"}', "unit") is None
        assert "not an object" in " | ".join(r.getMessage() for r in caplog.records)


class TestWarningVolumeIsBounded:
    """Failure logging must be bounded by the number of REASONS, not the request
    rate.

    The observability fix that made these paths visible made them log per CALL.
    At ~53k synth calls/day a systemic ffmpeg fault would emit ~53k WARNINGs —
    with a full traceback on the OSError/TimeoutExpired path. That is an
    operational regression: the old code logged once per process (single
    fail-open latch). These tests pin "visible AND bounded".
    """

    @staticmethod
    def _raise_enoent(*_a: object, **_k: object) -> object:
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    def test_thousand_consecutive_failures_produce_a_bounded_log_count(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
        monkeypatch.setattr(pp.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(pp.subprocess, "run", self._raise_enoent)

        src = _sine_wav(0.5, seconds=0.5)
        with caplog.at_level("WARNING", logger=pp.logger.name):
            for _ in range(1000):
                out, metrics = premaster(src)
        # Fail-open semantics survive the latch: every call still passes through.
        assert out == src
        assert metrics == AudioMetrics(degraded_reason="pass-1 loudnorm measurement failed")
        # Two distinct reasons fire here (measurement-pass ffmpeg-unrunnable, and
        # premaster's pass-1 fail-open), each exactly once. NOT 1000, and not 1001.
        assert len(caplog.records) == 2, [r.getMessage() for r in caplog.records]
        messages = " | ".join(r.getMessage() for r in caplog.records)
        assert "ffmpeg could not be run" in messages, messages
        assert "pass-1 loudnorm measurement failed" in messages, messages

    def test_thousand_failures_produce_at_most_one_traceback(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """exc_info is the expensive part: 53k tracebacks/day is the regression."""
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
        monkeypatch.setattr(pp.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(pp.subprocess, "run", self._raise_enoent)
        with caplog.at_level("WARNING", logger=pp.logger.name):
            for _ in range(1000):
                premaster(_sine_wav(0.5, seconds=0.5))
        # Truthiness, not `is not None`: logging stores a falsy `exc_info=False`
        # on the record verbatim, so `is not None` counts records with NO traceback.
        with_traceback = [r for r in caplog.records if r.exc_info]
        assert len(with_traceback) == 1, len(with_traceback)

    def test_pass2_exit_flood_with_varying_stderr_is_still_bounded(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The fail-open reason string embeds the exit code and stderr tail, which
        differ per clip. If the latch were keyed on that string instead of a stable
        key, 1000 failures would be 1000 distinct "reasons" and the latch would be
        no latch at all. No ffmpeg needed: both passes are faked."""
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
        monkeypatch.setattr(pp.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        pass1_json = '{"input_i":"-30.00","input_tp":"-6.00","input_lra":"1.00","input_thresh":"-40.00"}'
        seen = 0

        def _fake_run(cmd, *_a: object, **_k: object) -> subprocess.CompletedProcess[str]:  # noqa: ANN001
            nonlocal seen
            if "null" in cmd:  # pass-1 measurement: succeed
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr=pass1_json)
            seen += 1  # pass-2: fail, with a DIFFERENT stderr every time
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr=f"Conversion failed! (attempt {seen}, addr 0x{seen:x})"
            )

        monkeypatch.setattr(pp.subprocess, "run", _fake_run)
        src = _sine_wav(0.5, seconds=0.5)
        with caplog.at_level("WARNING", logger=pp.logger.name):
            for _ in range(1000):
                out, metrics = premaster(src)
        assert seen == 1000, seen  # every call really did reach pass-2
        assert out == src
        # The STABLE latch key, not the exit-code-bearing reason — the caller must be
        # able to group by this instead of seeing a new value every request.
        assert metrics == AudioMetrics(degraded_reason="pass-2 loudnorm nonzero exit")
        assert len(caplog.records) == 1, len(caplog.records)
        assert "pass-2 loudnorm exit 1" in caplog.records[0].getMessage()

    def test_a_second_distinct_reason_still_gets_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Per-REASON, not process-wide: the old single boolean latch silenced every
        cause after the first, so a second, different fault was invisible."""
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
        monkeypatch.setattr(pp.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        with caplog.at_level("WARNING", logger=pp.logger.name):
            monkeypatch.setattr(pp.subprocess, "run", self._raise_enoent)
            premaster(_sine_wav(0.5, seconds=0.5))
            first = len(caplog.records)

            def _nonzero(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(args=["ffmpeg"], returncode=1, stdout="", stderr="boom")

            monkeypatch.setattr(pp.subprocess, "run", _nonzero)
            premaster(_sine_wav(0.5, seconds=0.5))

        assert len(caplog.records) > first, "a second, different failure reason must not be swallowed"
        assert "exited 1" in " | ".join(r.getMessage() for r in caplog.records)


class TestUnmeasurableClipsAreNotNormalised:
    """Clips with no finite integrated loudness must be named, not mis-reported.

    Measured against real ffmpeg: for digital silence, a sub-0.4s clip (shorter
    than the EBU R128 gating block), or a ~-70 dBFS clip (below loudnorm's
    absolute gate), pass-1 returns `input_i: "-inf"` WITH exit 0. That string was
    interpolated into pass-2's filter, pass-2 then exited 1 ("Failed to inject
    frame into filter network: Result too large") and the clip surfaced as a
    generic pass-2 failure — or, after the first one latched, as nothing at all.
    """

    @_needs_ffmpeg
    @pytest.mark.parametrize(
        "src",
        [
            pytest.param(_sine_wav(0.0, seconds=3.0), id="digital_silence"),
            pytest.param(_sine_wav(0.5, seconds=0.2), id="clip_shorter_than_gating_block"),
            pytest.param(_sine_wav(0.0003, seconds=3.0), id="near_silent_minus_70_dbfs"),
        ],
    )
    def test_passes_through_with_a_named_reason_and_no_second_ffmpeg_run(
        self, src: bytes, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_run = pp.subprocess.run
        calls: list[list[str]] = []

        def _counting_run(cmd, *args, **kwargs):  # noqa: ANN001, ANN202
            calls.append(list(cmd))
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(pp.subprocess, "run", _counting_run)
        with caplog.at_level("WARNING", logger=pp.logger.name):
            out, metrics = premaster(src)

        assert out == src, "unmeasurable audio must be returned byte-identical"
        assert metrics == AudioMetrics(degraded_reason="pass-1 non-finite measurement"), metrics
        # Bailed BEFORE pass-2: we no longer feed `measured_I=-inf` to a filter
        # that cannot survive it.
        assert len(calls) == 1, [c[:6] for c in calls]
        message = " | ".join(r.getMessage() for r in caplog.records)
        assert "no measurable integrated loudness" in message, message
        assert "not loudness-normalisable" in message, message
        # ...and it is NOT reported as a pass-2 crash any more.
        assert "pass-2" not in message, message

    @_needs_ffmpeg
    def test_the_three_unmeasurable_shapes_share_one_latched_line(self, caplog: pytest.LogCaptureFixture) -> None:
        """One reason ⇒ one line, however many silent clips arrive."""
        with caplog.at_level("WARNING", logger=pp.logger.name):
            for src in (_sine_wav(0.0, 3.0), _sine_wav(0.5, 0.2), _sine_wav(0.0003, 3.0)) * 5:
                premaster(src)
        assert len(caplog.records) == 1, [r.getMessage() for r in caplog.records]


class TestNonFiniteNeverReachesTheHeader:
    """`-inf` must never be formatted into `X-TTS-LUFS`.

    `float("-inf")` coerces fine, so the `is None` guard in premaster did not stop
    it; `measure_audio()` returned `integrated_lufs=-inf` for every silent clip and
    `tts_routes.py` does `f"{value:.2f}"` → the literal header `X-TTS-LUFS: -inf`,
    which the orchestrator parses as a real float measurement.
    """

    @pytest.mark.parametrize("raw", ["-inf", "inf", "nan", "-Infinity", float("-inf"), float("nan")])
    def test_to_float_rejects_non_finite(self, raw: object) -> None:
        assert pp._to_float(raw) is None, raw

    @pytest.mark.parametrize("raw,expected", [("-18.5", -18.5), (-1.5, -1.5), ("0", 0.0)])
    def test_to_float_still_accepts_finite(self, raw: object, expected: float) -> None:
        assert pp._to_float(raw) == expected

    @_needs_ffmpeg
    @pytest.mark.parametrize(
        "src",
        [
            pytest.param(_sine_wav(0.0, seconds=3.0), id="digital_silence"),
            pytest.param(_sine_wav(0.5, seconds=0.2), id="clip_shorter_than_gating_block"),
            pytest.param(_sine_wav(0.0003, seconds=3.0), id="near_silent_minus_70_dbfs"),
        ],
    )
    def test_measure_audio_reports_none_not_negative_infinity(self, src: bytes) -> None:
        m = measure_audio(src)
        assert m.integrated_lufs is None, m.integrated_lufs
        # Whatever survives must be formattable as a real number for the header.
        for value in (m.integrated_lufs, m.true_peak_dbtp, m.loudness_range_lu):
            assert value is None or math.isfinite(value), value
        # Duration is soundfile-derived and still reported.
        assert m.duration_seconds is not None


def _float_wav(amplitude: float, seconds: float = 3.0, sr: int = _SR, freq: float = 440.0) -> bytes:
    """Render a mono sine WAV as 32-bit FLOAT — what the route now hands premaster."""
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    pcm = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


class TestPremasterFloatWav:
    """The premaster now runs on UN-quantized audio; PCM_16 happens once, last.

    The old order quantized to int16 in the route and applied loudnorm's gain
    afterwards, so a quiet clip's 16-bit quantization floor was boosted along with
    the signal (grit/hiss in the pauses) — and pass-2's `-c:a pcm_s16le` quantized
    a second, undithered time on top of it.
    """

    @_needs_ffmpeg
    def test_a_float_input_still_yields_a_pcm16_wav(self) -> None:
        out, metrics = pp.premaster_float_wav(_float_wav(0.5))
        assert sf.info(io.BytesIO(out)).subtype == "PCM_16"
        assert sf.info(io.BytesIO(out)).samplerate == _SR
        assert metrics.integrated_lufs is not None
        assert abs(metrics.integrated_lufs - (-18.0)) <= 1.5, metrics.integrated_lufs

    @_needs_ffmpeg
    def test_a_quiet_clip_keeps_a_lower_noise_floor_than_the_quantize_first_order(self) -> None:
        """The whole point of the reordering, measured.

        A -60 dBFS clip quantized to int16 FIRST and boosted SECOND carries a much
        higher noise floor than the same clip boosted at float and quantized LAST.
        Compared via loudness RANGE is unreliable, so compare the residual against
        an ideal sine of the same measured level: quantize-first has ~40 dB more
        error energy.
        """
        quiet = 10 ** (-60.0 / 20.0)
        float_out, _ = pp.premaster_float_wav(_float_wav(quiet))
        int_out, _ = premaster(_sine_wav(quiet))

        def _noise_floor_db(wav: bytes) -> float:
            data, _sr = sf.read(io.BytesIO(wav), dtype="float64", always_2d=False)
            spec = np.abs(np.fft.rfft(data * np.hanning(len(data))))
            peak_bin = int(np.argmax(spec))
            # Everything that is not the 440 Hz fundamental (±3 bins) is error.
            mask = np.ones(len(spec), dtype=bool)
            mask[max(0, peak_bin - 3) : peak_bin + 4] = False
            return 20.0 * math.log10(float(np.sqrt((spec[mask] ** 2).sum()) / spec[peak_bin]) + 1e-30)

        float_floor = _noise_floor_db(float_out)
        int_floor = _noise_floor_db(int_out)
        assert float_floor < int_floor - 10.0, (float_floor, int_floor)

    def test_fail_open_hands_back_pcm16_not_the_float_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The response contract must not depend on whether the premaster ran.

        `premaster`'s fail-open returns its INPUT bytes — which, on this path, are a
        32-bit float WAV. Callers of /api/tts/synthesize have always received PCM_16.
        """
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", False)
        src = _float_wav(0.5, seconds=0.5)
        out, metrics = pp.premaster_float_wav(src)
        assert sf.info(io.BytesIO(src)).subtype == "FLOAT"
        assert sf.info(io.BytesIO(out)).subtype == "PCM_16"
        # Not `== AudioMetrics()` any more: a fail-open now NAMES itself, so the
        # measurement fields stay empty but `degraded_reason` is set. Asserting the
        # whole dataclass is empty would forbid exactly the signal this path needs.
        assert metrics == AudioMetrics(degraded_reason="TTS_PREMASTER_ENABLED is off")
        # Same audio, only re-quantized: level preserved to within a bit.
        a, _ = sf.read(io.BytesIO(src), dtype="float32", always_2d=False)
        b, _ = sf.read(io.BytesIO(out), dtype="float32", always_2d=False)
        assert len(a) == len(b)
        assert float(np.abs(a - b).max()) < 1e-3


class TestNormalizationTypeIsReadBack:
    """`linear=true` is a REQUEST, not a guarantee.

    ffmpeg silently reverts to time-varying (`dynamic`) gain when the linear gain
    would breach the true peak, and nothing read the answer back — while
    `tts_routes` documents "linear=true loudnorm is pure gain" as a fact. Reporting
    only: the request itself is deliberately unchanged.
    """

    def test_dynamic_is_warned(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            observed = pp._check_normalization_type({"normalization_type": "dynamic", "output_i": "-18.0"})
        assert observed == "dynamic"
        assert "normalization_type" in caplog.text
        assert "dynamic" in caplog.text

    def test_linear_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            observed = pp._check_normalization_type({"normalization_type": "linear", "output_i": "-18.0"})
        assert observed == "linear"
        assert caplog.text == ""

    @pytest.mark.parametrize("post", [None, {}, {"normalization_type": 7}])
    def test_a_missing_or_non_string_type_is_not_a_warning(
        self, post: dict[str, object] | None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Absence is not evidence of a dynamic pass — do not manufacture a warning."""
        with caplog.at_level("WARNING"):
            assert pp._check_normalization_type(post) is None
        assert caplog.text == ""

    @_needs_ffmpeg
    def test_a_real_pass_reports_a_type_at_all(self) -> None:
        """Guard against the key being renamed/absent in the installed ffmpeg —
        which would make the warning above permanently unreachable."""
        out, _m = premaster(_sine_wav(0.5))
        assert out
        # Re-run pass 2's exact shape and confirm the key exists in its JSON.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.wav"
            dst = Path(tmp) / "out.wav"
            src.write_bytes(_sine_wav(0.5))
            measured = pp._measure_json(src, -18.0, -1.5)
            assert measured is not None
            res = subprocess.run(  # noqa: S603
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-y",
                    "-i",
                    str(src),
                    "-af",
                    f"loudnorm=I=-18.0:TP=-1.5:measured_I={measured['input_i']}"
                    f":measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}"
                    f":measured_thresh={measured['input_thresh']}:linear=true:print_format=json",
                    "-ar",
                    str(_SR),
                    "-c:a",
                    "pcm_s16le",
                    str(dst),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            parsed = pp._parse_loudnorm_stderr(res.stderr, "probe")
            assert parsed is not None
            assert "normalization_type" in parsed, sorted(parsed)


class TestPassTwoDithersTheQuantization:
    """Pass-2 takes `loudnorm`'s 192 kHz FLOAT bus down to `pcm_s16le`. swr's
    `dither_method` defaults to `none`, so that step was plain truncation — the
    quantization error correlates with the signal instead of being randomized,
    which is the audible staircase on quiet tails. See
    ``_PREMASTER_RESAMPLE_ARGS`` for the bench these tests pin.
    """

    def _pass2_argv(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Capture the argv of the SECOND ffmpeg run (pass-2) from a real premaster."""
        real_run = pp.subprocess.run
        seen: list[list[str]] = []

        def _capturing_run(cmd, *args, **kwargs):  # noqa: ANN001, ANN202
            seen.append(list(cmd))
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(pp.subprocess, "run", _capturing_run)
        pp.premaster(_sine_wav(0.3))
        assert len(seen) == 2, f"expected exactly two ffmpeg runs, saw {len(seen)}"
        return seen[1]

    @_needs_ffmpeg
    def test_pass2_argv_carries_the_dither_and_no_soxr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        argv = self._pass2_argv(monkeypatch)
        # The dither is the fix; assert the exact value, not merely presence — a
        # different method (`rectangular`, `none`) is a different acoustic result.
        assert "-dither_method" in argv, f"pass-2 quantizes float->s16 with NO dither: {argv}"
        assert argv[argv.index("-dither_method") + 1] == "triangular_hp"
        # soxr is measured-inert here AND absent from mini2's ffmpeg build (PR #64).
        assert "soxr" not in argv, argv

    @_needs_ffmpeg
    def test_the_args_are_output_options_not_filter_nodes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """They MUST sit after the `-af` chain and before the output path: an
        `aresample=...:resampler=soxr` node downstream of `loudnorm` cannot
        negotiate a channel layout with the wav muxer on ffmpeg 4.x and kills the
        render outright (measured in the sibling ACX path, renderer.py:8106)."""
        argv = self._pass2_argv(monkeypatch)
        assert argv.index("-dither_method") > argv.index("-af")
        assert argv.index("-dither_method") < argv.index("-c:a")
        # And nothing smuggled the resampler into the filter string instead.
        audio_filter = argv[argv.index("-af") + 1]
        assert "resampler" not in audio_filter
        assert "aresample" not in audio_filter

    @_needs_ffmpeg
    def test_a_quiet_tail_is_not_a_quantization_staircase(self) -> None:
        """The acoustic assertion, on the REAL two-pass path.

        `astats` "Flat factor" counts runs of consecutive identical sample values
        — exactly what an undithered truncation produces in a quiet passage. A
        -60 dBFS tail (ordinary room tone / breath) measured 13.98 truncated vs
        0.00 dithered; the floor below is deliberately slack against ffmpeg-build
        variation while still being far from the truncated value.
        """
        src = _quiet_tail_float_wav(tail_dbfs=-60.0)
        dithered, _ = premaster(src)

        # Same clip, same code path, dither removed — the counterfactual is
        # measured here rather than asserted from the bench comment.
        saved = pp._PREMASTER_RESAMPLE_ARGS
        try:
            pp._PREMASTER_RESAMPLE_ARGS = [a for a in saved if a not in ("-dither_method", "triangular_hp")]
            truncated, _ = premaster(src)
        finally:
            pp._PREMASTER_RESAMPLE_ARGS = saved

        assert dithered != truncated, "dither arg had no effect on the output bytes"
        flat_dithered = _tail_flat_factor(dithered)
        flat_truncated = _tail_flat_factor(truncated)
        assert flat_truncated is not None and flat_dithered is not None
        assert flat_truncated > 5.0, (
            f"fixture does not reproduce the staircase (truncated Flat factor {flat_truncated}) — "
            "the counterfactual is vacuous, fix the fixture before trusting the pass"
        )
        assert flat_dithered < 1.0, f"dithered tail still staircases: Flat factor {flat_dithered}"


def _quiet_tail_float_wav(tail_dbfs: float, sr: int = _SR) -> bytes:
    """A 2 s spoken-ish body followed by a 2 s very quiet tail, as FLOAT.

    Float in on purpose: this is the real route's shape (`premaster_float_wav`),
    so the ONLY quantization in the measurement is pass-2's — a PCM_16 fixture
    would arrive pre-truncated and hide the effect under test.
    """
    n = int(sr * 2.0)
    idx = np.arange(n)
    body = 0.3 * np.sin(2.0 * np.pi * 220.0 * idx / sr) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * idx / sr))
    tail = (10.0 ** (tail_dbfs / 20.0)) * np.sin(2.0 * np.pi * 300.0 * idx / sr)
    buf = io.BytesIO()
    sf.write(buf, np.concatenate([body, tail]).astype(np.float32), sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _tail_flat_factor(wav_bytes: bytes) -> float | None:
    """`astats` Flat factor over the last ~1.4 s only (the quiet tail).

    Seeks in a real file, not a pipe: `-ss`/`-sseof` against stdin silently does
    not seek, which makes the measurement read the LOUD body and report 0.0 for
    both arms — a vacuous pass. (Hit while writing this test.)
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.wav"
        path.write_bytes(wav_bytes)
        res = subprocess.run(  # noqa: S603
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-ss",
                "2.4",
                "-t",
                "1.4",
                "-i",
                str(path),
                "-af",
                "astats=metadata=1:reset=0",
                "-f",
                "null",
                "-",
            ],  # fmt: skip
            capture_output=True,
            text=True,
            timeout=60,
        )
    for line in res.stderr.splitlines():
        if "Flat factor" in line:
            return float(line.strip().split(": ")[-1])
    return None


class TestTheFailOpenQuantizationIsDitheredToo:
    """The fail-open path quantizes float -> s16 as well, and it used to TRUNCATE.

    `premaster_float_wav` feeds FLOAT in. On the success path pass-2's swr does the
    float->s16 hop carrying `_PREMASTER_RESAMPLE_ARGS` (dither). On the FAIL-OPEN
    path `_as_pcm16_wav` did the very same hop through a bare
    `sf.write(..., subtype="PCM_16")`, which applies no dither at all — so a clip
    shipped the exact staircase the dither exists to remove, and the caller could
    not tell, because a fail-open emitted no metrics and only a once-per-process
    WARNING.

    Note this hop does NOT change the sample rate (24 kHz in, 24 kHz out). The
    dither still applies: it is the FORMAT change that quantizes, not the rate
    change. `test_the_undithered_counterfactual_really_staircases` pins that the
    fixture reproduces the defect, so the pass below is not vacuous.
    """

    @_needs_ffmpeg
    def test_the_undithered_counterfactual_really_staircases(self) -> None:
        """Guard against a vacuous pass: the OLD implementation must still be bad."""
        src = _quiet_tail_float_wav(tail_dbfs=-60.0)
        flat = _tail_flat_factor(pp._as_pcm16_wav_undithered(src))
        assert flat is not None and flat > 5.0, (
            f"the undithered downconvert no longer staircases (Flat factor {flat}) — "
            "the fixture is broken, fix it before trusting the dithered assertion"
        )

    @_needs_ffmpeg
    def test_a_fail_open_clip_is_not_a_quantization_staircase(self) -> None:
        src = _quiet_tail_float_wav(tail_dbfs=-60.0)
        dithered = pp._as_pcm16_wav(src)

        assert sf.info(io.BytesIO(dithered)).subtype == "PCM_16"
        flat_dithered = _tail_flat_factor(dithered)
        flat_truncated = _tail_flat_factor(pp._as_pcm16_wav_undithered(src))
        assert flat_dithered is not None and flat_truncated is not None
        assert flat_dithered < 1.0, (
            f"fail-open clip still staircases: Flat factor {flat_dithered} (undithered counterfactual {flat_truncated})"
        )

    @_needs_ffmpeg
    def test_the_downconvert_argv_carries_the_same_dither_as_pass_two(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Assert the exact method, not mere presence — `rectangular`/`none` are
        different acoustic results, and the point is that this hop reuses the
        treatment already in the file rather than inventing a second one."""
        seen: list[list[str]] = []
        real_run = pp.subprocess.run

        def _capturing_run(cmd, *args, **kwargs):  # noqa: ANN001, ANN202
            seen.append(list(cmd))
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(pp.subprocess, "run", _capturing_run)
        pp._as_pcm16_wav(_float_wav(0.5, seconds=0.5))

        assert len(seen) == 1, f"expected one ffmpeg run, saw {len(seen)}"
        argv = seen[0]
        assert "-dither_method" in argv, f"fail-open downconvert quantizes with NO dither: {argv}"
        assert argv[argv.index("-dither_method") + 1] == "triangular_hp"
        assert "soxr" not in argv, argv
        # Output options, before the output path — same placement rule as pass-2.
        assert argv.index("-dither_method") < argv.index("-c:a")

    def test_ffmpeg_unrunnable_still_returns_a_pcm16_wav(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bottom rung of the ladder: a truncated PCM_16 beats a float WAV the
        caller does not expect. `ffmpeg absent` is ALSO a premaster fail-open
        reason, so this rung is reachable in production, not theoretical."""

        def _boom(*_a: object, **_k: object) -> None:
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(pp.subprocess, "run", _boom)
        out = pp._as_pcm16_wav(_float_wav(0.5, seconds=0.5))
        assert sf.info(io.BytesIO(out)).subtype == "PCM_16"

    @_needs_ffmpeg
    def test_the_audio_itself_is_unchanged_to_within_a_bit(self) -> None:
        """The dither must not be a level/length change smuggled in as a fix."""
        src = _float_wav(0.5, seconds=0.5)
        out = pp._as_pcm16_wav(src)
        a, _ = sf.read(io.BytesIO(src), dtype="float32", always_2d=False)
        b, _ = sf.read(io.BytesIO(out), dtype="float32", always_2d=False)
        assert len(a) == len(b)
        assert float(np.abs(a - b).max()) < 1e-3


class TestAFailOpenNamesItselfToTheCaller:
    """A fail-open returns UN-NORMALISED audio with a 200. It was visible only as
    the ABSENCE of X-TTS-LUFS (indistinguishable from an unparseable measurement)
    plus a WARNING latched once per process per reason — so clip #2 onward of a
    degraded process said nothing anywhere. That is how an undithered fail-open
    shipped unnoticed for a whole wave."""

    def test_the_kill_switch_fail_open_carries_its_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", False)
        _out, metrics = premaster(_sine_wav(0.3))
        assert metrics.degraded_reason == "TTS_PREMASTER_ENABLED is off"

    def test_ffmpeg_absent_carries_its_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pp.shutil, "which", lambda _n: None)
        _out, metrics = premaster(_sine_wav(0.3))
        assert metrics.degraded_reason == "ffmpeg not found on PATH"

    @_needs_ffmpeg
    def test_a_real_premaster_reports_no_degradation(self) -> None:
        """The field must DISCRIMINATE — always-set is as useless as never-set."""
        _out, metrics = premaster(_sine_wav(0.3))
        assert metrics.degraded_reason is None
        assert metrics.integrated_lufs is not None

    @_needs_ffmpeg
    def test_the_float_path_propagates_the_reason_through_the_downconvert(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", False)
        out, metrics = pp.premaster_float_wav(_float_wav(0.5, seconds=0.5))
        assert sf.info(io.BytesIO(out)).subtype == "PCM_16"
        assert metrics.degraded_reason == "TTS_PREMASTER_ENABLED is off"

    def test_a_variable_detail_reason_is_reported_by_its_stable_latch_key(self) -> None:
        """`reason` can embed an exit code and a 300-char stderr tail; the LATCH
        key is the detail-free identity, and that is what the caller gets — so a
        caller can group by it instead of seeing a new value every request."""
        _out, metrics = pp._fail_open(
            b"x", "pass-2 loudnorm exit 1: some\nvarying stderr", key="pass-2 loudnorm nonzero exit"
        )
        assert metrics.degraded_reason == "pass-2 loudnorm nonzero exit"


class TestTheDegradedReasonIsHeaderSafe:
    """starlette encodes header values as latin-1 and RAISES on anything else, so a
    reason that is not header-safe would turn a DEGRADED clip into a 500 — the
    visibility fix becoming an outage."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("plain reason", "plain reason"),
            ("has\nnewline\rand\ttabs", "has newline and tabs"),
            ("café — naïve", "caf? ? na?ve"),
            ("", "unspecified"),
            ("   ", "unspecified"),
        ],
    )
    def test_it_is_flattened_to_bare_ascii(self, raw: str, expected: str) -> None:
        assert pp._header_safe(raw) == expected

    def test_it_is_length_bounded(self) -> None:
        assert len(pp._header_safe("x" * 5000)) == 120

    def test_every_result_actually_encodes_as_a_header(self) -> None:
        for raw in ("pass-2 loudnorm exit 1: érreur\nline two", "x" * 500, "ok"):
            pp._header_safe(raw).encode("latin-1")
