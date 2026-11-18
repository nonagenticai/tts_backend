"""The premaster must run on the ffmpeg that is ACTUALLY deployed — and fail loudly if not.

Regression pinned here (PR #64, live 2026-09-15 21:51 -> fixed by this file's PR):
``-resampler soxr -precision 28`` was added to ``_PREMASTER_RESAMPLE_ARGS``. mini2's
Homebrew ffmpeg 8.1.1 is built WITHOUT libsoxr, so pass-2 exited 234 ("Conversion
failed!") on EVERY clip, the premaster failed open, and the fail-open dithered
downconvert (same constant) failed too. Nothing noticed for two days because:

1. CI's ffmpeg HAS libsoxr, so every real-ffmpeg test passed. A test that "just runs
   ffmpeg" cannot catch a missing optional feature on a runner that has it. Hence
   ``TestNoOptionalFfmpegFeature`` asserts the argv names no optional external
   library at all — it goes red on a soxr-enabled runner too.
2. The WARNING was latched once per process, so clips 2..N were silent. Hence
   ``TestRecurringFailureStaysVisible``.
3. Nothing exercised the real invocation at process start. Hence
   ``TestStartupSelfTest`` + the ``/health`` surface.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import services.post_processor as pp

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")

# ffmpeg's optional EXTERNAL libraries (the `--enable-lib<name>` configure switches),
# i.e. features a given build may or may not carry. Static so the guard holds even
# where `ffmpeg -buildconf` is unavailable; unioned below with whatever the local
# build actually reports, so a newly-enabled lib is covered without editing this.
_KNOWN_OPTIONAL_LIBS = frozenset(
    {
        "soxr",
        "rubberband",
        "fdk_aac",
        "mp3lame",
        "opus",
        "vorbis",
        "speex",
        "zimg",
        "vidstab",
        "bs2b",
        "mysofa",
        "lv2",
        "ladspa",
        "flite",
        "openal",
        "twolame",
        "shine",
        "gsm",
        "ilbc",
        "opencore",
        "vo_amrwbenc",
        "codec2",
    }
)


@pytest.fixture(autouse=True)
def _clean_state() -> Any:
    pp._reset_log_latches()
    yield
    pp._reset_log_latches()


def _build_optional_lib_names() -> frozenset[str]:
    names = set(_KNOWN_OPTIONAL_LIBS)
    if _HAS_FFMPEG:
        res = subprocess.run(  # noqa: S603
            ["ffmpeg", "-hide_banner", "-buildconf"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
        names |= set(re.findall(r"--enable-lib([a-z0-9_]+)", res.stdout + res.stderr))
    return frozenset(names)


# Computed ONCE at import — before any test monkeypatches subprocess.run.
_OPTIONAL_LIBS = _build_optional_lib_names()


def _optional_features_named(argv: list[str]) -> list[str]:
    """Every argv token that names an optional ffmpeg library (bare or `lib`-prefixed)."""
    libs = _OPTIONAL_LIBS
    hits = []
    for arg in argv:
        for part in re.split(r"[=:,\s]+", arg.lower()):
            bare = part.removeprefix("lib")
            if part in libs or bare in libs:
                hits.append(arg)
    return hits


def _float_wav(seconds: float = 1.5, amplitude: float = 0.3, sr: int = 24000) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    buf = io.BytesIO()
    sf.write(buf, (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32), sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _capture_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    real_run = pp.subprocess.run
    seen: list[list[str]] = []

    def _capturing_run(cmd, *args, **kwargs):  # noqa: ANN001, ANN202
        seen.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(pp.subprocess, "run", _capturing_run)
    return seen


class TestNoOptionalFfmpegFeature:
    """Why this and not "run ffmpeg and see": the CI runner's ffmpeg has libsoxr, so a
    run-and-see test is GREEN with #64's argv. A name-level assertion is red on every
    build, including the ones that happen to carry the feature."""

    def test_the_shared_resample_args_name_no_optional_library(self) -> None:
        hits = _optional_features_named(pp._PREMASTER_RESAMPLE_ARGS)
        assert not hits, (
            f"_PREMASTER_RESAMPLE_ARGS depends on an optional ffmpeg build feature {hits}; "
            "mini2's ffmpeg 8.1.1 has no libsoxr and pass-2 exits 234 on every clip (PR #64)"
        )

    def test_the_dither_is_still_there(self) -> None:
        args = pp._PREMASTER_RESAMPLE_ARGS
        assert "-dither_method" in args
        assert args[args.index("-dither_method") + 1] == "triangular_hp"

    @_needs_ffmpeg
    def test_every_premaster_ffmpeg_argv_names_no_optional_library(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both real invocations: pass-2 (success path) and the fail-open downconvert."""
        seen = _capture_argv(monkeypatch)
        src = _float_wav()
        _out, metrics = pp.premaster(src)
        assert metrics.degraded_reason is None, metrics
        assert pp._dithered_pcm16(src)[0] is not None
        assert len(seen) == 3, [a[:6] for a in seen]  # pass-1, pass-2, downconvert
        for argv in seen:
            hits = _optional_features_named(argv)
            assert not hits, f"ffmpeg argv depends on optional build feature(s) {hits}: {argv}"


class TestStartupSelfTest:
    @_needs_ffmpeg
    def test_it_passes_on_the_real_invocation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
        status = pp.run_premaster_self_test()
        assert status["self_test"] == "passed", status
        assert status["integrated_lufs"] is not None
        assert abs(status["integrated_lufs"] - (-18.0)) < 1.0, status
        assert pp.premaster_health()["self_test"] == "passed"

    @_needs_ffmpeg
    def test_an_ffmpeg_that_rejects_the_argv_fails_loudly_at_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The #64 shape against REAL ffmpeg: an argument this build cannot honour.
        (A resampler engine no build has — the local build may well have soxr.)"""
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
        monkeypatch.setattr(
            pp, "_PREMASTER_RESAMPLE_ARGS", ["-resampler", "definitely_not_built", "-dither_method", "triangular_hp"]
        )
        with caplog.at_level(logging.ERROR, logger=pp.logger.name):
            status = pp.run_premaster_self_test()
        assert status["self_test"] == "failed", status
        assert "pass-2" in (status["detail"] or ""), status
        assert "downconvert" in (status["detail"] or ""), status
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors, "a failed self-test must log at ERROR, not WARNING"
        assert "SELF-TEST FAILED" in errors[0].getMessage()

    def test_kill_switch_off_is_disabled_not_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", False)
        assert pp.run_premaster_self_test()["self_test"] == "disabled"

    def test_missing_ffmpeg_is_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
        monkeypatch.setattr(pp.shutil, "which", lambda _n: None)

        def _boom(*_a: object, **_k: object) -> None:
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(pp.subprocess, "run", _boom)
        status = pp.run_premaster_self_test()
        assert status["self_test"] == "failed"
        assert "ffmpeg not found on PATH" in (status["detail"] or "")

    def test_the_self_test_does_not_pollute_the_request_counters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
        monkeypatch.setattr(pp.shutil, "which", lambda _n: None)
        pp.run_premaster_self_test()
        assert pp.premaster_health()["warning_counts"] == {}


_TEMP_PATH = "/var/folders/zz/T/tmpq8x_selftest/failopen_in.wav"
_RAW_STDERR = f"{_TEMP_PATH}: Requested resampling engine is unavailable\nConversion failed!"
_ABS_PATH = re.compile(r"(?<![\w.-])/(?:[\w.-]+/)+[\w.-]*")


class _LoadedSvc:
    def loaded_models(self) -> list[str]:
        return ["kokoro"]


@pytest.fixture
def healthy_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Models reported loaded via the seam `main._models_loaded` reads, so the ONLY
    thing that can make /health degraded is the premaster self-test."""
    import main

    monkeypatch.setattr(main.app.state, "tts_service", _LoadedSvc(), raising=False)


def _fail_ffmpeg_with_a_temp_path_in_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
    monkeypatch.setattr(pp.shutil, "which", lambda _n: "/opt/homebrew/bin/ffmpeg")

    def _rejecting_run(cmd, *_a, **_k):  # noqa: ANN001, ANN202
        return subprocess.CompletedProcess(cmd, 234, stdout="", stderr=_RAW_STDERR)

    monkeypatch.setattr(pp.subprocess, "run", _rejecting_run)


class TestHealthSurfacesTheSelfTest:
    """The client must otherwise be HEALTHY, or `degraded` is explained by "no models
    loaded" and the self-test term can be deleted with every assertion still green
    (auditor mutant `ok = bool(loaded)` survived the previous version of this class)."""

    @pytest.mark.usefixtures("healthy_models")
    def test_a_passed_self_test_with_models_loaded_is_healthy(
        self, monkeypatch: pytest.MonkeyPatch, test_client: Any
    ) -> None:
        monkeypatch.setitem(pp._self_test_status, "self_test", "passed")
        monkeypatch.setitem(pp._self_test_status, "integrated_lufs", -18.0)
        for path in ("/health", "/health/ready"):
            r = test_client.get(path)
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "healthy", (path, body)
            assert body["premaster"]["self_test"] == "passed", (path, body)

    @pytest.mark.usefixtures("healthy_models")
    def test_a_failed_self_test_marks_health_degraded_but_still_200(
        self, monkeypatch: pytest.MonkeyPatch, test_client: Any
    ) -> None:
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", True)
        monkeypatch.setattr(pp.shutil, "which", lambda _n: None)
        pp.run_premaster_self_test()
        for path in ("/health", "/health/ready"):
            r = test_client.get(path)
            assert r.status_code == 200
            body = r.json()
            assert body["models_loaded"] == ["kokoro"], body
            assert body["status"] == "degraded", (path, body)
            assert body["premaster"]["self_test"] == "failed", (path, body)

    @pytest.mark.usefixtures("healthy_models")
    def test_health_never_carries_raw_ffmpeg_stderr_or_paths(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, test_client: Any
    ) -> None:
        """/health is unauthenticated: the stderr (temp paths) stays in the ERROR log."""
        _fail_ffmpeg_with_a_temp_path_in_stderr(monkeypatch)
        with caplog.at_level(logging.ERROR, logger=pp.logger.name):
            status = pp.run_premaster_self_test()
        # Precondition: the fixture really does put the path into the detail + the log.
        assert status["self_test"] == "failed", status
        assert _TEMP_PATH in (status["detail"] or ""), status
        assert any(_TEMP_PATH in r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)

        for path in ("/health", "/health/ready"):
            r = test_client.get(path)
            assert r.status_code == 200
            raw = r.text
            body = r.json()
            assert body["status"] == "degraded", body
            assert _TEMP_PATH not in raw, raw
            assert "Conversion failed" not in raw and "resampling engine" not in raw, raw
            assert not _ABS_PATH.findall(raw), (_ABS_PATH.findall(raw), raw)
            assert "detail" not in body["premaster"], body
            reasons = body["premaster"]["reasons"]
            assert reasons, body
            assert len(reasons) <= pp._HEALTH_REASONS_MAX
            assert all(len(x) <= 120 for x in reasons), reasons
            assert "float fail-open dithered downconvert failed" in reasons, reasons

    def test_request_failures_are_counted_in_health(self, monkeypatch: pytest.MonkeyPatch, test_client: Any) -> None:
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", False)
        for _ in range(7):
            pp.premaster(b"x")
        body = test_client.get("/health").json()
        assert body["premaster"]["warning_counts"]["TTS_PREMASTER_ENABLED is off"] == 7, body


class TestRecurringFailureStaysVisible:
    """#64 was logged ONCE at 21:51 and never again, while every clip for two days
    kept failing. Volume stays bounded; recurrence does not go silent."""

    def test_a_recurring_reason_is_re_logged_after_the_interval_with_its_count(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        now = [1000.0]
        monkeypatch.setattr(pp, "_monotonic", lambda: now[0])
        monkeypatch.setattr(pp, "_PREMASTER_ENABLED", False)
        with caplog.at_level(logging.WARNING, logger=pp.logger.name):
            for _ in range(500):
                pp.premaster(b"x")
            assert len(caplog.records) == 1  # bounded within the interval
            now[0] += pp._RELOG_INTERVAL_S + 1
            for _ in range(500):
                pp.premaster(b"x")
        assert len(caplog.records) == 2, [r.getMessage() for r in caplog.records]
        second = caplog.records[1].getMessage()
        assert "STILL RECURRING" in second and "501 occurrences" in second, second
        assert pp.premaster_health()["warning_counts"]["TTS_PREMASTER_ENABLED is off"] == 1000

    def test_only_the_first_occurrence_carries_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        now = [0.0]
        monkeypatch.setattr(pp, "_monotonic", lambda: now[0])
        with caplog.at_level(logging.WARNING, logger=pp.logger.name):
            for _ in range(5):
                try:
                    raise RuntimeError("x")
                except RuntimeError:
                    pp._warn_bounded("r", "boom", exc_info=True)
                now[0] += pp._RELOG_INTERVAL_S + 1
        assert len(caplog.records) == 5
        assert sum(1 for r in caplog.records if r.exc_info) == 1


class TestTheSelfTestIsWiredIntoStartup:
    """Defined-but-never-called is the failure this guards: the lifespan must run it."""

    async def test_lifespan_runs_the_self_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import main

        class _Svc:
            async def initialize(self) -> None:
                return None

            def loaded_models(self) -> list[str]:
                return []

        async def _no_redis(_url: str) -> None:
            return None

        async def _close(_r: object) -> None:
            return None

        calls: list[int] = []
        monkeypatch.setattr(main, "get_tts_service", lambda: _Svc())
        monkeypatch.setattr(main, "init_metering_redis", _no_redis)
        monkeypatch.setattr(main, "close_metering_redis", _close)
        monkeypatch.setattr(main, "run_premaster_self_test", lambda: calls.append(1) or {})
        async with main.lifespan(main.app):
            assert calls == [1]
