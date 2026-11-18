"""Per-clip loudness pre-master for the live kokoro synth path.

A single, self-contained -18 LUFS two-pass ffmpeg ``loudnorm`` (true-peak
-1.5 dBTP) applied to each synthesized WAV before it leaves ``/api/tts/synthesize``.
This normalizes clip-to-clip loudness so a narration assembled from many separate
synth calls doesn't lurch between quiet and blaring lines.

⚠️ This runs on the LIVE narrate path (kokoro-onnx on-device, no GitHub CI,
manual deploy), so it is built to **fail open**: any misconfiguration, missing
ffmpeg, or ffmpeg error returns the INPUT bytes unchanged and never raises — a
bug here must degrade to "un-normalized audio", never to a broken narration.
The measurement logic is ported from ``w2v/tts_service`` (``AudioPostProcessor.
measure_audio``) but stripped of its schema/config deps.

Gated by ``TTS_PREMASTER_ENABLED`` (default on). A bad rollout is a one-env-flip
rollback with no code change; the flag is read once at import (a restart is the
deploy boundary).
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import shutil
import subprocess  # nosec B404
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default ON unless explicitly disabled. Read once at import; flipping requires a
# service restart, which is the deploy boundary on this launchd path.
_PREMASTER_ENABLED = os.getenv("TTS_PREMASTER_ENABLED", "true").strip().lower() not in ("false", "0", "no")

# Per-reason BOUNDED warning log. Every degraded-measurement path here can fire on
# EVERY request: at this service's ~53k synth calls/day a systemic ffmpeg fault
# logging per-call would emit ~53k lines — with tracebacks — which is an
# operational regression, not observability. So volume is bounded per REASON.
#
# ⚠️ BUT A PURE ONCE-PER-PROCESS LATCH HID A TWO-DAY OUTAGE. PR #64 made pass-2 exit
# 234 on every clip on mini2 (ffmpeg without libsoxr). The latch logged it ONCE at
# 21:51 and then said nothing while every subsequent clip skipped the premaster.
# A reason that keeps recurring must keep being visible, so:
#   - every occurrence is COUNTED (`_reason_counts`, surfaced in `/health` as
#     `premaster.warning_counts`), unconditionally;
#   - the first occurrence logs (with traceback if any);
#   - a reason still firing is re-logged at most once per `_RELOG_INTERVAL_S`,
#     WITHOUT a traceback, carrying its cumulative and since-last-log counts.
# Worst case is ~144 lines/day per reason, not 53k — and never silence.
_RELOG_INTERVAL_S = 600.0
_monotonic = time.monotonic  # indirection so tests can drive the clock without patching the stdlib
_log_lock = threading.Lock()
_reason_counts: dict[str, int] = {}
_reason_last_logged: dict[str, tuple[float, int]] = {}  # reason -> (monotonic ts, count at that log)


def _warn_bounded(reason: str, message: str, *args: object, exc_info: bool = False) -> bool:
    """Count ``reason``; WARN on its first occurrence and re-WARN while it recurs.

    Returns ``True`` if this call actually emitted a record. ``exc_info`` is only
    rendered on the FIRST occurrence — a traceback per re-log is what this bounds.

    ``reason`` is the counter/latch identity and must be a STABLE string: never
    embed variable detail (exit codes, stderr tails) in it, or every distinct ffmpeg
    error message counts as a new reason and defeats the bound. Put the detail in
    ``message``/``args`` instead.
    """
    now = _monotonic()
    with _log_lock:
        count = _reason_counts.get(reason, 0) + 1
        _reason_counts[reason] = count
        last = _reason_last_logged.get(reason)
        if last is not None and now - last[0] < _RELOG_INTERVAL_S:
            return False
        _reason_last_logged[reason] = (now, count)
    if last is None:
        logger.warning(message + " [first occurrence; re-logged while recurring]", *args, exc_info=exc_info)
    else:
        logger.warning(
            message + " [STILL RECURRING: %d occurrences in this process, %d since last logged %.0fs ago]",
            *args,
            count,
            count - last[1],
            now - last[0],
        )
    return True


def warning_counts() -> dict[str, int]:
    """Snapshot of per-reason occurrence counts since process start (for ``/health``)."""
    with _log_lock:
        return dict(_reason_counts)


def _reset_log_latches() -> None:
    """Clear counters, re-log timers and the self-test status. Test hook only."""
    with _log_lock:
        _reason_counts.clear()
        _reason_last_logged.clear()
    _self_test_status.clear()
    _self_test_status.update(_SELF_TEST_NOT_RUN)


@dataclass
class AudioMetrics:
    """Measured loudness metrics for one clip.

    All fields are nullable: measurement can fail (or be skipped on the fail-open
    path) without raising, in which case the fields stay ``None`` and the route
    omits the corresponding response header.

    ``degraded_reason`` is the ONE field that is not a measurement. It is the
    fail-open's identity, carried back to the caller. Before it existed, a fail-open
    was visible only as the ABSENCE of ``X-TTS-LUFS`` — indistinguishable from a
    measurement that merely could not be parsed — and in the log only as a WARNING
    latched once per process per reason, so the 2nd..53,000th degraded clip of a
    process said nothing anywhere (the log is now counted and re-logged while
    recurring — see ``_warn_bounded``). That silence is how an undithered fail-open
    quantization shipped unnoticed for a whole wave. The route turns this into
    ``X-TTS-Premaster-Degraded``; ``None`` means the premaster genuinely ran.
    """

    integrated_lufs: float | None = None
    loudness_range_lu: float | None = None
    true_peak_dbtp: float | None = None
    duration_seconds: float | None = None
    degraded_reason: str | None = None


def _to_float(value: Any) -> float | None:
    """Best-effort float coercion of a loudnorm JSON value; ``None`` on failure.

    Non-finite is NOT a measurement. loudnorm reports digital silence, near-silence
    and sub-gate-length clips as the literal string ``"-inf"``, and ``float("-inf")``
    coerces happily — so without this guard ``f"{value:.2f}"`` in the route would
    emit the header ``X-TTS-LUFS: -inf``, which the orchestrator parses as a float
    and treats as a real loudness reading. "Unmeasurable" must reach the route as
    ``None`` so the header is omitted entirely.
    """
    if value is None:
        return None
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


# ---------------------------------------------------------------------------
# Pass-2 quantize args (the float -> int16 step).
# ---------------------------------------------------------------------------
# `loudnorm` emits a **192 kHz float** bus regardless of its input rate (an
# ebur128 internals artifact; the compiler documents the same gotcha at
# `video_compiler_service/src/services/renderer.py` `_master_filter` /
# `_A48_SOXR`). So pass-2's `-ar <sr>` is really a 192000 -> <sr> conversion
# plus a float -> `pcm_s16le` quantization, both performed by the ONE
# auto-inserted swresample instance on the output. These are therefore OUTPUT
# options, deliberately: an `aresample=...` node inside `-af` downstream of
# `loudnorm` cannot negotiate a channel layout with the wav muxer on ffmpeg 4.x
# (the deploy host is now mini2's Homebrew ffmpeg 8.1.1, verified 2026-09-17) and kills
# the render outright.
#
# ⭐ DITHER IS THE WHOLE FIX. swr's `dither_method` defaults to `none`, so that
# float -> int16 step was plain TRUNCATION on a deliverable that is nothing but
# voice: the quantization error correlates with the signal instead of being
# randomized, which is audible as a staircase on quiet tails and breaths.
# Bench (wave 13, an ffmpeg 4.4 build — NOT the deploy host; mini2 runs 8.1.1 with
# no libsoxr). Isolated stage:
# a -90 dBFS 1 kHz tone on a 192 kHz float bus taken to 24 kHz / `pcm_s16le`
# with these exact args measures `astats` **Flat factor 19.08 undithered vs
# 0.005 with `triangular_hp`** (long runs of identical quantized samples vs
# none). The sibling ACX path measured 23.5 -> 0.016 for the same change at
# 44.1 kHz (`renderer.py:8120-8132`) — same effect, different rate.
#
# ⭐ AND THROUGH THIS FUNCTION, ON REALISTIC LEVELS — the part that answers "but
# is a -90 dBFS tone a real narration?". A 2 s spoken body followed by a 2 s
# quiet tail, premastered to -18 LUFS, astats over the tail only:
#
#     tail level   truncated   dithered
#     -60 dBFS       13.98       0.00
#     -70 dBFS       16.90       0.00
#     -80 dBFS       20.83       0.00
#     -90 dBFS       29.25       0.14
#
# -60 dBFS is ordinary room tone / breath, so the staircase was live on the
# whole fleet, not only on pathologically quiet material.
#
# ⛔ NO `-resampler soxr` HERE — IT IS INERT AND IT BROKE PRODUCTION.
# Measured: on this hop `-resampler soxr -precision 28` produced a **byte-identical**
# output file (identical in-band RMS to 0.00005 dB; both engines rejected 14/20/30/47
# kHz out-of-band tones to digital silence) and, alone, left the -60 dBFS-tail Flat
# factor at 9.54 — i.e. it improves NOTHING. 192000 -> 24000 is an EXACT 8:1 integer
# decimation, and swr's default `filter_type` is kaiser windowed sinc (NOT `cubic`).
# PR #64 added it anyway; mini2's Homebrew ffmpeg 8.1.1 is built WITHOUT libsoxr, so
# pass-2 exited 234 ("Conversion failed!") on EVERY clip from 2026-09-15 21:51, the
# premaster failed open, and the fail-open dithered downconvert (this same constant)
# failed too — so neither the loudness premaster nor the dither ever ran in prod.
# The dither needs no optional library: it is built into swresample.
# Pinned by tests/test_premaster_ffmpeg_portability.py (argv may name no optional
# ffmpeg library) and by `run_premaster_self_test` at process start.
#
# ⚠️ RESAMPLE-HOP INVENTORY — a narration sample crosses FOUR rate conversions
# between here and delivery. Do not add a fifth, and do not "helpfully" re-stamp
# a rate that a later hop overwrites:
#   (a) HERE:  192000 -> <sr> (24000 for kokoro), float -> s16.
#   (b) compiler `_A48_SOXR` (renderer.py:3130, applied at :4983 and elsewhere):
#       24000 -> 48000, soxr/precision=28.
#   (c) compiler master chain (renderer.py:3666): loudnorm's 192 kHz bus ->
#       48000 via `aformat=sample_rates=48000`.
#   (d) delivery AAC (renderer.py:141 `_AUDIO_CODEC_ARGS`): `-ar 48000`.
# Hop (a) band-limits the clip to 12 kHz purely to re-stamp the engine rate,
# and hop (b) immediately upsamples it back — two conversions that buy nothing.
# Removing (a) (emitting 48 kHz, or float, from this service) is a REAL
# simplification but it changes this service's public response format, which
# `/api/tts/synthesize` callers and `_as_pcm16_wav` both pin to PCM_16 at the
# input rate. That is an API decision, not a cleanup: it is NOT done here.
_PREMASTER_RESAMPLE_ARGS = [
    "-dither_method",
    "triangular_hp",
]


def _sample_rate(path: Path, default: int = 24000) -> int:
    """Input sample rate via soundfile; falls back to the kokoro default (24 kHz)."""
    try:
        import soundfile as sf

        return int(sf.info(str(path)).samplerate)
    except Exception:  # noqa: BLE001 — sample-rate probing must never break premaster
        return default


def _duration(path: Path) -> float | None:
    """Clip duration in seconds via soundfile; ``None`` if it can't be read."""
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001 — duration is advisory only
        return None


def _parse_loudnorm_stderr(stderr: str, context: str) -> dict[str, Any] | None:
    """Parse the trailing ``{...}`` ``print_format=json`` block out of ffmpeg stderr.

    ffmpeg writes the loudnorm JSON block to *stderr*, so we take the last brace
    pair (rfind). Shared by the pass-1 measurement (``_measure_json``) and the
    pass-2 post-normalisation read in ``premaster``. Returns ``None`` — never
    raises — on any unparseable output, logging a distinct once-per-reason WARNING
    first (see ``_warn_bounded``): these used to be silent, which is exactly how a
    missing ``X-TTS-LUFS`` header reached the orchestrator with nothing in the log
    to explain it. The latch key is scoped by ``context`` so pass-1 and pass-2 each
    get their own line.
    """
    start = stderr.rfind("{")
    end = stderr.rfind("}") + 1
    if start < 0 or end <= start:
        _warn_bounded(
            f"{context}/no-json-block", "loudnorm measurement unusable (%s): no JSON block in ffmpeg stderr", context
        )
        return None
    try:
        parsed = json.loads(stderr[start:end])
    except json.JSONDecodeError, ValueError, TypeError:
        _warn_bounded(
            f"{context}/malformed-json", "loudnorm measurement unusable (%s): malformed JSON in ffmpeg stderr", context
        )
        return None
    if not isinstance(parsed, dict):
        _warn_bounded(
            f"{context}/json-not-an-object",
            "loudnorm measurement unusable (%s): JSON block was %s, not an object",
            context,
            type(parsed).__name__,
        )
        return None
    return parsed


# The four pass-1 figures interpolated into pass-2's filter. ALL must be finite:
# `measured_I=-inf` makes pass-2 exit 1 (see the guard in `premaster`).
_PASS1_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh")


def _measure_json(path: Path, target_lufs: float, true_peak: float) -> dict[str, Any] | None:
    """Run ``loudnorm`` in measurement mode and return its parsed JSON dict.

    ffmpeg writes the ``print_format=json`` block (``input_i``/``input_tp``/
    ``input_lra``/``input_thresh``) to *stderr*; we parse the trailing ``{...}``
    (rfind of the last brace pair). Output is discarded to ``-f null`` — this is
    the standard two-pass measurement pass. (We deliberately do NOT pass
    ``measured_only``: that option only exists on newer ffmpeg builds and errors
    the filter init on the 4.x on the deploy host; omitting it still prints the
    input measurements, just having also processed the discarded stream.) Returns
    ``None`` on any non-zero exit or unparseable output — every such path logs a
    distinct WARNING, so a degraded measurement is visible in the narrate log.
    """
    context = "measurement pass"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(path),
        "-af",
        f"loudnorm=I={target_lufs}:TP={true_peak}:print_format=json",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # nosec B603 B607  # noqa: S603, S607 -- fixed ffmpeg argv, no user-controlled tokens
    except OSError, subprocess.SubprocessError:
        # ffmpeg binary missing/not executable (OSError) or it hung past the
        # timeout (subprocess.TimeoutExpired) — honor measure_audio()'s "never
        # raises" contract and fail open to None so callers degrade gracefully.
        # No `as` binding: PEP 758 only allows the unparenthesized tuple without
        # one, and exc_info=True carries the type/message anyway. The traceback is
        # latched to the FIRST occurrence — a missing ffmpeg is a per-request
        # condition, and one traceback per synth call is a log flood, not a signal.
        _warn_bounded(
            f"{context}/ffmpeg-unrunnable",
            "loudnorm measurement unusable (%s): ffmpeg could not be run",
            context,
            exc_info=True,
        )
        return None
    if result.returncode != 0:
        # Latch key excludes the exit code / stderr tail on purpose: those vary per
        # clip, and keying on them would make the latch unbounded again.
        _warn_bounded(
            f"{context}/ffmpeg-nonzero-exit",
            "loudnorm measurement unusable (%s): ffmpeg exited %d: %s",
            context,
            result.returncode,
            result.stderr[-300:],
        )
        return None
    return _parse_loudnorm_stderr(result.stderr, context)


def measure_audio(wav_bytes: bytes) -> AudioMetrics:
    """Measure integrated LUFS / LRA / true-peak of a WAV, best-effort.

    Never raises: on any ffmpeg/parse failure the loudness fields stay ``None``
    (duration is still reported when soundfile can read the clip).
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "measure.wav"
        path.write_bytes(wav_bytes)
        duration = _duration(path)
        data = _measure_json(path, -18.0, -1.5)
        if data is None:
            return AudioMetrics(duration_seconds=duration)
        return AudioMetrics(
            integrated_lufs=_to_float(data.get("input_i")),
            loudness_range_lu=_to_float(data.get("input_lra")),
            true_peak_dbtp=_to_float(data.get("input_tp")),
            duration_seconds=duration,
        )


def _fail_open(
    wav_bytes: bytes, reason: str, *, key: str | None = None, exc_info: bool = False
) -> tuple[bytes, AudioMetrics]:
    """Pass the input bytes through unchanged; count + log the reason (bounded).

    ``key`` overrides the latch identity — pass a stable one whenever ``reason``
    embeds variable detail (exit codes, stderr tails). Previously this used a
    single process-wide boolean, so the SECOND distinct fail-open cause of a
    process was silent; keying by reason keeps every distinct cause visible while
    still bounding volume.
    """
    _warn_bounded(key or reason, "premaster passing audio through unchanged (%s)", reason, exc_info=exc_info)
    # `key or reason` is the LATCH identity, which is already required to be a
    # stable, detail-free string — exactly what a header value should be. `reason`
    # alone can carry an exit code and a 300-char stderr tail (newlines included),
    # which is not header-safe; `_header_safe` is the belt to that braces.
    return wav_bytes, AudioMetrics(degraded_reason=_header_safe(key or reason))


def _header_safe(reason: str) -> str:
    """Collapse a fail-open reason to something that can sit in an HTTP header.

    Latin-1 only (starlette encodes header values as latin-1 and RAISES on anything
    else — a UnicodeEncodeError inside the response would turn a DEGRADED clip into
    a 500, i.e. the visibility fix would become an outage), no CR/LF (response
    splitting), bounded length.
    """
    flat = " ".join(reason.split())
    ascii_only = flat.encode("ascii", "replace").decode("ascii")
    return ascii_only[:120] or "unspecified"


def _check_normalization_type(post: dict[str, Any] | None) -> str | None:
    """WARN once if pass-2 did not actually apply the LINEAR (pure-gain) normalisation.

    We *request* ``linear=true``, but that is a preference, not a guarantee: ffmpeg
    silently reverts to ``dynamic`` (time-varying, per-frame gain) whenever the
    single linear gain that would hit the target would breach ``TP``. Nothing read
    the answer back, while ``tts_routes._word_timings_in_output_timebase`` documents
    "``linear=true`` loudnorm is pure gain" as the reason word timings survive the
    premaster untouched. A dynamic pass is still frame-count-preserving, so the
    timings are fine either way — but the clip is no longer a pure gain change, and
    that is exactly the kind of thing that should not be invisible.

    Reporting only. ``_pass2_filter`` pre-limits peaks so linear is reachable; a
    ``dynamic`` here now means that guard could not apply. Returns the observed
    type (``None`` when pass-2 reported nothing) so a caller/test can read it.
    """
    if post is None:
        return None
    observed = post.get("normalization_type")
    if not isinstance(observed, str):
        return None
    if observed.strip().lower() != "linear":
        _warn_bounded(
            "pass-2 normalization_type not linear",
            "premaster requested linear=true but loudnorm applied normalization_type=%r "
            "(ffmpeg falls back to dynamic gain when the linear gain would breach the true peak); "
            "the clip is no longer a pure gain change",
            observed,
        )
    return observed


# ---------------------------------------------------------------------------
# Pass-2 true-peak headroom limiter — what makes `linear=true` actually linear
# ---------------------------------------------------------------------------
# ffmpeg's loudnorm honours `linear=true` only if `measured_TP + (I - measured_I)
# <= TP` (and measured_LRA <= LRA); otherwise it silently runs DYNAMIC gain. Kokoro
# clips are ~-25.5 LUFS with -3..-7 dBTP plosive peaks and ~1.5 LU LRA, so the
# +7.3..+7.8 dB linear gain would land peaks at +0.4..+4.9 dBTP: TRUE PEAK is the
# breaker on every clip, never LRA (measured on mini2's ffmpeg 8.1.1, 3 real clips,
# 2026-09-17: all three `dynamic` before, delivered TP up to -1.39 dBTP i.e. OVER the
# ceiling after the s16 downconvert; all three `linear` after, delivered I within
# 0.25 LU of -18, TP -1.78 dBTP, frame count and alignment unchanged).
#
# Fix: when (and only when) the linear gain would breach TP, lookahead-limit the
# peaks FIRST, at 192 kHz (loudnorm's own true-peak rate, so sample peak ~= true
# peak), to exactly the ceiling the linear gain can afford, and tell loudnorm that
# ceiling as `measured_TP`. The integrated loudness moves by <0.3 LU (a few ms of
# plosives), so pass-1's `measured_I` still stands.
#
# ⚠️ PORTABILITY: alimiter's `latency=true` does not exist on ffmpeg 4.4 (dev/CI);
# without compensation the limiter delays the clip by `attack` (120 samples at
# 24 kHz, measured), which would shift every word timing. So the delay is removed
# with builtin filters that exist on every build: `apad` the tail by the lookahead,
# `atrim` the same count off the head, re-stamp pts. Measured lag 0 samples on both
# 4.4 and 8.1.1. No optional external library (see _PREMASTER_RESAMPLE_ARGS).
_LIMITER_RATE = 192000
_LIMITER_ATTACK_MS = 5.0
_LIMITER_RELEASE_MS = 50.0
# Below TP so the 192k -> <sr>/s16 dithered downconvert cannot push it back over.
_LIMITER_MARGIN_DB = 0.3
# alimiter's `limit` domain is [0.0625, 1] (linear) — i.e. [-24.08, 0] dBFS.
_ALIMITER_MIN_LIMIT_DB = 20.0 * math.log10(0.0625)


def _linear_peak_ceiling_db(measured: dict[str, Any], target_lufs: float, true_peak: float) -> float | None:
    """The dBTP the input's peaks must be limited to for a linear pass-2, or ``None``.

    ``None`` means no limiter: either the linear gain already fits under ``true_peak``
    (the common quiet-peak case — pass-2 stays exactly the plain linear loudnorm), or
    the required ceiling is outside alimiter's range (then loudnorm falls back to
    dynamic and ``_check_normalization_type`` still says so).
    """
    input_i = _to_float(measured.get("input_i"))
    input_tp = _to_float(measured.get("input_tp"))
    if input_i is None or input_tp is None:
        return None
    offset = target_lufs - input_i
    if input_tp + offset <= true_peak:
        return None
    ceiling = true_peak - offset - _LIMITER_MARGIN_DB
    if not (_ALIMITER_MIN_LIMIT_DB <= ceiling < 0.0):
        return None
    return ceiling


def _pass2_filter(measured: dict[str, Any], target_lufs: float, true_peak: float) -> str:
    """Build pass-2's ``-af`` graph: [headroom limiter,] seeded linear loudnorm."""
    ceiling = _linear_peak_ceiling_db(measured, target_lufs, true_peak)
    prefix = ""
    measured_tp = measured["input_tp"]
    if ceiling is not None:
        lookahead = round(_LIMITER_RATE * _LIMITER_ATTACK_MS / 1000.0)
        prefix = (
            f"aresample={_LIMITER_RATE},"
            f"apad=pad_len={lookahead},"
            f"alimiter=limit={10.0 ** (ceiling / 20.0):.6f}"
            f":attack={_LIMITER_ATTACK_MS}:release={_LIMITER_RELEASE_MS}:level=false,"
            f"atrim=start_sample={lookahead},"
            f"asetpts=N/SR/TB,"
        )
        measured_tp = f"{ceiling:.2f}"
    return (
        prefix + f"loudnorm=I={target_lufs}:TP={true_peak}"
        f":measured_I={measured['input_i']}"
        f":measured_TP={measured_tp}"
        f":measured_LRA={measured['input_lra']}"
        f":measured_thresh={measured['input_thresh']}"
        f":linear=true"
        f":print_format=json"
    )


def premaster(
    wav_bytes: bytes,
    target_lufs: float = -18.0,
    true_peak: float = -1.5,
    *,
    _diag: dict[str, Any] | None = None,
) -> tuple[bytes, AudioMetrics]:
    """Two-pass ``loudnorm`` a clip to ``target_lufs`` / ``true_peak``, fail-open.

    Short-clip measurement (synthetic speech-like clips, 0.3-3.0s, both ffmpeg
    4.4 and 8.1.1 mini2): the pre-fix pass-2 (no headroom limiter) landed
    -33..-19 LUFS on 0.5-2.0s clips depending on ffmpeg build (worst on 4.4:
    -36..-32) while this version lands -18.0 +/- 0.5 on every duration >= 0.5s
    on both builds. Below ~0.4s pass-1's gating block can't measure either
    version and both fail open identically (unchanged behaviour). Verdict: the
    limiter graph is strictly better on short clips, never worse, on both
    ffmpeg builds tested.

    Pass 1 measures the input; pass 2 applies a linear ``loudnorm`` seeded with
    those measurements (``linear=true`` → exact, non-dynamic gain) and re-encodes
    to 16-bit PCM at the input sample rate. Pass 2 also carries
    ``print_format=json``, so the post-normalisation figures come out of its OWN
    stderr — there is deliberately NO third ffmpeg run. ``duration_seconds`` comes
    from soundfile (no subprocess). Exactly two ffmpeg processes per clip.

    Returns ``(wav_bytes_unchanged, AudioMetrics())`` — byte-identical input,
    empty metrics — and NEVER raises when: the kill-switch is off; ffmpeg is not
    on PATH; pass-1 measurement fails; pass-1 reports no FINITE loudness (silent /
    near-silent / sub-0.4 s clip — not normalisable at all, as distinct from a
    failure); pass-2 exits non-zero; the output is empty; or anything at all goes
    wrong. Empty metrics means the route omits ``X-TTS-LUFS`` — which is the
    honest answer for an unmeasurable clip, and better than a fabricated one.
    Every one of those outcomes is counted and logged, bounded per reason (``_warn_bounded``).
    """
    if not _PREMASTER_ENABLED:
        return _fail_open(wav_bytes, "TTS_PREMASTER_ENABLED is off")
    if not shutil.which("ffmpeg"):
        return _fail_open(wav_bytes, "ffmpeg not found on PATH")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "in.wav"
            out_path = Path(tmp) / "out.wav"
            in_path.write_bytes(wav_bytes)

            measured = _measure_json(in_path, target_lufs, true_peak)
            if measured is None:
                return _fail_open(wav_bytes, "pass-1 loudnorm measurement failed")

            # A clip with no measurable integrated loudness — digital silence,
            # near-silence (~-70 dBFS, i.e. everything below loudnorm's absolute
            # gate), or anything shorter than the ~0.4 s EBU R128 gating block —
            # makes pass-1 report `input_i: "-inf"` WITH exit 0. Interpolating that
            # into pass-2's filter kills pass-2 ("Failed to inject frame into
            # filter network: Result too large"), so this used to surface as a
            # generic pass-2 failure — and, under the old single latch, usually as
            # nothing at all. Such a clip cannot be loudness-normalised (there is
            # no finite loudness to move to the target), so name that outcome and
            # hand the audio back untouched instead of pretending it failed.
            unmeasurable = [k for k in _PASS1_KEYS if _to_float(measured.get(k)) is None]
            if unmeasurable:
                return _fail_open(
                    wav_bytes,
                    "clip has no measurable integrated loudness (digital silence, near-silence, or shorter than the"
                    f" ~0.4s gating block) — non-finite pass-1 {', '.join(unmeasurable)}; not loudness-normalisable",
                    key="pass-1 non-finite measurement",
                )

            sr = _sample_rate(in_path)
            audio_filter = _pass2_filter(measured, target_lufs, true_peak)
            if _diag is not None:
                # Self-test only: which pass-2 graph ran, and what loudnorm did with it.
                _diag["limited"] = audio_filter.startswith("aresample=")
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                str(in_path),
                "-af",
                audio_filter,
                "-ar",
                str(sr),
                # DITHER for the 192 kHz-float -> <sr>/s16 step. See
                # `_PREMASTER_RESAMPLE_ARGS`: without `-dither_method` this is an
                # undithered truncation (astats Flat factor 19.08 vs 0.005).
                *_PREMASTER_RESAMPLE_ARGS,
                "-c:a",
                "pcm_s16le",
                str(out_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # nosec B603 B607  # noqa: S603, S607 -- fixed ffmpeg argv, no user-controlled tokens
            if result.returncode != 0:
                return _fail_open(
                    wav_bytes,
                    f"pass-2 loudnorm exit {result.returncode}: {result.stderr[-300:]}",
                    key="pass-2 loudnorm nonzero exit",
                )
            if not out_path.exists() or out_path.stat().st_size == 0:
                return _fail_open(wav_bytes, "pass-2 loudnorm produced empty output")

            out_bytes = out_path.read_bytes()

            # Pass-2 already reports the POST-normalisation figures in its own
            # `print_format=json` block (`output_i`/`output_tp`/`output_lra`),
            # empirically identical to re-measuring out.wav from scratch. Reading
            # them here deletes a third full ffmpeg run per clip.
            #
            # This used to be `measure_audio(out_bytes)` — a separate ffmpeg
            # invocation whose five failure paths all returned None silently. When
            # it failed, premaster handed back correctly-normalised bytes with
            # integrated_lufs=None and the route dropped X-TTS-LUFS with nothing
            # logged (66 orchestrator complaints/24h against 0 fail-open latches).
            post = _parse_loudnorm_stderr(result.stderr, "pass-2 normalisation")
            observed_type = _check_normalization_type(post)
            if _diag is not None:
                _diag["normalization_type"] = observed_type
            duration = _duration(out_path)
            # `_to_float`, not a bare `is None`: a non-finite `output_i` is not a
            # measurement either, and must not reach the header as `-inf`.
            if post is None or _to_float(post.get("output_i")) is None:
                # Degrade, don't regress: fall back to the old re-measure rather
                # than returning metrics-free (which is what dropped the header).
                _warn_bounded(
                    "pass-2 output measurement missing",
                    "premaster pass-2 reported no output measurement; re-measuring the output clip",
                )
                return out_bytes, measure_audio(out_bytes)
            return out_bytes, AudioMetrics(
                integrated_lufs=_to_float(post.get("output_i")),
                loudness_range_lu=_to_float(post.get("output_lra")),
                true_peak_dbtp=_to_float(post.get("output_tp")),
                duration_seconds=duration,
            )
    except Exception:  # noqa: BLE001 — premaster must NEVER fail the live synth path
        return _fail_open(wav_bytes, "unexpected error", exc_info=True)


def _as_pcm16_wav_undithered(wav_bytes: bytes) -> bytes:
    """`soundfile` float -> PCM_16. LAST RESORT ONLY — this is a plain truncation.

    ``sf.write(..., subtype="PCM_16")`` applies NO dither, so the quantization error
    correlates with the signal: exactly the staircase on quiet tails that
    ``_PREMASTER_RESAMPLE_ARGS`` exists to remove (measured here at astats Flat
    factor 9.54 undithered vs 0.00 dithered, ffmpeg 4.4, same-rate hop). Reached only
    when ffmpeg itself cannot run — which is also one of the reasons the premaster
    failed open in the first place — so a truncated PCM_16 clip still beats handing a
    caller a float WAV it does not expect. Never call this directly; go through
    ``_as_pcm16_wav``.
    """
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _dithered_pcm16(wav_bytes: bytes) -> tuple[bytes | None, str | None]:
    """ffmpeg float -> DITHERED PCM_16 at the same rate. ``(bytes, None)`` or ``(None, detail)``.

    Never raises. Split out of ``_as_pcm16_wav`` so the startup self-test can
    exercise this exact invocation and SEE its failure — ``_as_pcm16_wav`` itself
    must degrade silently to truncation, which is precisely what hid PR #64 here.
    """
    try:
        with tempfile.TemporaryDirectory() as td:
            in_path = Path(td) / "failopen_in.wav"
            out_path = Path(td) / "failopen_out.wav"
            in_path.write_bytes(wav_bytes)
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                str(in_path),
                "-ar",
                str(_sample_rate(in_path)),
                # The SAME dither as the success path's pass-2 quantization.
                *_PREMASTER_RESAMPLE_ARGS,
                "-c:a",
                "pcm_s16le",
                str(out_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # nosec B603 B607  # noqa: S603, S607 -- fixed ffmpeg argv, no user-controlled tokens
            if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                return out_path.read_bytes(), None
            return None, f"ffmpeg exit {result.returncode}: {result.stderr[-300:]}"
    except Exception as exc:  # noqa: BLE001 — this is itself a fail-open path
        return None, f"{type(exc).__name__}: {exc}"


def _as_pcm16_wav(wav_bytes: bytes) -> bytes:
    """Re-encode a WAV to DITHERED 16-bit PCM, degrading rather than raising.

    Only used on ``premaster_float_wav``'s fail-open path, where the bytes handed
    back are the 32-bit FLOAT input rather than pass-2's output. Callers of
    ``/api/tts/synthesize`` expect the same PCM_16 WAV they have always received, so
    the response contract must not depend on whether the premaster succeeded.

    ⚠️ IT IS STILL A QUANTIZATION, SO IT STILL NEEDS THE DITHER. This used to be a
    bare ``sf.write(..., subtype="PCM_16")``, which is an undithered truncation.
    Same treatment, same constant: one swr instance carrying
    ``_PREMASTER_RESAMPLE_ARGS``. Measured on this hop (float 24 kHz -> s16 24 kHz,
    -60 dBFS tail): astats Flat factor **9.54 undithered vs 0.00 dithered**, so the
    dither engages on a same-RATE conversion too — it is the format change, not the
    rate change, that quantizes.

    Ladder, most to least faithful: dithered ffmpeg -> undithered ``soundfile``
    (ffmpeg unrunnable) -> the float bytes unchanged (both failed).
    """
    out, detail = _dithered_pcm16(wav_bytes)
    if out is not None:
        return out
    _warn_bounded(
        "float fail-open dithered downconvert failed",
        "premaster fail-open could not dither the float WAV to PCM_16 (%s); truncating instead",
        detail,
    )
    try:
        return _as_pcm16_wav_undithered(wav_bytes)
    except Exception:  # noqa: BLE001 — bottom of the ladder
        # A float WAV is still a valid, playable RIFF file, so handing it back on a
        # triple failure is a degraded format, never a broken response.
        _warn_bounded(
            "float fail-open downconvert failed",
            "premaster fail-open could not re-encode the float WAV to PCM_16; returning float WAV",
            exc_info=True,
        )
        return wav_bytes


def premaster_float_wav(
    float_wav_bytes: bytes,
    target_lufs: float = -18.0,
    true_peak: float = -1.5,
) -> tuple[bytes, AudioMetrics]:
    """``premaster`` for an UN-QUANTIZED (32-bit float) WAV input.

    Why this exists: the route used to ``encode_wav`` to int16 and only then run the
    premaster, so a quiet clip was truncated to 16 bits FIRST and loudnorm's gain was
    applied SECOND — lifting the quantization floor along with the signal, which is
    audible as grit/hiss in the pauses of a quiet line. Worse, pass-2's
    ``-c:a pcm_s16le`` quantized a second time, undithered. Feeding float in means
    there is exactly ONE quantization, and it is the LAST step.

    The output format is unchanged (PCM_16 WAV at the input rate) — pass-2 already
    encoded ``pcm_s16le``, so the public response stays byte-shaped as before. The
    one thing ``premaster`` cannot do for a float input is fail open: its fail-open
    returns the INPUT bytes, which here are float. We detect that by identity (
    ``_fail_open`` hands back the very same object) and down-convert, so every exit
    from this function is a PCM_16 WAV.
    """
    out_bytes, metrics = premaster(float_wav_bytes, target_lufs, true_peak)
    if out_bytes is float_wav_bytes:
        return _as_pcm16_wav(float_wav_bytes), metrics
    return out_bytes, metrics


# ---------------------------------------------------------------------------
# Startup self-test
# ---------------------------------------------------------------------------
# Every failure above is FAIL-OPEN by design, which means a broken invocation (an
# ffmpeg build missing a feature the argv needs — PR #64) is indistinguishable from
# "working" unless someone reads a log line. So at process start we run the REAL
# premaster (pass-1 + pass-2) and the REAL fail-open downconvert on a synthetic clip.
#
# On failure: log at ERROR and mark `/health` + `/health/ready` `status=degraded`
# (still HTTP 200). Deliberately NOT a crash: the premaster is fail-open, so the
# unit still delivers correct (un-normalised) speech; crashing would turn a quality
# regression into a narration outage and put launchd into a restart loop on a host
# whose deploy is a 5-minute git sync. Degraded-and-visible is the honest state.
_SELF_TEST_NOT_RUN: dict[str, Any] = {"self_test": "not_run", "detail": None, "reasons": [], "integrated_lufs": None}
# `/health` is UNAUTHENTICATED. `detail` carries raw ffmpeg stderr (temp-file paths
# included) and goes to the ERROR log only; `/health` gets `reasons`: stable,
# path-free, bounded codes (the same latch keys `_fail_open`/`_warn_bounded` use).
_HEALTH_REASONS_MAX = 4
_self_test_status: dict[str, Any] = dict(_SELF_TEST_NOT_RUN)
# The self-test clip lands -18.29 LUFS (ffmpeg 4.4). 1 LU catches a wrong-gain
# graph (ffmpeg 4.4's short-input branch was 14 LU off) without flaking on limiter loss.
_SELF_TEST_LUFS_TOLERANCE = 1.0


def _self_test_clip(sr: int = 24000, seconds: float = 3.5) -> bytes:
    """A FLOAT mono WAV — the route's real premaster input shape — long enough to gate.

    Deliberately PEAKY: a quiet tone alternating -30.5/-24.4 dBFS every 0.5 s (a constant
    tone measures LRA 0, which loudnorm refuses for linear) carrying 1 ms -12 dBFS
    transients every 0.5 s, like a real kokoro clip's plosives. >= 3 s on purpose: below that
    loudnorm takes a short-input branch whose gain differs across ffmpeg versions
    (4.4 lands ~-32 LUFS, 8.1.1 ~-18.2 on the same 1-2 s speech). The linear gain to -18 LUFS would breach TP,
    so pass-2 MUST take the headroom-limiter graph (aresample/apad/alimiter/atrim/
    asetpts). A plain loud sine never did, so a build missing any of those filters
    would have passed the self-test and failed open on every real clip (the soxr lesson).
    """
    import numpy as np
    import soundfile as sf

    t = np.arange(int(sr * seconds)) / sr
    level = np.where((t // 0.5) % 2 == 0, 0.03, 0.06)
    pcm = level * np.sin(2.0 * np.pi * 440.0 * t)
    burst = int(0.001 * sr)
    for start in range(int(0.1 * sr), len(pcm) - burst, int(0.5 * sr)):
        pcm[start : start + burst] += 0.25 * np.sin(2.0 * np.pi * 1000.0 * t[:burst])
    pcm = pcm.astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def run_premaster_self_test() -> dict[str, Any]:
    """Run the real premaster invocations once; record + return the outcome. Never raises.

    ``self_test`` is ``passed`` | ``failed`` | ``disabled`` (kill-switch off). The
    self-test's own failures are NOT counted in ``warning_counts`` — those count
    request traffic only.
    """
    if not _PREMASTER_ENABLED:
        status: dict[str, Any] = {
            "self_test": "disabled",
            "detail": "TTS_PREMASTER_ENABLED is off",
            "reasons": ["TTS_PREMASTER_ENABLED is off"],
            "integrated_lufs": None,
        }
        _self_test_status.update(status)
        return dict(status)

    with _log_lock:
        counts_before = dict(_reason_counts)
        logged_before = dict(_reason_last_logged)
    problems: list[str] = []
    reasons: list[str] = []  # path-free codes for /health; `problems` keeps the detail for the log
    lufs: float | None = None
    try:
        clip = _self_test_clip()
        diag: dict[str, Any] = {}
        target_lufs = -18.0
        out, metrics = premaster(clip, target_lufs, _diag=diag)
        lufs = metrics.integrated_lufs
        if metrics.degraded_reason is not None or out is clip:
            problems.append(f"premaster failed open: {metrics.degraded_reason}")
            reasons.append(metrics.degraded_reason or "premaster failed open")
        elif lufs is None:
            problems.append("premaster produced no integrated LUFS (X-TTS-LUFS would be omitted)")
            reasons.append("no integrated LUFS")
        else:
            if diag.get("limited") is not True:
                problems.append("self-test clip did not exercise the pass-2 headroom limiter graph")
                reasons.append("headroom limiter not exercised")
            if abs(lufs - target_lufs) > _SELF_TEST_LUFS_TOLERANCE:
                problems.append(f"premaster landed {lufs} LUFS, expected {target_lufs} +/- {_SELF_TEST_LUFS_TOLERANCE}")
                reasons.append("premaster off target loudness")
            if diag.get("normalization_type") != "linear":
                problems.append(f"pass-2 normalization_type={diag.get('normalization_type')!r}, expected 'linear'")
                reasons.append("pass-2 not linear")
        _dc, dc_detail = _dithered_pcm16(clip)
        if dc_detail is not None:
            problems.append(f"fail-open dithered downconvert failed: {dc_detail}")
            reasons.append("float fail-open dithered downconvert failed")
    except Exception as exc:  # noqa: BLE001 — the self-test must never take the unit down
        problems.append(f"self-test raised {type(exc).__name__}: {exc}")
        reasons.append(f"self-test raised {type(exc).__name__}")
    finally:
        with _log_lock:
            # Keep the first-occurrence log lines the self-test emitted (they carry
            # the ffmpeg stderr), but do not let a synthetic clip inflate the
            # request-traffic counters or suppress the first REAL occurrence's log.
            _reason_counts.clear()
            _reason_counts.update(counts_before)
            _reason_last_logged.clear()
            _reason_last_logged.update(logged_before)

    if problems:
        detail = " | ".join(problems)
        # Pass-2's stderr tail is only in the WARNING above; put the argv here so the
        # ERROR alone says what was run.
        logger.error(
            "PREMASTER SELF-TEST FAILED — every synthesized clip will skip loudness normalisation and dither "
            "until fixed; /health reports degraded. %s (quantize args: %s)",
            detail,
            " ".join(_PREMASTER_RESAMPLE_ARGS),
        )
        status = {
            "self_test": "failed",
            "detail": detail[:2000],
            "reasons": [_header_safe(r) for r in reasons[:_HEALTH_REASONS_MAX]],
            "integrated_lufs": lufs,
        }
    else:
        logger.info("premaster self-test passed (integrated %.2f LUFS)", lufs)
        status = {"self_test": "passed", "detail": None, "reasons": [], "integrated_lufs": lufs}
    _self_test_status.update(status)
    return dict(status)


def premaster_health() -> dict[str, Any]:
    """The premaster block of ``/health``: self-test outcome + per-reason counts.

    Deliberately an ALLOWLIST, not ``{**_self_test_status}``: ``detail`` (raw ffmpeg
    stderr, temp paths) must never reach the unauthenticated endpoint.
    """
    return {
        "self_test": _self_test_status["self_test"],
        "reasons": list(_self_test_status["reasons"]),
        "integrated_lufs": _self_test_status["integrated_lufs"],
        "warning_counts": warning_counts(),
    }
