"""TTS service — multiplexes between Kokoro and Kitten providers."""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess  # nosec B404
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from services.g2p import phonemize_sentence
from services.ssml_processor import SsmlSegment, effective_speed, is_ssml, ssml_to_segments, strip_ssml
from services.text_preproc import normalize_text_for_tts, split_for_tts

logger = logging.getLogger(__name__)

_PHONEMIZER_LINE_COUNT_ERROR = "number of lines in input and output must be equal"

# Inter-sentence silence inserted between synthesized pieces (see synthesize). Kokoro
# silence-trims each batch and butts them with zero gap, which rushes the pacing;
# ~170ms restores a natural sentence pause without sounding stilted. This is also the
# fallback/kill-switch value: with content-pacing OFF, EVERY inter-piece gap is this
# flat constant (byte-identical to the pre-TTSB-2 output).
#
# ⚠️ MATCHED PAIR with ``_TRIM_GAP_COMPENSATION_SECONDS`` / ``_trim_model_padding_enabled``
# — read that constant before touching this one. 0.17 is sized for a world where
# kokoro's own ~0.30 s lead + ~0.44 s trail padding is STILL IN THE AUDIO and supplies
# the rest of the perceived pause. That is true of kokoro-ONNX (which silence-trims its
# own batches, so 0.17 IS the whole gap) and of kokoro-TORCH with the trim off. With the
# torch trim ARMED the padding is gone, and 0.17 alone would leave a 0.25 s inter-sentence
# hole — breathless, the same complaint from the other side. The compensation below is
# what keeps the two in step; it is applied to this value, not baked into it, so the
# trim-off output stays byte-identical.
_GAP_SECONDS = 0.17

# Bounds for any content-derived pause — never shorter than a perceptible breath,
# never long enough to read as dead air / a dropped connection.
_GAP_MIN_SECONDS = 0.05
_GAP_MAX_SECONDS = 0.35

# Hard cap on an SSML <break> silence. A timed break is authored silence (not a
# content-derived inter-piece gap), so it bypasses _GAP_MIN/_GAP_MAX entirely —
# but still cap it so a runaway "10s" break can't emit a huge dead-air block.
#
# ⚠️ 2.5s reads as a dropped connection mid-narration, not an authored pause
# (A/V quality wave 8, plan item 2e) — lowered to 1.2s, still comfortably above
# any real dramatic-pause use of <break> in narration copy.
_BREAK_MAX_SECONDS = 1.2

# Punctuation-set used to peel a piece's trailing punctuation cluster. Includes the
# raw ellipsis/dash forms even though normalize_text_for_tts rewrites most of them
# (… -> ..., em/en-dash -> comma) so _trailing_punct also works on un-normalized
# input (e.g. phoneme-adjacent SSML text, direct unit tests).
_TRAILING_PUNCT_CHARS = frozenset(".!?,;:…—–-")
_TRAILING_STRIP_CHARS = "\"'”’)]}»"  # closing quotes/brackets that trail real punctuation


def _trailing_punct(text: str) -> str | None:
    """Return the run of trailing punctuation on a piece, or None if it has none.

    Strips trailing whitespace and closing quotes/brackets first (so ``he said.\"``
    still reports ``.``), then returns the contiguous tail of punctuation chars.
    """
    s = text.rstrip().rstrip(_TRAILING_STRIP_CHARS)
    if not s:
        return None
    i = len(s)
    while i > 0 and s[i - 1] in _TRAILING_PUNCT_CHARS:
        i -= 1
    tail = s[i:]
    return tail or None


def _gap_seconds_for(trailing_punct: str | None) -> float:
    """Content-derived inter-piece pause from a piece's trailing punctuation.

    A terminal sentence hold after ``.``/``!``; a slightly longer, distinct hold
    after ``?`` (questions want a beat more); a short clause breath after ``,``/
    ``;``/``:``; a floor for trailing-off marks (ellipsis / dash). Anything else
    (incl. no punctuation) uses the neutral ``_GAP_SECONDS`` default. The result is
    always clamped to ``[_GAP_MIN_SECONDS, _GAP_MAX_SECONDS]``.
    """
    if not trailing_punct:
        return _GAP_SECONDS
    p = trailing_punct.strip()
    if not p:
        return _GAP_SECONDS
    # Order matters: '...' ends with '.', and en/em dashes must beat the bare marks.
    if p.endswith(("…", "...", "—", "–", "--")):
        gap = 0.28
    elif p.endswith("?"):
        gap = 0.25
    elif p.endswith((".", "!")):
        gap = 0.22
    elif p.endswith((",", ";", ":")):
        gap = 0.09
    else:
        gap = _GAP_SECONDS
    return min(_GAP_MAX_SECONDS, max(_GAP_MIN_SECONDS, gap))


def _content_pacing_enabled() -> bool:
    """Kill-switch for content-derived inter-piece pacing (TTSB-2), default OFF.

    OFF: every gap is the flat ``_GAP_SECONDS`` — byte-identical to the pre-TTSB-2
    output, the rollback guarantee on this no-CI live path. Set
    ``TTS_CONTENT_PACING_ENABLED=true`` to derive each pause from the preceding
    piece's trailing punctuation. Read per-call (not at import) so it can be flipped
    on-device / in tests without a module reload.
    """
    return os.getenv("TTS_CONTENT_PACING_ENABLED", "false").strip().lower() in ("true", "1", "yes")


# Kill-switch for SSML on /synthesize (default on). A bad SSML parse can be shut
# off without a redeploy by setting TTS_SSML_ENABLED=false — the request then
# treats its text as plain (tags stripped by normalization downstream).
_SSML_ENABLED = os.getenv("TTS_SSML_ENABLED", "true").strip().lower() not in ("false", "0", "no")

# --- Engine selection -------------------------------------------------------
# Two kokoro backends live side by side behind ONE provider, because everything
# except the per-piece model call (SSML lowering, sentence splitting, gap pacing,
# concatenation) is identical between them:
#
#   kokoro-onnx  (DEFAULT)  the shipped kokoro_onnx.Kokoro path. Returns audio
#                           only — the .onnx graph has a single `audio` output,
#                           so it structurally CANNOT report word timings.
#   kokoro-torch            the PyTorch `kokoro` package's KPipeline. Its misaki
#                           MTokens carry per-WORD start_ts/end_ts derived from
#                           the model's predicted durations, which is what lets
#                           /synthesize emit word_timings and delete the
#                           orchestrator's Whisper-ASR round-trip.
#
# Default stays kokoro-onnx: this ships DARK on a no-CI, manually-deployed live
# path, and is flipped on-device with one env var after proof (mirrors the
# TTS_G2P / TTS_CONTENT_PACING_ENABLED kill-switch convention).
_ENGINE_ONNX = "kokoro-onnx"
_ENGINE_TORCH = "kokoro-torch"

# KPipeline emits 24 kHz, same as the ONNX model. Used only to convert a piece's
# sample offsets to seconds BEFORE the outer concat re-bases them.
_TORCH_SAMPLE_RATE = 24000

# Splice de-click. Butt-joining two float arrays lands whatever sample value the
# first one ended on directly against whatever the second one starts on — a step
# discontinuity, i.e. an audible click. Both join loops below did exactly that:
# once per KPipeline sub-chunk inside a piece, and again at every mid-sentence
# piece boundary (<emphasis>, <phoneme>, the word-packer), which carry no gap BY
# DESIGN.
#
# ⚠️ MEASURED, and the measurement did NOT find a click (kokoro-torch, local
# render, 2026-09-11). Every real piece boundary in a 21.6s multi-sentence line
# with two <emphasis> spans sat in the middle of >166 ms of kokoro's own
# near-silent padding on BOTH sides: |prev[-1]| and |next[0]| were 0.000000 at all
# 6 joins, so the butt-join step was 0.00000 — there is nothing there to click. A
# 600-char punctuation-free run-on produced ZERO sub-chunk splits, so that site did
# not fire either. So this is a cheap guard against a splice landing mid-waveform,
# NOT a fix for the reported narration artifacts: if those are still audible, the
# cause is elsewhere and this line should not be cited as having addressed it.
#
# 3 ms at 24 kHz = 72 samples: long enough to take a step out, far too short to
# smear a consonant.
_CROSSFADE_SECONDS = 0.003


# Amplitude below which a sample counts as silence for edge-trimming: -60 dBFS.
# Chosen from the live measurement on mini2 (see _trim_edge_silence) — kokoro's
# head/tail padding sits far below this and its speech far above, so the split is
# not delicate. Deliberately NOT a "near-zero" epsilon: the padding is low-level
# model noise, not digital silence.
_TRIM_THRESHOLD = 1e-3

# How much silence to LEAVE at each trimmed edge. A hard cut at the first loud
# sample would clip a plosive onset or a fricative tail, which is a worse artifact
# than the padding being removed. 40 ms is inaudible as a pause and comfortably
# preserves the attack/release.
_TRIM_RESIDUAL_SECONDS = 0.04

# ⚠️ MATCHED PAIR with ``_GAP_SECONDS`` — NEITHER of these two may be changed alone,
# and the trim kill-switch below is the third member. This value is added to EVERY
# inter-piece pacing gap, and ONLY while the torch edge-trim actually ran.
#
# WHY IT EXISTS: ``_GAP_SECONDS = 0.17`` was sized against the module comment "Kokoro
# silence-trims each batch" — TRUE of kokoro-ONNX, FALSE of kokoro-torch, which is the
# engine that runs live on mini2. Untrimmed, a real inter-sentence gap is
# trail(0.44) + 0.17 + lead(0.30) ~= 0.91 s, which is the "strange pauses" complaint.
# Trimmed, it collapses to 0.17 + 2 x _TRIM_RESIDUAL_SECONDS = 0.25 s, which is the SAME
# complaint inverted — narration that reads clipped and breathless. Natural
# inter-sentence pause in read narration is ~0.4-0.7 s, so the trim without this
# compensation ships an under-corrected version of the defect it is fixing.
#
# 0.18 puts the flat gap at 0.35 s inserted and 0.43 s delivered (0.35 + 2 x 0.04
# residual), i.e. the low end of natural. It is ADDITIVE rather than folded into
# ``_GAP_SECONDS`` for two reasons that are not style:
#   1. the trim-OFF output stays byte-identical, which is the rollback guarantee on a
#      repo whose merge IS the deploy;
#   2. it shifts the content-pacing table (``_gap_seconds_for``) uniformly, so every
#      ordering that table encodes — clause < statement < question < trailing-off —
#      survives untouched instead of needing four literals re-derived by hand.
# It is gated on the trim rather than documented beside it so that arming one without
# the other is not a thing a future reader CAN do.
_TRIM_GAP_COMPENSATION_SECONDS = 0.18


def _trim_model_padding_enabled() -> bool:
    """Kill-switch for trimming kokoro's per-utterance edge padding (default ON).

    Read per-call, not at import, matching the TTS_CONTENT_PACING_ENABLED /
    TTS_SSML_ENABLED convention, so it can be flipped on-device without a code
    change on this no-CI-deploy live path.

    ⚠️ DEFAULT FLIPPED TO ON. It shipped OFF because the trim shortens every clip and
    the caller's speed correction was believed to be modelling a fixed ~150 wpm, so a
    shorter clip would force narration BELOW the 0.80 speed floor. That objection no
    longer stands and was checked in the orchestrator rather than assumed:
    ``speech_rate.resolve_effective_wpm`` is the SINGLE owner of the rate for both the
    word budget and the timeline solver, and it computes
    ``wpm = words / (sum(clip.duration_seconds) / 60)`` — the TOTAL clip duration,
    padding included. So the trim is directly visible to the measurement: the same
    sampled lines come back shorter, the measured rate rises, and the word budget
    grows to fill the same beat. Nothing in THIS repo models a speech rate at all.
    The ``speech_seconds`` / ``pause_seconds`` split this service reports is carried
    as evidence and branched on NOWHERE, so widening the gaps cannot move a budget
    either.

    ⚠️ Arming this ALSO arms ``_TRIM_GAP_COMPENSATION_SECONDS`` — see that constant.
    They are one change with two constants; shipping the trim without the wider gap
    trades padded narration for breathless narration.

    Now that the default is ON, the truthiness test is the DEFAULT-ON form this module
    already uses for ``_SSML_ENABLED`` — an affirmative negative disarms it, anything
    else leaves it armed. A typo in the mini2 env must not silently revert the live
    narrate path to padded output; disarming has to be deliberate.
    """
    return os.getenv("TTS_TRIM_MODEL_PADDING", "true").strip().lower() not in ("false", "0", "no")


def _trim_edge_silence(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Strip kokoro's own leading/trailing silence; return (trimmed, lead_removed).

    ⚠️ WHY: measured live on mini2 (kokoro-torch, af_heart, -60 dBFS), KPipeline
    emits ~0.30 s of lead and ~0.44 s of trail silence around EVERY utterance at
    speed 1.0. Concatenating N of them multiplies that into 35-40% of the file, and
    it lands on top of the deliberate ``_GAP_SECONDS`` — so a real inter-sentence
    gap measured 0.77-0.92 s where the service only meant to insert 0.17 s. The
    module comment on ``_GAP_SECONDS`` ("Kokoro silence-trims each batch") is true
    of kokoro-ONNX and false of the torch path, which trims nothing.

    ⚠️ ONLY the two edges are touched. Slicing a head and a tail cannot, by
    construction, reach anything between them — so kokoro's intra-utterance prosody
    (the ~0.05 s pause it renders for a comma) is structurally safe here, not merely
    safe by a threshold that might drift. That property is why this is an edge trim
    and not a silence-squeezer.

    An all-silent array is returned UNCHANGED: that is either an authored ``<break>``
    or a degenerate chunk, and "trim everything below the threshold" would delete it
    outright. Never destroy audio on the strength of a threshold.

    The second element is the number of LEAD samples removed. The caller MUST
    subtract it from that chunk's token timestamps, which are relative to the
    UNTRIMMED chunk start — forgetting it drifts every caption in the chunk late by
    the padding, which is exactly the class of bug the word-timing tests exist for.
    """
    if samples.size == 0 or sample_rate <= 0:
        return samples, 0
    loud = np.flatnonzero(np.abs(samples) > _TRIM_THRESHOLD)
    if loud.size == 0:
        return samples, 0
    keep = int(_TRIM_RESIDUAL_SECONDS * sample_rate)
    start = max(0, int(loud[0]) - keep)
    end = min(int(samples.size), int(loud[-1]) + 1 + keep)
    if start == 0 and end == samples.size:
        return samples, 0
    return samples[start:end], start


def _crossfade_append(parts: list[np.ndarray], nxt: np.ndarray, sample_rate: int, *, fade: bool = True) -> int:
    """Append ``nxt`` to ``parts`` over a short crossfade; return the samples ADDED.

    The join is an OVERLAP, not an insertion: the last ``n`` samples of the running
    buffer are mixed with the first ``n`` of ``nxt`` under a raised-cosine ramp pair
    (which sums to 1, so the level stays flat through the join) and the buffer grows
    by ``len(nxt) - n``. Inserting a ramp instead would push every later word later.

    ⚠️ The return value — NOT ``len(nxt)`` — is the only correct increment for a
    caller's running ``offset_samples``. Both call sites re-base word timings against
    that offset, so advancing it by the un-overlapped length would drift every
    timing after the join late by one overlap, growing toward the end of the line.

    The overlap is either the full ``n`` or nothing at all: an array shorter than the
    fade is butt-joined rather than partially ramped, which keeps the arithmetic a
    caller can verify and leaves sub-3ms stubs (test fixtures, degenerate pieces)
    byte-identical. ``fade=False`` does the same, and is what authored ``<break>``
    silence and the inserted inter-sentence gaps take: ramping into or out of
    deliberate silence would both shorten the authored pause and fade speech that is
    meant to stop cleanly.
    """
    if nxt.size == 0:
        return 0
    n = int(_CROSSFADE_SECONDS * sample_rate) if sample_rate > 0 else 0
    if not fade or not parts or n <= 0 or parts[-1].size < n or nxt.size < n:
        parts.append(nxt)
        return int(nxt.size)
    prev = parts[-1]
    # Strictly inside (0, 1): ramp endpoints of exactly 0 and 1 would spend two of
    # the n samples reproducing the butt-join this exists to remove.
    ramp = np.arange(1, n + 1, dtype=np.float32) / np.float32(n + 1)
    fade_in = (0.5 - 0.5 * np.cos(np.pi * ramp)).astype(np.float32)
    mixed = prev[-n:] * (np.float32(1.0) - fade_in) + nxt[:n] * fade_in
    parts[-1] = np.concatenate([prev[:-n], mixed]).astype(np.float32)
    parts.append(nxt[n:])
    return int(nxt.size - n)


def _configured_engine() -> str:
    """Resolve ``TTS_ENGINE``; anything unrecognized degrades to the ONNX default.

    Read once per :class:`KokoroProvider` construction (i.e. at service startup),
    not per request: the choice decides WHICH model gets loaded, so a mid-flight
    flip would let ``_render_units`` disagree with the loaded engine. An unknown
    value must never take the live narrate path down, so it warns and falls back
    rather than raising.
    """
    engine = os.getenv("TTS_ENGINE", _ENGINE_ONNX).strip().lower()
    if engine not in (_ENGINE_ONNX, _ENGINE_TORCH):
        logger.warning("Unknown TTS_ENGINE=%r; falling back to %s", engine, _ENGINE_ONNX)
        return _ENGINE_ONNX
    return engine


# kokoro-onnx truncates a phoneme batch at 510 tokens; keep passthrough phoneme
# segments (from <phoneme>) safely under it, splitting on spaces if ever longer.
_MAX_PHONEME_CHARS = 480

# Serialize ALL synthesis through one process-wide lock. The FastAPI route runs
# each /api/tts/synthesize in a worker thread (`asyncio.to_thread(svc.synthesize…)`),
# so concurrent requests land in DIFFERENT OS threads that all call into the
# SINGLE shared provider model (`KokoroProvider._kokoro`) and its espeak
# phonemizer. kokoro-onnx's create() and phonemizer-fork's espeak backend are
# NOT thread-safe — espeak holds process-global state — so overlapping calls
# corrupt each other's phoneme stream and the model emits garbled / repeated
# half-sentences. Proven live (2026-06-03, run 019e8d57): kokoro is fully
# deterministic sequentially (same text → identical duration 20/20), clips
# synthesized at concurrency=1 are always clean, but at concurrency≥2 the
# overlapping requests (in a narrate burst, the opening lines) drift run-to-run
# and lengthen — exactly the "repeats in the first ~50s" defect. A threading.Lock
# (NOT asyncio) is the right primitive because the contention is between worker
# threads; the event loop keeps accepting requests while inference serializes,
# and the box runs a single CPU-bound model anyway so no throughput is lost.
_SYNTH_LOCK = threading.Lock()


def _silence_phonemizer_words_mismatch() -> None:
    """Neutralize phonemizer's fatal words-count-mismatch check.

    kokoro-onnx phonemizes via phonemizer-fork's espeak backend with the
    default ``words_mismatch='ignore'``. Despite the name, ALL THREE word-
    mismatch processors (Ignore/Warn/Remove) call ``_mismatched_lines()``,
    which **raises** ``RuntimeError("number of lines in input and output must
    be equal")`` whenever espeak emits a different number of lines than the
    input had. espeak does that for many ordinary narration sentences — it
    inserts or merges clause-level newlines in its phoneme stream — so the
    raise fires on plain, single-line text (confirmed live: ``input=4,
    output=3`` and ``words count mismatch … (1/1)``). It surfaced as a flood of
    500s on ``/api/tts/synthesize`` under the narrate workload.

    The check is **purely diagnostic accounting**, run AFTER espeak has already
    produced the phonemes. We replace ``_mismatched_lines`` with a non-raising
    version that zips to the shorter list; for the default ``Ignore`` processor
    this returns the phonemes completely unchanged (verified on-device), so the
    audio is identical — we only drop a spurious post-hoc consistency assertion.

    Idempotent: re-assigning the patched method is harmless. Best-effort: if the
    phonemizer internals ever change, the import/attr access fails loudly at
    load rather than silently at synth time.
    """
    from phonemizer.backend.espeak import words_mismatch

    def _non_raising_mismatched_lines(self: Any) -> list[tuple[int, int, int]]:
        return [(n, t, p) for n, (t, p) in enumerate(zip(self._count_txt, self._count_phn, strict=False)) if t != p]

    words_mismatch.BaseWordsMismatch._mismatched_lines = _non_raising_mismatched_lines


def _normalize_for_phonemizer(text: str) -> str:
    """Collapse all whitespace runs (incl. newlines) into single spaces.

    Input hygiene before synthesis: flattens multi-line/blank-line/trailing-
    whitespace input to one clean line while preserving sentence punctuation
    (kokoro's prosody/chunk boundary). NOTE: this does NOT by itself prevent the
    line-count RuntimeError — espeak re-splits a single line internally — that
    is handled by ``_silence_phonemizer_words_mismatch``. ``str.split()`` with
    no args splits on any whitespace and drops empty fragments.
    """
    return " ".join(text.split())


def _split_phoneme_payload(phonemes: str) -> list[str]:
    """Split a phoneme passthrough string into <=510-token pieces on spaces.

    ``<phoneme>`` spans are normally short (a single word), but guard the kokoro
    truncation cap anyway so a long custom-pronunciation string never loses its
    tail. Whitespace-only input yields nothing.
    """
    phonemes = phonemes.strip()
    if not phonemes:
        return []
    if len(phonemes) <= _MAX_PHONEME_CHARS:
        return [phonemes]
    pieces: list[str] = []
    current = ""
    for token in phonemes.split(" "):
        candidate = f"{current} {token}".strip()
        if current and len(candidate) > _MAX_PHONEME_CHARS:
            pieces.append(current)
            current = token
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


@contextmanager
def _bounded_ort_session() -> Generator[None]:
    """Force kokoro-onnx's InferenceSession to disable the CPU memory arena.

    kokoro-onnx 0.5.0 builds ``rt.InferenceSession(model_path, providers=...)``
    with no ``sess_options``, so onnxruntime's CPU arena retains the largest
    working set ever seen and never releases it. On a long-running narration
    worker this ratchets the process footprint up with every long/concurrent
    request (measured: +114 MB per 8k-char synth, never reclaimed → ~7 GB over
    3 days). Disabling the arena trades a little per-inference malloc for a
    bounded, stable footprint. Patch is scoped to the load() call and restored
    immediately. ``ORT_INTRA_OP_THREADS`` optionally pins thread count.
    """
    try:
        import onnxruntime as ort
    except ImportError:  # kokoro-onnx absent too; let the caller's import fail
        yield
        return

    orig = ort.InferenceSession

    def _patched(*args: Any, **kwargs: Any) -> Any:
        if "sess_options" not in kwargs:
            so = ort.SessionOptions()
            so.enable_cpu_mem_arena = False
            intra = int(os.getenv("ORT_INTRA_OP_THREADS", "0") or "0")
            if intra > 0:
                so.intra_op_num_threads = intra
            kwargs["sess_options"] = so
        return orig(*args, **kwargs)

    setattr(ort, "InferenceSession", _patched)  # noqa: B010
    try:
        yield
    finally:
        ort.InferenceSession = orig


@dataclass(frozen=True)
class WordTiming:
    """One spoken word with its start/end offset, in seconds.

    Two timebases use this type, and they must not be confused:

    * inside :meth:`KokoroProvider._create_piece_torch` the offsets are relative
      to THAT PIECE's audio;
    * on :class:`SynthesisResult` they have been re-based onto the FINAL
      concatenated clip (see the concat loop in ``KokoroProvider.synthesize``).
    """

    word: str
    start: float
    end: float


@dataclass
class SynthesisResult:
    audio_pcm: np.ndarray  # float32, mono
    sample_rate: int
    duration_sec: float
    provider: str
    voice: str
    model: str = ""  # provider-specific identifier (Kokoro ONNX path / Kitten HF repo id)
    # Per-word offsets into `audio_pcm`, in seconds at `sample_rate`. EMPTY unless
    # the kokoro-torch engine served the request (ONNX cannot produce them); also
    # empty for the parts of a clip that came from <phoneme> passthrough, which
    # carries no word tokens. Never None, so callers can iterate unconditionally.
    word_timings: list[WordTiming] = field(default_factory=list)
    # Seconds of `audio_pcm` that are silence THIS SERVICE inserted deliberately:
    # the inter-piece pacing gap (flat _GAP_SECONDS, or content-derived when the
    # TTS_CONTENT_PACING_ENABLED kill-switch is on) plus any authored SSML <break>
    # duration. Summed from the exact sample counts the concat loop appended —
    # never re-measured from the waveform — so it is exact and free, and it does
    # NOT include kokoro's own per-piece near-silent head/tail padding.
    #
    # ⚠️ That exclusion is a MEASURED decision, not an assumption. Measured live
    # on mini2 (kokoro-torch, af_heart, 4-sentence line, -60 dBFS threshold):
    # kokoro emits ~0.30 s lead + ~0.44 s trail of silence PER PIECE at speed
    # 1.0, falling to ~0.23 s / ~0.24 s at speed 1.81 — i.e. it is ~93% of
    # proportionally scaled BY `speed`, so it behaves like speech under a speed
    # correction and belongs in the denominator. Only the fixed zero-sample runs
    # counted here are genuinely unscaled. See tests/test_pause_accounting.py.
    #
    # ⚠️ The `_GAP_SECONDS` comment above ("Kokoro silence-trims each batch and
    # butts them with zero gap") is true of kokoro-ONNX and FALSE of the
    # kokoro-torch path that runs live: `_create_piece_torch` trims nothing, so
    # a real inter-sentence gap is trail(prev) + _GAP_SECONDS + lead(next) —
    # measured 0.77-0.92 s, of which this service contributes only the 0.17 s.
    # Reported, deliberately not retuned here.
    #
    # 0.0 for providers (kitten) that insert no pacing silence at all.
    pause_seconds: float = 0.0

    @property
    def speech_seconds(self) -> float:
        """``duration_sec`` minus the pacing silence this service inserted."""
        return max(0.0, self.duration_sec - self.pause_seconds)


@dataclass
class _RenderUnit:
    """One ordered render unit for the kokoro concatenation pipeline.

    A normal unit carries a ``payload`` to synthesize (text or phonemes) plus the
    ``gap`` silence to insert AFTER it. A ``break_ms``-carrying unit is authored
    SSML silence: it calls no kokoro ``create()`` and inserts no surrounding gap —
    the break itself IS the silence.

    ``sentence_final`` records whether the piece actually ended on punctuation. A
    unit boundary is not by itself a place to pause: ``<emphasis>``/``<phoneme>``
    and the word-packer all split MID-sentence, and inserting an inter-piece gap
    there is audible dead air inside a phrase (~340ms around every emphasized
    word, since such a span is bounded on both sides). Only a piece that ends a
    sentence or clause earns a gap.
    """

    payload: str
    is_phonemes: bool
    speed: float
    gap: float
    break_ms: int | None = None
    sentence_final: bool = True


class ProviderUnavailableError(RuntimeError):
    pass


class _Provider(Protocol):
    name: str
    default_voice: str
    voices: list[str]

    def is_loaded(self) -> bool: ...
    def synthesize(self, text: str, voice: str | None, speed: float, ssml: bool | None = None) -> SynthesisResult: ...


class KokoroProvider:
    name = "kokoro"

    def __init__(self) -> None:
        self.default_voice = os.getenv("KOKORO_DEFAULT_VOICE", "af_heart")
        self.model_path = os.getenv("KOKORO_MODEL_PATH", "/Users/mini2/.cache/tts/kokoro-v1.0.onnx")
        self.voices_path = os.getenv("KOKORO_VOICES_PATH", "/Users/mini2/.cache/tts/voices-v1.0.bin")
        self.engine = _configured_engine()
        # kokoro-torch only. lang_code "a" = American English (matches af_* voices).
        self.torch_lang_code = os.getenv("KOKORO_TORCH_LANG_CODE", "a")
        # None → the `kokoro` package's own default weights repo. Pin via env if the
        # deploy host must resolve the model from a mirror / local snapshot.
        self.torch_repo_id = os.getenv("KOKORO_TORCH_REPO_ID") or None
        self.voices: list[str] = []
        self._kokoro: Any = None
        self._pipeline: Any = None  # kokoro-torch KPipeline
        self.custom_voices: dict[str, np.ndarray] = {}

    def is_loaded(self) -> bool:
        return self._kokoro is not None or self._pipeline is not None

    def emits_word_timings(self) -> bool:
        """True when this provider's engine can report per-word offsets."""
        return self.engine == _ENGINE_TORCH or bool(getattr(self._kokoro, "has_timings", False))

    def load(self) -> None:
        if self.engine == _ENGINE_TORCH:
            self._load_torch()
            self._load_custom_voices()
            return
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise ProviderUnavailableError("kokoro-onnx not installed") from exc
        if not os.path.exists(self.model_path):
            raise ProviderUnavailableError(f"Kokoro model file missing at {self.model_path}")
        if not os.path.exists(self.voices_path):
            raise ProviderUnavailableError(f"Kokoro voices file missing at {self.voices_path}")
        with _bounded_ort_session():
            self._kokoro = Kokoro(self.model_path, self.voices_path)
        # kokoro-onnx imported phonemizer above; neutralize its fatal (and
        # purely diagnostic) words-count-mismatch raise that was flooding
        # /synthesize with 500s on ordinary narration text.
        _silence_phonemizer_words_mismatch()
        self.voices = list(self._kokoro.get_voices())
        if os.getenv("KOKORO_REQUIRE_TIMINGS", "false").lower() == "true" and not self.emits_word_timings():
            self._kokoro = None
            raise ProviderUnavailableError("Kokoro ONNX graph must expose native durations")
        self._load_custom_voices()
        logger.info(
            "KokoroProvider loaded engine=%s model=%s voices=%d word_timings=%s",
            self.engine,
            self.model_path,
            len(self.voices),
            self.emits_word_timings(),
        )

    def _load_custom_voices(self) -> None:
        if self.engine == _ENGINE_TORCH and os.path.isfile(self.voices_path):
            with np.load(self.voices_path, allow_pickle=False) as stock:
                self.voices = sorted(stock.files)
        directory = Path(os.getenv("KOKORO_CUSTOM_VOICES_DIR", "~/.local/share/tts/voices")).expanduser()
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.npy")):
            voice_id = path.stem
            if not re.fullmatch(r"(?:af|am|bf|bm)_[a-z0-9][a-z0-9_]{0,63}", voice_id):
                raise ProviderUnavailableError(f"Invalid custom voice ID: {voice_id}")
            if voice_id in self.voices:
                raise ProviderUnavailableError(f"Custom voice shadows stock voice: {voice_id}")
            pack = np.load(path, allow_pickle=False)
            if pack.shape != (510, 1, 256) or pack.dtype != np.float32 or not np.isfinite(pack).all():
                raise ProviderUnavailableError(f"Invalid custom voice tensor: {voice_id}")
            self.custom_voices[voice_id] = pack
        self.voices = sorted(set(self.voices) | self.custom_voices.keys())

    def _resolve_voice(self, voice: str) -> Any:
        pack = self.custom_voices.get(voice)
        if pack is None:
            return voice
        if self.engine == _ENGINE_TORCH:
            import torch

            return torch.from_numpy(pack)
        return pack

    def _create_piece_onnx_timed(
        self, payload: str, voice: str, speed: float, is_phonemes: bool
    ) -> tuple[np.ndarray, int, list[WordTiming]]:
        from services.onnx_words import tokenize, words_from_timings

        tokens = []
        phonemes = payload
        if not is_phonemes:
            phonemes, tokens = tokenize(payload, british=voice.startswith("b"))
        if not phonemes or not self._kokoro.tokenizer.known(phonemes).strip():
            raise ValueError("Text contains no pronounceable phonemes")
        samples, sr, timings = self._kokoro.create_timed(
            phonemes,
            voice=self._resolve_voice(voice),
            speed=speed,
            is_phonemes=True,
            trim=False,
            sentence_pause=0.0,
            clause_pause=0.0,
        )
        words = words_from_timings(tokens, timings, self._kokoro.tokenizer.known) if tokens else []
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise RuntimeError("Kokoro ONNX produced empty or non-finite audio")
        lead = 0.0
        if _trim_model_padding_enabled():
            samples, removed = _trim_edge_silence(samples, sr)
            lead = removed / sr
        duration = len(samples) / sr
        return (
            samples,
            sr,
            [
                WordTiming(word, min(duration, max(0.0, start - lead)), min(duration, max(0.0, end - lead)))
                for word, start, end in words
            ],
        )

    def _load_torch(self) -> None:
        """Build the PyTorch KPipeline (TTS_ENGINE=kokoro-torch).

        Unlike the ONNX path there are no model/voices FILES to check: `kokoro`
        resolves its weights through huggingface_hub (cached on the deploy host),
        so a missing model surfaces as a load-time exception, which we translate
        into :class:`ProviderUnavailableError` so ``TTSService.initialize`` degrades
        to "no kokoro provider" exactly as it does for a missing ONNX file — rather
        than taking the whole service down at boot.

        NOTE on spaCy: KPipeline's misaki G2P needs the ``en_core_web_sm`` model,
        which misaki otherwise fetches at __init__ via ``spacy.cli.download`` — and
        that download shells out to pip, which a ``uv venv`` does not have ("No
        package installer found"). This repo already pins ``en_core_web_sm`` as a
        real wheel dependency in pyproject.toml (added for services/g2p.py), so the
        model is present at import time and no runtime download is attempted. Do not
        drop that pin: without it kokoro-torch is a latent STARTUP failure.
        """
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise ProviderUnavailableError("kokoro (PyTorch package) not installed") from exc
        try:
            self._pipeline = KPipeline(lang_code=self.torch_lang_code, repo_id=self.torch_repo_id)
        except Exception as exc:  # noqa: BLE001 — hub/model/torch failures are all "unavailable"
            raise ProviderUnavailableError(f"kokoro-torch KPipeline unavailable: {exc}") from exc
        # KPipeline has no voice registry to enumerate; it loads a voice on demand.
        # Leaving `voices` empty makes synthesize() skip the membership check and
        # lets kokoro itself reject an unknown voice (surfaced as a 400 by the route).
        self.voices = []
        logger.info(
            "KokoroProvider loaded engine=%s lang_code=%s repo_id=%s",
            self.engine,
            self.torch_lang_code,
            self.torch_repo_id or "<default>",
        )

    @staticmethod
    def _is_word(text: str) -> bool:
        """Keep real words out of punctuation-only tokens (design rule 5).

        misaki already leaves ``start_ts``/``end_ts`` as ``None`` on tokens with no
        phonemes — which is exactly the punctuation case — so this is belt-and-braces
        for anything that slips through with a timestamp but no pronounceable text.
        """
        return any(ch.isalnum() for ch in text)

    def _create_piece_torch(
        self, payload: str, voice: str, speed: float, is_phonemes: bool
    ) -> tuple[np.ndarray, int, list[WordTiming]]:
        """Synthesize one piece with KPipeline, returning audio + PIECE-relative timings.

        ⚠️ ONE call can yield MANY results. ``KPipeline.__call__`` splits its input
        (on ``split_pattern``, then again via ``en_tokenize`` to stay under the
        model's token cap) and yields one ``Result`` per sub-chunk. Crucially,
        ``KPipeline.join_timestamps`` is invoked PER RESULT and restarts its
        accumulator at ~0 each time, so every chunk's ``start_ts``/``end_ts`` are
        CHUNK-relative. They are therefore re-based here against the running sample
        offset before being handed up — the same problem the outer concat loop solves
        one level higher.

        The offset is advanced by the ACTUAL emitted sample count, never by the
        timestamps themselves: the timestamps are derived from the model's predicted
        durations and can disagree with the rendered audio by a frame or two, and
        trusting them for the offset would let that error ACCUMULATE across chunks.
        Advancing by real samples keeps any such error bounded inside its own chunk.

        A ``<phoneme>`` passthrough goes through ``generate_from_tokens``, which
        yields ``tokens=None`` — audio only, no timings. That is reported honestly
        (an empty list) rather than guessed at.
        """
        trim_padding = _trim_model_padding_enabled()
        chunks: list[np.ndarray] = []
        words: list[WordTiming] = []
        offset_samples = 0
        if is_phonemes:
            results = self._pipeline.generate_from_tokens(payload, voice=self._resolve_voice(voice), speed=speed)
        else:
            results = self._pipeline(payload, voice=self._resolve_voice(voice), speed=speed)
        # Materialized (not iterated lazily): KPipeline may yield a generator, and
        # the audio=None check below needs a stable count for a useful log message.
        results = list(results)
        for i, res in enumerate(results):
            audio = getattr(res, "audio", None)
            if audio is None:
                # KPipeline yielded a sub-chunk with NO audio (a real observed
                # failure mode, not hypothetical). Silently `continue`-ing here
                # used to drop the whole sub-chunk AND its word timings with no
                # log, no retry, no error — the caller got a 200 with a shorter
                # file and text that was never spoken, and nothing downstream
                # compares returned audio length against the input text. Fail
                # loudly instead: a caller that gets a 200 must be able to trust
                # every word it asked for was actually rendered.
                logger.error(
                    "kokoro-torch KPipeline sub-chunk %d/%d returned audio=None for payload=%r (voice=%s, speed=%s)",
                    i + 1,
                    len(results),
                    payload,
                    voice,
                    speed,
                )
                raise RuntimeError(
                    f"kokoro-torch produced no audio for sub-chunk {i + 1}/{len(results)} "
                    f"of piece {payload!r} — text would be silently dropped"
                )
            # KPipeline hands back a torch tensor (possibly on MPS); detach+cpu is a
            # no-op for the plain arrays the unit tests feed in.
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            samples = np.asarray(audio, dtype=np.float32)
            # Strip kokoro's own edge padding (kill-switch, default OFF). This is the
            # ONE place the torch path can do it: every KPipeline sub-chunk is its own
            # utterance and carries its own head/tail silence, so trimming per PIECE
            # would leave the sub-chunk joins inside a piece still padded.
            lead_sec = 0.0
            if trim_padding:
                samples, lead_removed = _trim_edge_silence(samples, _TORCH_SAMPLE_RATE)
                lead_sec = lead_removed / _TORCH_SAMPLE_RATE
            # Crossfade BEFORE re-basing: the overlap pulls this chunk's start
            # earlier by exactly the samples it did not add, and the tokens below
            # are re-based against that start.
            added = _crossfade_append(chunks, samples, _TORCH_SAMPLE_RATE)
            base = (offset_samples - (int(samples.size) - added)) / _TORCH_SAMPLE_RATE
            for tok in getattr(res, "tokens", None) or ():
                start, end = getattr(tok, "start_ts", None), getattr(tok, "end_ts", None)
                text = (getattr(tok, "text", None) or "").strip()
                if start is None or end is None or not self._is_word(text):
                    continue
                # ⚠️ `start_ts`/`end_ts` are relative to the UNTRIMMED chunk, so the
                # lead we just removed has to come back off them. Omitting this
                # leaves every caption in the chunk late by the padding — invisible
                # on a one-chunk clip and compounding across a long line.
                words.append(
                    WordTiming(
                        word=text,
                        start=max(0.0, base + float(start) - lead_sec),
                        end=max(0.0, base + float(end) - lead_sec),
                    )
                )
            offset_samples += added
        audio_pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return audio_pcm, _TORCH_SAMPLE_RATE, words

    def _create_piece(self, payload: str, voice: str, speed: float, is_phonemes: bool) -> tuple[np.ndarray, int]:
        """Call kokoro.create for one piece, wrapping the espeak line-count raise."""
        try:
            if is_phonemes:
                samples, sr = self._kokoro.create(
                    payload, voice=self._resolve_voice(voice), speed=speed, is_phonemes=True
                )
            else:
                samples, sr = self._kokoro.create(payload, voice=self._resolve_voice(voice), speed=speed)
        except RuntimeError as exc:
            # espeak line-count raise should not survive normalization, but if it
            # trips, make it diagnosable instead of an opaque 500. Re-raise anything
            # else untouched.
            if _PHONEMIZER_LINE_COUNT_ERROR in str(exc):
                raise RuntimeError(
                    f"{_PHONEMIZER_LINE_COUNT_ERROR} (persisted after normalization, piece_len={len(payload)})"
                ) from exc
            raise
        return np.asarray(samples, dtype=np.float32), sr

    def _render_units(self, segments: list[SsmlSegment], speed: float) -> list[_RenderUnit]:
        """Lower SSML/plain segments to ordered create-units (:class:`_RenderUnit`).

        Plain segments go through the normal normalize → split → misaki-or-espeak
        pipeline; phoneme segments (from ``<phoneme>``) bypass straight to kokoro.
        A ``break_ms`` segment becomes a silence-only unit (no kokoro call).
        ``gap`` is the content-derived silence to insert AFTER this unit (from the
        piece's trailing punctuation). Phoneme/IPA units carry no recoverable
        sentence punctuation, so they fall back to the ``_GAP_SECONDS`` default. The
        gap is only applied when content-pacing is enabled (see ``synthesize``);
        with the kill-switch off every gap becomes the flat ``_GAP_SECONDS``.
        """
        units: list[_RenderUnit] = []
        for seg in segments:
            if seg.break_ms is not None:
                # Authored SSML silence — no create(); the concat step splices it in
                # with no surrounding gap (see synthesize).
                units.append(_RenderUnit("", False, speed, 0.0, break_ms=seg.break_ms))
                continue
            seg_speed = effective_speed(seg, speed)
            if seg.is_phonemes:
                # Already phonemes — bypass normalize/split; only guard the 510 cap.
                # IPA carries no recoverable sentence punctuation, so a phoneme span
                # is treated as mid-sentence: no gap after it.
                for chunk in _split_phoneme_payload(seg.payload):
                    units.append(_RenderUnit(chunk, True, seg_speed, _gap_seconds_for(None), sentence_final=False))
                continue
            clean = _normalize_for_phonemizer(seg.payload)
            if not clean:
                continue
            for piece in split_for_tts(normalize_text_for_tts(clean)):
                # Drop a piece with nothing pronounceable in it (emoji-only,
                # punctuation-only, a lone symbol). Such a piece phonemises to the
                # empty string, and on the ONNX engine the VENDORED kokoro_onnx then
                # does a bare `np.concatenate([])` over zero phoneme batches and
                # raises "need at least one array to concatenate" — surfaced to the
                # caller as a raw 400. That killed E2E `showcase` on 2026-08-25 at
                # demo_video_narrate, and the orchestrator burned BOTH stage retries
                # re-issuing the identical unspeakable text, which can never succeed.
                #
                # `_is_word` is the class's existing "has an alnum character"
                # predicate, already used to keep punctuation out of word timings.
                #
                # If EVERY piece drops out, `synthesize` still raises its own
                # "Text is empty after normalization" — a clear 400 with a real
                # message, instead of numpy speaking for us.
                if not self._is_word(piece):
                    logger.debug("skipping unpronounceable piece (%d chars)", len(piece))
                    continue
                # Trailing punctuation is recoverable from the pre-phonemized piece
                # (before misaki/espeak turns it into IPA), so derive the gap here.
                trailing = _trailing_punct(piece)
                gap = _gap_seconds_for(trailing)
                # Prefer misaki G2P (dictionary-quality proper nouns / initialisms /
                # homographs) when TTS_G2P=misaki. A miss returns None → fall back
                # to kokoro's espeak path; misaki can never turn a working narration
                # into a 500.
                #
                # EXCEPT on kokoro-torch: KPipeline runs misaki INTERNALLY and only
                # emits word timings when it did the grapheme→phoneme step itself
                # (generate_from_tokens on a pre-made phoneme string yields
                # tokens=None). Pre-phonemizing here would therefore silently strip
                # the timings this engine exists to produce, so the torch path always
                # hands kokoro the TEXT. Pronunciation is unaffected — it is the same
                # misaki lexicon either way. Explicit <phoneme> SSML spans are
                # untouched (handled above) and still bypass to the model.
                phon = None if self.emits_word_timings() else phonemize_sentence(piece)
                if phon:
                    units.append(_RenderUnit(phon, True, seg_speed, gap, sentence_final=trailing is not None))
                else:
                    units.append(_RenderUnit(piece, False, seg_speed, gap, sentence_final=trailing is not None))
        return units

    def synthesize(self, text: str, voice: str | None, speed: float, ssml: bool | None = None) -> SynthesisResult:
        if not self.is_loaded():
            raise ProviderUnavailableError("Kokoro not loaded")
        v = voice or self.default_voice
        # Validate before calling kokoro-onnx, whose create() raises a bare
        # AssertionError on an unknown voice (would surface as a 500).
        if self.voices and v not in self.voices:
            raise ValueError(f"Unknown voice '{v}' for kokoro. Valid voices: {', '.join(self.voices)}")

        # SSML: explicit flag wins; else auto-detect. Kill-switch forces plain text.
        use_ssml = (ssml if ssml is not None else is_ssml(text)) and _SSML_ENABLED
        if use_ssml:
            # Lower SSML to ordered segments (plain text runs + phoneme spans +
            # per-span prosody speed). Plain segments still flow through the normal
            # normalize/split/misaki pipeline below; phoneme spans bypass to kokoro.
            segments = ssml_to_segments(text)
        else:
            # SSML present but NOT honored (kill-switch TTS_SSML_ENABLED=false, or an
            # explicit ssml=false): STRIP the markup to plain text first. Without this
            # the raw <speak>/<break>/<emphasis> tags flow into the phonemizer and get
            # SPOKEN ALOUD — a footgun because a caller (the orchestrator) may still be
            # emitting SSML when only the backend switch is flipped off. strip_ssml
            # yields the clean prose; a plain line (is_ssml false) is untouched.
            plain = strip_ssml(text) if is_ssml(text) else text
            # Input hygiene: flatten to one clean line. (The "number of lines …"
            # 500 is prevented by _silence_phonemizer_words_mismatch, set at load.)
            # A request that is only whitespace is a bad request, not a 500.
            clean = _normalize_for_phonemizer(plain)
            if not clean:
                raise ValueError("Text is empty after whitespace normalization")
            segments = [SsmlSegment(payload=clean)]

        units = self._render_units(segments, speed)
        if not units:
            raise ValueError("Text is empty after normalization")
        # A request whose ONLY units are authored <break> silences renders a
        # perfectly valid, perfectly silent clip — and used to return 200 with a
        # metered cost event for it. Silence is not synthesis: report it as the bad
        # request it is rather than billing the caller for dead air it cannot use.
        if all(unit.break_ms is not None for unit in units):
            raise ValueError("Text contains no speakable content (only SSML <break> silence)")

        # (audio, gap_after, is_break, piece_relative_word_timings, sentence_final)
        # per rendered unit. A break unit is authored silence: no kokoro create(), no
        # word timings, and it suppresses the inter-piece gap on both sides (the
        # break itself IS the pause). ``sentence_final`` suppresses the gap after a
        # piece that did not end a sentence/clause — see :class:`_RenderUnit`.
        rendered: list[tuple[np.ndarray, float, bool, list[WordTiming], bool]] = []
        sr = 24000
        use_torch = self.engine == _ENGINE_TORCH
        for unit in units:
            if unit.break_ms is not None:
                break_secs = min(unit.break_ms / 1000, _BREAK_MAX_SECONDS)
                rendered.append((np.zeros(int(break_secs * sr), dtype=np.float32), 0.0, True, [], True))
                continue
            if use_torch:
                samples, sr, piece_words = self._create_piece_torch(unit.payload, v, unit.speed, unit.is_phonemes)
            elif self.emits_word_timings():
                samples, sr, piece_words = self._create_piece_onnx_timed(unit.payload, v, unit.speed, unit.is_phonemes)
            else:
                samples, sr = self._create_piece(unit.payload, v, unit.speed, unit.is_phonemes)
                piece_words = []
            rendered.append((samples, unit.gap, False, piece_words, unit.sentence_final))
        # Concatenate the pieces with an inserted inter-sentence silence gap so
        # kokoro's zero-gap butting no longer rushes/clips the pacing. No trailing
        # gap after the final piece — that would pad every standalone clip.
        # Kill-switch OFF (default): every gap is the flat _GAP_SECONDS. ON: each gap
        # is derived from the preceding piece's trailing punctuation. A break
        # neighbour on either side suppresses the gap, and so does a preceding piece
        # that did not end a sentence/clause — in BOTH modes. (That last rule is the
        # one deliberate departure from "byte-identical to pre-TTSB-2": the old
        # output padded ~170ms of dead air into the middle of a phrase wherever
        # <emphasis>/<phoneme> or the word-packer forced a unit boundary.)
        #
        # ⚠️ Word timings are re-based IN THIS SAME LOOP, and deliberately so. Each
        # piece's timings are relative to that piece, so the whole feature hinges on
        # adding the exact number of samples that precede the piece in the final
        # buffer. The offset is therefore accumulated from ``len()`` of the very
        # arrays being appended to ``joined`` — not recomputed from durations or
        # from the gap constants — so it cannot drift from the audio no matter which
        # branch a unit takes: silence-only <break> units advance it (their zeros are
        # in ``rendered``), a flat gap advances it by its own length, a content-paced
        # gap by ITS length, and a suppressed gap advances it by nothing at all.
        # Getting this wrong is invisible at the start of a line and grows toward the
        # end, which is precisely the drift this feature exists to remove.
        content_pacing = _content_pacing_enabled()
        # The other half of the edge-trim change, and the reason it cannot ship alone.
        # The trim removes ~0.74 s of kokoro padding from between every pair of pieces,
        # of which the gap constants were silently relying on ~0.66 s; `_GAP_SECONDS`
        # alone would then leave a 0.25 s hole where narration wants ~0.4-0.7 s. Added
        # here, at the ONE place a gap becomes samples, so it lands on the flat gap and
        # on every content-paced gap identically — see `_TRIM_GAP_COMPENSATION_SECONDS`.
        #
        # Gated on `use_torch` as well as the flag because the trim itself is: only
        # `_create_piece_torch` trims, kokoro-ONNX silence-trims its own batches, so
        # compensating on the ONNX path would widen gaps that never lost anything.
        gap_bonus = (
            _TRIM_GAP_COMPENSATION_SECONDS if (self.emits_word_timings() and _trim_model_padding_enabled()) else 0.0
        )
        flat_gap = np.zeros(int((_GAP_SECONDS + gap_bonus) * sr), dtype=np.float32)
        joined: list[np.ndarray] = []
        word_timings: list[WordTiming] = []
        offset_samples = 0
        # Exact count of deliberately-inserted silence samples: every authored
        # <break> unit's zeros (rendered[i] itself, is_break=True) plus every
        # inter-piece pacing gap appended below. Never derived from durations or
        # re-measured from the waveform, so it cannot drift from what was actually
        # spliced in.
        pause_samples = 0
        # Whether the last thing appended was speech. A crossfade is only correct
        # between two speech arrays: authored <break> silence and the inserted gaps
        # are deliberate pauses, and ramping into or out of one would both eat into
        # the pause and blur an ending that is supposed to be clean.
        prev_is_speech = False
        for i, (seg, gap_after, is_break, piece_words, sentence_final) in enumerate(rendered):
            added = _crossfade_append(joined, seg, sr, fade=prev_is_speech and not is_break)
            # The overlap (if any) moved this piece's start earlier by exactly the
            # samples it did not add — re-base against THAT, not the raw offset.
            offset_sec = (offset_samples - (int(seg.size) - added)) / sr if sr else 0.0
            word_timings.extend(
                WordTiming(word=w.word, start=offset_sec + w.start, end=offset_sec + w.end) for w in piece_words
            )
            offset_samples += added
            if is_break:
                # Authored SSML silence — the whole unit IS the pause. `added` (not
                # seg.size) matches what actually landed in `joined`, though the two
                # agree here since a break is always butt-joined (fade=False above).
                pause_samples += added
            if seg.size:
                prev_is_speech = not is_break
            if i < len(rendered) - 1:
                if is_break or rendered[i + 1][2]:
                    continue  # authored break silence stands alone; no extra gap
                if not sentence_final:
                    continue  # mid-sentence boundary (emphasis/phoneme/word-pack): no pause
                gap = np.zeros(int((gap_after + gap_bonus) * sr), dtype=np.float32) if content_pacing else flat_gap
                joined.append(gap)
                offset_samples += len(gap)
                pause_samples += len(gap)
                if gap.size:
                    prev_is_speech = False
        audio = np.concatenate(joined) if joined else np.zeros(0, dtype=np.float32)
        duration_sec = float(len(audio) / sr) if sr else 0.0
        pause_seconds = float(pause_samples / sr) if sr else 0.0
        # Clamp to the clip: a token's end_ts comes from PREDICTED durations and can
        # overshoot the last piece's real audio by a frame. A caption that runs past
        # the end of the file is a consumer bug waiting to happen, and clamping costs
        # at most a couple of milliseconds on the final word.
        word_timings = [
            WordTiming(word=w.word, start=min(max(w.start, 0.0), duration_sec), end=min(max(w.end, 0.0), duration_sec))
            for w in word_timings
        ]
        return SynthesisResult(
            audio_pcm=audio,
            sample_rate=int(sr),
            duration_sec=duration_sec,
            provider=self.name,
            voice=v,
            # Metering reads this: report what actually synthesized the clip, not the
            # ONNX file path, which is not even loaded on the torch path.
            model=(self.torch_repo_id or _ENGINE_TORCH) if use_torch else self.model_path,
            word_timings=word_timings,
            pause_seconds=pause_seconds,
        )


# Matches the premaster loudnorm measurement pass (post_processor, timeout=60).
_ATEMPO_TIMEOUT_S = 60


def _time_stretch(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Change tempo by ``speed`` WITHOUT changing pitch, via ffmpeg ``atempo``.

    The request model bounds speed to [0.5, 2.0], which is exactly the range one
    ``atempo`` instance accepts. ffmpeg is already a runtime dependency (the
    premaster pass). If it is missing we refuse rather than fall back to a
    resample: a pitch-shifted clip is a wrong deliverable, not a degraded one.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ProviderUnavailableError("kitten speed != 1.0 requires ffmpeg (atempo), which is not on PATH")
    fmt = ["-f", "f32le", "-ar", str(sample_rate), "-ac", "1"]
    try:
        proc = subprocess.run(  # noqa: S603  # nosec B603 — fixed argv, no shell
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                *fmt,
                "-i",
                "pipe:0",
                "-filter:a",
                f"atempo={speed}",
                *fmt,
                "pipe:1",
            ],
            input=np.ascontiguousarray(audio, dtype="<f4").tobytes(),
            capture_output=True,
            check=False,
            timeout=_ATEMPO_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        # Same surface as a non-zero exit: a normal error, never a resample fallback.
        raise RuntimeError(f"ffmpeg atempo timed out after {_ATEMPO_TIMEOUT_S}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg atempo failed (rc={proc.returncode}): {proc.stderr.decode(errors='replace')[-300:]}"
        )
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


class KittenProvider:
    name = "kitten"

    def __init__(self) -> None:
        self.default_voice = os.getenv("KITTEN_DEFAULT_VOICE", "expr-voice-2-f")
        self.model_id = os.getenv("KITTEN_MODEL_ID", "KittenML/kitten-tts-mini-0.8")
        self.voices: list[str] = []
        self._pipeline: Any = None
        self._sample_rate: int = 24000

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        try:
            # Kitten ships as an ONNX-style runtime; the import path may evolve.
            # We intentionally lazy-import here so missing weights don't crash the service.
            from kittentts import KittenTTS
        except ImportError as exc:
            raise ProviderUnavailableError("kittentts not installed") from exc
        self._pipeline = KittenTTS(self.model_id)
        self.voices = list(getattr(self._pipeline, "available_voices", []) or [])
        logger.info("KittenProvider loaded model=%s voices=%d", self.model_id, len(self.voices))

    def synthesize(self, text: str, voice: str | None, speed: float, ssml: bool | None = None) -> SynthesisResult:
        if self._pipeline is None:
            raise ProviderUnavailableError("Kitten not loaded")
        v = voice or self.default_voice
        if self.voices and v not in self.voices:
            raise ValueError(f"Unknown voice '{v}' for kitten. Valid voices: {', '.join(self.voices)}")
        # Kitten has no SSML support — strip tags to plain text whenever present,
        # regardless of the TTS_SSML_ENABLED kill-switch: kitten can NEVER speak
        # markup, so leaving tags in (the switch-off case) would voice them aloud.
        if ssml if ssml is not None else is_ssml(text):
            text = strip_ssml(text)
        samples = self._pipeline.generate(text, voice=v)
        audio = np.asarray(samples, dtype=np.float32)
        # Kitten doesn't expose speed natively. Time-stretch with ffmpeg ``atempo``
        # (pitch-preserving). This used to be a linear np.interp resample, which
        # shifts PITCH with speed (0.5 = an octave down) and aliases.
        if speed != 1.0 and len(audio) > 0:
            audio = _time_stretch(audio, self._sample_rate, speed)
        return SynthesisResult(
            audio_pcm=audio,
            sample_rate=self._sample_rate,
            duration_sec=float(len(audio) / self._sample_rate) if self._sample_rate else 0.0,
            provider=self.name,
            voice=v,
            model=self.model_id,
        )


class TTSService:
    def __init__(self) -> None:
        self.default_provider: str = os.getenv("TTS_DEFAULT_PROVIDER", "kokoro")
        self.kokoro_enabled: bool = os.getenv("KOKORO_ENABLED", "true").lower() == "true"
        # Default OFF: kittentts is not installed on the deployment, and an
        # eager second-model load at boot would double the resident footprint
        # on the shared 16 GB host. Opt in explicitly via KITTEN_ENABLED=true.
        self.kitten_enabled: bool = os.getenv("KITTEN_ENABLED", "false").lower() == "true"
        self.providers: dict[str, _Provider] = {}

    async def initialize(self) -> None:
        if self.kokoro_enabled:
            kp = KokoroProvider()
            try:
                kp.load()
                self.providers[kp.name] = kp
            except ProviderUnavailableError as exc:
                logger.warning("Kokoro unavailable: %s", exc)
        if self.kitten_enabled:
            kt = KittenProvider()
            try:
                kt.load()
                self.providers[kt.name] = kt
            except ProviderUnavailableError as exc:
                logger.warning("Kitten unavailable: %s", exc)

    def loaded_models(self) -> list[str]:
        return [p.name for p in self.providers.values() if p.is_loaded()]

    def synthesize(
        self,
        text: str,
        provider: str | None,
        voice: str | None,
        speed: float,
        ssml: bool | None = None,
    ) -> SynthesisResult:
        target = provider or self.default_provider
        prov = self.providers.get(target)
        if prov is None or not prov.is_loaded():
            available = ", ".join(self.providers.keys()) or "none"
            raise ProviderUnavailableError(f"Provider '{target}' not available. Loaded: {available}")
        # Serialize the model call — concurrent synthesis on the shared kokoro
        # instance + non-thread-safe espeak phonemizer garbles output. See
        # _SYNTH_LOCK. WAV encoding (encode_wav) happens in the route, outside
        # the lock, so only the inference critical section is serialized.
        with _SYNTH_LOCK:
            return prov.synthesize(text, voice, speed, ssml)


_singleton: TTSService | None = None


def get_tts_service() -> TTSService:
    global _singleton
    if _singleton is None:
        _singleton = TTSService()
    return _singleton


def encode_wav(pcm: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> bytes:
    """PCM float32 → WAV bytes (16-bit PCM by default, ``FLOAT`` for 32-bit float).

    An over-unity array is **peak-normalized**, not clipped. ``np.clip`` alone bakes
    flat-top clipping — a hard, permanently audible distortion — into the samples
    before loudnorm's -1.5 dBTP ceiling ever gets a chance to help; the ceiling then
    turns the whole clip DOWN and the clipped tops ride along. Scaling by
    ``0.99 / peak`` is the same headroom decision made losslessly: level is a
    reversible change, a squared-off waveform is not. 0.99 (not 1.0) keeps a sliver
    below full scale so the subsequent quantization cannot round back up to clipping.

    ``np.clip`` stays as the last-resort net for anything the scale could not fix
    (a NaN/inf sample), and the peak is measured over the FINITE samples only — a
    stray inf must not become the peak and scale the real audio to silence.

    ``subtype="FLOAT"`` exists so the premaster stage can be handed un-quantized
    audio: the old path quantized to int16 here and only THEN applied loudnorm's
    gain, which lifts the 16-bit quantization floor along with the signal (grit in
    the pauses). See ``post_processor.premaster_float_wav``.
    """
    import soundfile as sf

    finite = np.isfinite(pcm)
    peak = float(np.abs(pcm[finite]).max()) if finite.any() else 0.0
    if peak > 1.0:
        pcm = pcm * (0.99 / peak)
    buf = io.BytesIO()
    pcm_clipped = np.clip(pcm, -1.0, 1.0)
    sf.write(buf, pcm_clipped, sample_rate, format="WAV", subtype=subtype)
    return buf.getvalue()
