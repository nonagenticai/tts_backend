"""`TTS_CONTENT_PACING_ENABLED=true`: does the per-punctuation table actually reach the audio?

WHY THIS EXISTS — the table in ``tts_service._gap_seconds_for`` has shipped DARK.
``TTS_CONTENT_PACING_ENABLED`` defaults to ``"false"`` and is set nowhere in
gitops, so every gap on the live narrate path is still the flat ``_GAP_SECONDS``
(~0.17 s): a clause ending in ``,`` gets exactly the same hole as a full stop.
Someone is about to flip that env to ``true`` on mini2, on a repo whose merge IS
the deploy. These tests are the proof that the flip does what it says BEFORE it
happens.

The existing pacing coverage (``test_pause_accounting``) asserts the TOTAL
``pause_seconds`` against ``_gap_seconds_for(".")`` — i.e. it re-derives its own
expectation from the table under test, and it only ever exercises ``.``. It can
therefore not tell a short clause gap from a long terminal one, which is the
entire feature. So these tests:

* assert **literal seconds** (0.09 / 0.22 / 0.25), never ``_gap_seconds_for`` — a
  silent edit to the table is a red test, which is the point of a contract;
* read the gaps out of the **rendered waveform** as runs of exact zeros, not out
  of an accounting field, so they cover the whole read path
  (``_trailing_punct`` → ``_gap_seconds_for`` → ``_RenderUnit.gap`` → the
  ``content_pacing`` branch of the concat loop) rather than any one hop of it;
* pin BOTH marks in ONE synthesis, so "the table is wired up" cannot pass by
  applying one constant everywhere.

The fixture leans on a real property of ``split_for_tts``: a sentence over
``max_chars`` is sub-split at clause boundaries (``, ; :``) and the mark STAYS on
the piece, which is the only way a non-terminal gap ever occurs in production.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.text_preproc import normalize_text_for_tts, split_for_tts
from services.tts_service import _GAP_SECONDS, _TRIM_GAP_COMPENSATION_SECONDS, _TRIM_RESIDUAL_SECONDS

from .test_word_timings import SR, FakeResult, FakeToken, _torch_provider

# The literal contract under test. Deliberately hardcoded: these are the numbers
# a reviewer of the mini2 env flip is agreeing to, and re-deriving them from the
# module would make this file agree with any future value.
CLAUSE_GAP = 0.09  # , ; :
TERMINAL_GAP = 0.22  # . !
QUESTION_GAP = 0.25  # ?
FLAT_GAP = 0.17  # _GAP_SECONDS, i.e. pacing OFF

# ⚠️ THE FOUR NUMBERS ABOVE ARE THE TABLE, NOT THE DELIVERED GAP. They are what
# `_gap_seconds_for` returns, and they are unchanged by the edge-trim work — the
# whole table survived it untouched, which is the point of making the compensation
# additive rather than folding it into `_GAP_SECONDS`.
#
# What a listener actually gets also includes `_TRIM_GAP_COMPENSATION_SECONDS`,
# added to EVERY gap while the torch edge-trim is armed (the default) because the
# trim removed ~0.66 s of kokoro padding from between every pair of pieces. So each
# test below states which world it is in, deliberately and per test: the
# `*_TRIMMED` constants are the live path, the bare ones are the rollback path
# (`TTS_TRIM_MODEL_PADDING=false`, or the ONNX engine, which self-trims).
#
# Both worlds are pinned, and pinned as LITERALS in the same spirit as the four
# above: re-deriving `0.35` as `FLAT_GAP + _TRIM_GAP_COMPENSATION_SECONDS` would
# make this file agree with any future value of a constant whose value is exactly
# what a reviewer of the mini2 deploy is agreeing to.
CLAUSE_GAP_TRIMMED = 0.27  # 0.09 + 0.18
TERMINAL_GAP_TRIMMED = 0.40  # 0.22 + 0.18
QUESTION_GAP_TRIMMED = 0.43  # 0.25 + 0.18
FLAT_GAP_TRIMMED = 0.35  # 0.17 + 0.18


@pytest.fixture
def untrimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disarm the edge-trim, i.e. the pre-wave-14 gap arithmetic.

    Not a convenience: with the trim armed (the default) every gap below carries
    `_TRIM_GAP_COMPENSATION_SECONDS`, so a test that pins a bare table value is
    pinning the ROLLBACK contract and has to say so.
    """
    monkeypatch.setenv("TTS_TRIM_MODEL_PADDING", "false")


def _text(mark: str) -> str:
    """One >320-char sentence whose first sub-split piece ends in ``mark``, then a full stop.

    ``split_for_tts`` groups the first two clauses into a 225-char piece ending in
    ``mark``, leaves the third clause as its own piece ending in ``.``, and the
    short following sentence is the third piece. That yields exactly two
    inter-piece gaps: one after ``mark``, one after ``.``.
    """
    return (
        "The survey team walked the whole northern ridge line for eleven long days "
        "in weather that never once broke at all, "
        "and they came back down with a single battered notebook of readings "
        f"that nobody in the office could reconcile{mark} "
        "so the archivist filed the whole thing under unresolved and then went "
        "quietly home for the evening. "
        "Then it stopped."
    )


def _zero_runs(audio: np.ndarray, min_len: int = 100) -> list[int]:
    """Lengths of the runs of exact zeros in the buffer, in order.

    The fake engine emits a constant 0.1 for every speech sample, so a run of
    exact zeros can only be silence this service spliced in. ``min_len`` drops
    nothing in practice (the shortest table value is 0.09 s = 2160 samples at
    24 kHz) and exists so a future one-sample artefact at a join cannot be
    mistaken for a gap.
    """
    nz = np.flatnonzero(audio != 0.0)
    if nz.size == 0:
        return [len(audio)] if len(audio) >= min_len else []
    runs: list[int] = []
    for a, b in zip(nz[:-1], nz[1:], strict=True):
        gap = int(b - a - 1)
        if gap >= min_len:
            runs.append(gap)
    return runs


def _synthesize(monkeypatch: pytest.MonkeyPatch, mark: str):
    text = _text(mark)
    pieces = split_for_tts(normalize_text_for_tts(text))
    assert len(pieces) == 3, f"fixture no longer splits into 3 pieces: {[len(p) for p in pieces]}"
    assert pieces[0].endswith(mark), f"fixture piece 0 must end in {mark!r}, got {pieces[0][-1]!r}"
    assert pieces[1].endswith("."), "fixture piece 1 must end in a full stop"
    prov = _torch_provider(
        monkeypatch,
        [[FakeResult(SR, [FakeToken("w", 0.0, 0.9)])] for _ in pieces],
    )
    return prov.synthesize(text, "af_heart", 1.0, ssml=False)


class TestTheTableReachesTheAudio:
    """The table with the edge-trim DISARMED — the numbers the table itself holds."""

    @pytest.mark.parametrize("mark", [",", ";", ":"])
    def test_a_clause_mark_gets_the_SHORT_gap_and_a_full_stop_the_LONG_one(
        self, monkeypatch: pytest.MonkeyPatch, untrimmed: None, mark: str
    ):
        """The whole feature in one assertion: two marks, one synthesis, two different holes."""
        monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", "true")

        result = _synthesize(monkeypatch, mark)

        runs = _zero_runs(result.audio_pcm)
        assert runs == [int(CLAUSE_GAP * SR), int(TERMINAL_GAP * SR)], (
            f"expected a {CLAUSE_GAP}s hole after {mark!r} then a {TERMINAL_GAP}s hole after '.', "
            f"got {[r / SR for r in runs]}"
        )
        assert runs[0] < runs[1], "a clause breath must be SHORTER than a terminal hold"

    def test_a_question_mark_gets_its_own_longer_hold(self, monkeypatch: pytest.MonkeyPatch, untrimmed: None):
        """'?' is the one mark that is longer than a full stop — it must not collapse into it."""
        monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", "true")
        text = "Did the survey team ever finish? They did not. Then it stopped."
        prov = _torch_provider(
            monkeypatch,
            [[FakeResult(SR, [FakeToken("w", 0.0, 0.9)])] for _ in range(3)],
        )

        result = prov.synthesize(text, "af_heart", 1.0, ssml=False)

        assert _zero_runs(result.audio_pcm) == [int(QUESTION_GAP * SR), int(TERMINAL_GAP * SR)]


class TestTheKillSwitchStillGates:
    @pytest.mark.parametrize("mark", [",", ";", ":"])
    def test_with_the_flag_OFF_every_gap_is_the_flat_one(
        self, monkeypatch: pytest.MonkeyPatch, untrimmed: None, mark: str
    ):
        """Today's production behaviour, pinned: the clause and the full stop are indistinguishable.

        This is the half that proves the flip is not a no-op — without it, a test
        that only ever ran with the flag on could pass against an implementation
        that ignores the flag entirely.
        """
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)

        result = _synthesize(monkeypatch, mark)

        flat = int(FLAT_GAP * SR)
        assert _GAP_SECONDS == FLAT_GAP, "the flat default moved; this file's contract numbers need re-reading"
        assert _zero_runs(result.audio_pcm) == [flat, flat]

    @pytest.mark.parametrize("value", ["false", "0", "no", "", "TRUE-ish", "yes-please"])
    def test_only_an_affirmative_value_arms_it(self, monkeypatch: pytest.MonkeyPatch, untrimmed: None, value: str):
        """A typo in the mini2 env must leave production exactly where it is."""
        monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", value)

        result = _synthesize(monkeypatch, ",")

        flat = int(FLAT_GAP * SR)
        assert _zero_runs(result.audio_pcm) == [flat, flat]

    @pytest.mark.parametrize("value", ["true", "True", " TRUE ", "1", "yes"])
    def test_the_affirmative_spellings_all_arm_it(self, monkeypatch: pytest.MonkeyPatch, untrimmed: None, value: str):
        monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", value)

        result = _synthesize(monkeypatch, ",")

        assert _zero_runs(result.audio_pcm) == [int(CLAUSE_GAP * SR), int(TERMINAL_GAP * SR)]


class TestTheLivePathGapIsWideEnoughToBreatheIn:
    """The edge-trim's other half, pinned in delivered seconds.

    ⛔ THIS IS THE TEST THAT STOPS THE TRIM SHIPPING ALONE. `TTS_TRIM_MODEL_PADDING`
    removes ~0.30 s of lead and ~0.44 s of trail from every piece. That is the right
    thing to do — it is the "strange pauses" complaint — but `_GAP_SECONDS = 0.17`
    was sized in a world where that padding was still there and supplied most of the
    perceived pause. Trim it and do nothing else and the inter-sentence hole collapses
    to 0.17 + 2 x 0.04 residual = 0.25 s, against a natural read-narration pause of
    ~0.4-0.7 s: the same defect, inverted, shipped as its own fix.

    So these assert the DELIVERED figure, not the inserted one, and assert it against
    the natural band rather than against a constant — a floor derived from
    `_GAP_SECONDS` would pass for any value of `_GAP_SECONDS`, including 0.17.
    """

    def test_the_flat_gap_a_listener_gets_lands_inside_the_natural_band(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        monkeypatch.delenv("TTS_TRIM_MODEL_PADDING", raising=False)  # the default: ARMED

        result = _synthesize(monkeypatch, ",")

        runs = _zero_runs(result.audio_pcm)
        assert runs == [int(FLAT_GAP_TRIMMED * SR), int(FLAT_GAP_TRIMMED * SR)]
        # What the listener hears: the inserted gap plus the residual the trim leaves
        # at each of the two facing edges (_TRIM_RESIDUAL_SECONDS, 0.04, x2).
        delivered = FLAT_GAP_TRIMMED + 2 * 0.04
        assert 0.40 <= delivered <= 0.70, (
            f"delivered inter-sentence gap {delivered:.3f}s is outside the natural "
            "read-narration band; trimming the padding without widening the gap ships "
            "breathless narration"
        )

    def test_the_trim_alone_would_be_too_short(self):
        """The refutation itself, as an assertion: 0.17 + residuals does NOT breathe.

        Without this, a future reader who deletes `_TRIM_GAP_COMPENSATION_SECONDS`
        sees only tests re-derived from constants, all of which follow the deletion.
        """
        # Against the MODULE constants, not this file's literals: an assertion over
        # two literals is a tautology that no code change can ever turn red.
        residuals = 2 * _TRIM_RESIDUAL_SECONDS
        assert _GAP_SECONDS + residuals < 0.40, (
            "`_GAP_SECONDS` alone now reaches the natural band, so the compensation's "
            "premise changed — re-read the pair rather than adjusting this number"
        )
        assert 0.40 <= _GAP_SECONDS + _TRIM_GAP_COMPENSATION_SECONDS + residuals <= 0.70

    def test_the_whole_table_shifts_by_the_same_amount_so_its_ordering_survives(self, monkeypatch: pytest.MonkeyPatch):
        """Compensating per-gap must not flatten the clause/terminal distinction."""
        monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", "true")
        monkeypatch.delenv("TTS_TRIM_MODEL_PADDING", raising=False)  # ARMED

        result = _synthesize(monkeypatch, ",")

        runs = _zero_runs(result.audio_pcm)
        assert runs == [int(CLAUSE_GAP_TRIMMED * SR), int(TERMINAL_GAP_TRIMMED * SR)]
        assert runs[0] < runs[1], "a clause breath must still be SHORTER than a terminal hold"

    def test_disarming_the_trim_takes_the_widened_gap_with_it(self, monkeypatch: pytest.MonkeyPatch):
        """The pair, proven in the disarm direction: one env var reverts BOTH halves.

        This is the rollback guarantee on a repo whose merge IS the deploy — a
        compensation left behind by a disarmed trim would be 0.35 s of gap on top of
        0.74 s of padding, i.e. worse than what shipped before either existed.
        """
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        monkeypatch.setenv("TTS_TRIM_MODEL_PADDING", "false")

        result = _synthesize(monkeypatch, ",")

        assert _zero_runs(result.audio_pcm) == [int(FLAT_GAP * SR), int(FLAT_GAP * SR)]
