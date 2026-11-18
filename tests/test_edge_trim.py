"""Trimming kokoro's per-utterance edge padding — IMPLEMENTED, MEASURED, ARMED.

THE DEFECT: kokoro-torch emits ~0.30 s lead + ~0.44 s trail of silence around every
utterance (measured live on mini2, af_heart, -60 dBFS). `_create_piece_torch` trimmed
none of it, so a real inter-sentence gap was trail + `_GAP_SECONDS` + lead — measured
0.77-0.96 s where the service only meant to insert 0.17 s. That is the owner's
"strange pauses" complaint, ~0.74 s of it per sentence.

IT WORKS. Measured live on mini2, three narration shapes, trim OFF -> ON: silence
drops from 25-35 % of the file to ~10 %, and inter-sentence gaps land on exactly
0.25 s (= 0.17 inserted + 2 x 0.04 residual), precisely as designed.

⛔ WHY 0.25 s IS NOT THE ANSWER EITHER, and why this file no longer ends here.
0.25 s is BELOW the ~0.4-0.7 s natural inter-sentence pause of read narration: the
trim alone trades padded narration for breathless narration. `_GAP_SECONDS = 0.17`
was sized against the module comment "Kokoro silence-trims each batch", which is TRUE
of kokoro-ONNX and FALSE of the kokoro-torch path that runs live. So the trim ships
with `_TRIM_GAP_COMPENSATION_SECONDS = 0.18` — added to every inter-piece gap while
and only while the trim is armed, landing the delivered gap at 0.43 s. The two are
one change with two constants; the pairing is enforced by the gate, not by a comment.
See `test_content_pacing_read_path.py::TestTheLivePathGapIsWideEnoughToBreatheIn`.

⛔ THE HISTORICAL OBJECTION, AND WHY IT NO LONGER STANDS. This file previously said
the trim must stay OFF because the shorter file makes the caller's speed correction
swing the other way — measured converged speeds OFF -> ON of 1.186 -> 0.946,
1.593 -> 1.308 and 0.894 -> 0.660, the last through the orchestrator's 0.80 floor.
The reasoning behind it was that af_heart is not a ~150 wpm voice (it delivers
~199-248 wpm over SPEECH) and that the padding was doing the job the ~150 wpm
assumption credited to the voice.

That is all still true about the VOICE. What was wrong was the assumption about the
CALLER: the orchestrator does not budget at a fixed rate. `speech_rate.resolve_effective_wpm`
is the single owner of the rate for both the word budget AND the timeline solver, and
it computes `wpm = words / (sum(clip.duration_seconds) / 60)` over freshly synthesized
sample lines — the TOTAL clip duration, padding included. The trim is therefore
directly visible to the measurement: the same lines come back shorter, the measured
rate rises toward the voice's real rate, and the word budget grows to fill the same
beat instead of the speed being dragged down to do it. The measured rate stays inside
`[MIN_PLAUSIBLE_WPM=60, MAX_PLAUSIBLE_WPM=260]` — the delivered rate is the speech
rate diluted by the inserted gaps, so it sits under the 199-248 speech figure, and
widening the gaps moves it further from the ceiling, not closer.

Nothing in THIS repo models a speech rate at all. The `speech_seconds`/`pause_seconds`
split this service reports is carried by the orchestrator as evidence and branched on
NOWHERE, so the wider gaps cannot move a budget either.

These tests pin the MECHANISM, the DEFAULT (now ON), and the ROLLBACK (one env var
reverts both halves).
"""

from __future__ import annotations

import numpy as np
import pytest

from services.tts_service import (
    _TRIM_RESIDUAL_SECONDS,
    _TRIM_THRESHOLD,
    _trim_edge_silence,
    _trim_model_padding_enabled,
)

from .test_word_timings import SR, FakeResult, FakeToken, _torch_provider

RESIDUAL = int(_TRIM_RESIDUAL_SECONDS * SR)


def _padded(lead: float, speech: float, trail: float, level: float = 0.2) -> np.ndarray:
    """A kokoro-shaped utterance: quiet head, loud body, quiet tail."""
    return np.concatenate(
        [
            np.zeros(int(lead * SR), dtype=np.float32),
            np.full(int(speech * SR), level, dtype=np.float32),
            np.zeros(int(trail * SR), dtype=np.float32),
        ]
    )


class TestTheDefaultIsOn:
    def test_armed_unless_explicitly_disabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TTS_TRIM_MODEL_PADDING", raising=False)
        assert _trim_model_padding_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "FALSE", " No "])
    def test_opt_out_values(self, monkeypatch: pytest.MonkeyPatch, value: str):
        monkeypatch.setenv("TTS_TRIM_MODEL_PADDING", value)
        assert _trim_model_padding_enabled() is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE", "", "flase", "off"])
    def test_anything_that_is_not_an_affirmative_negative_stays_armed(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ):
        """A typo in the mini2 env must not silently revert the live path to padded.

        Note ``"off"`` and ``""``: this is the DEFAULT-ON form (`not in (false,0,no)`)
        that `_SSML_ENABLED` already uses in this module, not the opt-in form. Reading
        it as opt-in and expecting ``"off"`` to disarm is the mistake this pins.
        """
        monkeypatch.setenv("TTS_TRIM_MODEL_PADDING", value)
        assert _trim_model_padding_enabled() is True

    def test_off_leaves_the_audio_byte_identical(self, monkeypatch: pytest.MonkeyPatch):
        """The rollback guarantee on a repo whose merge deploys within 5 minutes."""
        monkeypatch.setenv("TTS_TRIM_MODEL_PADDING", "false")
        chunk = FakeResult(0, [FakeToken("alpha", 0.5, 0.9)])
        chunk.audio = _padded(0.3, 1.0, 0.45)
        prov = _torch_provider(monkeypatch, [[chunk]])

        result = prov.synthesize("Alpha.", "af_heart", 1.0, ssml=False)

        assert np.array_equal(result.audio_pcm, chunk.audio)
        assert result.duration_sec == pytest.approx(len(chunk.audio) / SR)
        assert len(result.word_timings) == 1
        assert result.word_timings[0].start == pytest.approx(0.5), "untrimmed timings must not move"
        assert result.word_timings[0].end == pytest.approx(0.9)


class TestTheMechanism:
    def test_both_edges_go_leaving_the_residual(self):
        trimmed, lead = _trim_edge_silence(_padded(0.3, 1.0, 0.45), SR)

        assert lead == int(0.3 * SR) - RESIDUAL
        assert len(trimmed) == pytest.approx(int(1.0 * SR) + 2 * RESIDUAL, abs=2)

    def test_interior_silence_is_structurally_safe(self):
        """kokoro's ~0.05 s comma pause is prosody and must survive.

        Not a threshold question: slicing a head and a tail cannot reach between
        them. This asserts that property directly — a 0.30 s interior pause,
        far longer than any residual, comes through untouched.
        """
        body = np.concatenate(
            [
                np.full(int(0.5 * SR), 0.2, dtype=np.float32),
                np.zeros(int(0.3 * SR), dtype=np.float32),  # the comma
                np.full(int(0.5 * SR), 0.2, dtype=np.float32),
            ]
        )
        padded = np.concatenate([np.zeros(int(0.3 * SR), dtype=np.float32), body])

        trimmed, _ = _trim_edge_silence(padded, SR)

        interior = np.flatnonzero(np.abs(trimmed) <= _TRIM_THRESHOLD)
        assert len(interior) >= int(0.3 * SR) - 2, "the comma pause was eaten"

    def test_an_all_silent_array_is_never_destroyed(self):
        """An authored <break> or a degenerate chunk must come back whole."""
        silence = np.zeros(int(0.5 * SR), dtype=np.float32)

        trimmed, lead = _trim_edge_silence(silence, SR)

        assert lead == 0
        assert np.array_equal(trimmed, silence)

    def test_an_unpadded_array_is_returned_unchanged(self):
        loud = np.full(SR, 0.2, dtype=np.float32)

        trimmed, lead = _trim_edge_silence(loud, SR)

        assert lead == 0
        assert np.array_equal(trimmed, loud)

    def test_a_degenerate_sample_rate_does_nothing(self):
        a = _padded(0.3, 1.0, 0.45)
        trimmed, lead = _trim_edge_silence(a, 0)
        assert lead == 0
        assert np.array_equal(trimmed, a)


class TestTimingsFollowTheTrim:
    def test_a_removed_lead_comes_off_the_word_timings(self, monkeypatch: pytest.MonkeyPatch):
        """THE trap: `start_ts` is relative to the UNTRIMMED chunk.

        Forget the subtraction and every caption sits late by the padding —
        invisible on a one-chunk clip, compounding across a long line.
        """
        monkeypatch.setenv("TTS_TRIM_MODEL_PADDING", "true")
        chunk = FakeResult(0, [FakeToken("alpha", 0.5, 0.9)])
        chunk.audio = _padded(0.3, 1.0, 0.45)
        prov = _torch_provider(monkeypatch, [[chunk]])

        result = prov.synthesize("Alpha.", "af_heart", 1.0, ssml=False)

        removed = (int(0.3 * SR) - RESIDUAL) / SR
        assert result.word_timings[0].start == pytest.approx(0.5 - removed, abs=1e-4)
        assert result.word_timings[0].end == pytest.approx(0.9 - removed, abs=1e-4)

    def test_timings_stay_inside_the_shortened_clip(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TTS_TRIM_MODEL_PADDING", "true")
        chunks = []
        for _ in range(3):
            c = FakeResult(0, [FakeToken("word", 0.4, 0.8)])
            c.audio = _padded(0.3, 1.0, 0.45)
            chunks.append(c)
        prov = _torch_provider(monkeypatch, [chunks])

        result = prov.synthesize("Word word word.", "af_heart", 1.0, ssml=False)

        assert len(result.word_timings) == 3
        previous = 0.0
        for w in result.word_timings:
            assert w.start >= 0.0
            assert w.start >= previous - 1e-6, "timings went backwards"
            assert w.end <= result.duration_sec + 1e-6
            previous = w.start


class TestTheAccountingSurvivesTrimming:
    def test_speech_plus_pause_still_reconstructs_the_duration(self, monkeypatch: pytest.MonkeyPatch):
        """Trimming removes model audio, never inserted silence — so the tally holds."""
        monkeypatch.setenv("TTS_TRIM_MODEL_PADDING", "true")
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        program = []
        for _ in range(3):
            c = FakeResult(0, [FakeToken("word", 0.4, 0.8)])
            c.audio = _padded(0.3, 1.0, 0.45)
            program.append([c])
        prov = _torch_provider(monkeypatch, program)

        result = prov.synthesize("One. Two. Three.", "af_heart", 1.0, ssml=False)

        assert result.speech_seconds + result.pause_seconds == pytest.approx(result.duration_sec)
        # The trim only ever removes MODEL output, never inserted silence, so it
        # cannot decrease this. It DOES increase it: `_TRIM_GAP_COMPENSATION_SECONDS`
        # is added to every inserted gap while the trim is armed, so the tally must
        # come out at 0.17 + 0.18 per gap, not the 0.17 the pre-wave-14 measurement
        # (pause_seconds identical trim on vs off, 0.340 / 0.510) recorded.
        assert result.pause_seconds == pytest.approx(2 * (0.17 + 0.18), abs=1e-3)
