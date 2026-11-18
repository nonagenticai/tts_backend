"""The splice de-click: `_crossfade_append`, the one helper both join loops use.

The owner reported audible "digital artifacts" in delivered narration. Butt-joining
two float arrays is a step discontinuity — the first array's last sample lands hard
against the second's first sample, at whatever values those happen to be — and that
step is a click. `tts_service` did it twice: once per KPipeline sub-chunk inside a
piece, and again at every mid-sentence piece boundary (`<emphasis>`, `<phoneme>`,
the word-packer), which carry no gap BY DESIGN and therefore splice at full speaking
level.

The fix is an OVERLAP, not an inserted ramp: inserting would push every later word
later, and this service's whole word-timing contract is that an offset matches the
audio to the sample. These tests therefore pin BOTH halves of that — the step is
gone AND the buffer grew by exactly what the helper said it did.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.tts_service import _CROSSFADE_SECONDS, _crossfade_append

SR = 24000
N = int(_CROSSFADE_SECONDS * SR)  # 72 samples at 24 kHz


def _flat(value: float, n: int = SR) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def _max_step(audio: np.ndarray) -> float:
    """Largest sample-to-sample jump anywhere in the buffer — the click metric."""
    return float(np.abs(np.diff(audio)).max())


class TestTheStepIsGone:
    def test_a_hard_splice_becomes_a_bounded_ramp(self):
        """+0.6 butted against -0.6 is a 1.2 step. After the fade, no jump survives.

        The bound is not arbitrary: a raised-cosine ramp spread over N samples can
        move the signal by at most `step * max(d/dx raised-cosine) = step * pi/2/(N+1)`
        per sample. Anything at or under that is the ramp; anything over it is a step
        that leaked through.
        """
        a, b = _flat(0.6), _flat(-0.6)

        assert _max_step(np.concatenate([a, b])) == pytest.approx(1.2), "the defect: a raw butt-join"

        parts: list[np.ndarray] = []
        _crossfade_append(parts, a, SR)
        _crossfade_append(parts, b, SR)
        joined = np.concatenate(parts)

        theoretical = 1.2 * (np.pi / 2) / (N + 1)
        assert _max_step(joined) <= theoretical * 1.01
        # ...and in absolute terms, ~35x smaller than the step it replaced.
        assert _max_step(joined) < 0.05

    def test_the_join_is_monotonic_not_a_wobble(self):
        """A ramp that overshoots either endpoint would be its own artifact."""
        parts: list[np.ndarray] = []
        _crossfade_append(parts, _flat(0.6), SR)
        _crossfade_append(parts, _flat(-0.6), SR)
        joined = np.concatenate(parts)

        assert joined.min() >= -0.6 - 1e-6
        assert joined.max() <= 0.6 + 1e-6
        seam = joined[SR - N - 1 : SR + 1]
        assert np.all(np.diff(seam) <= 1e-6), "the crossfade region must descend monotonically"

    def test_level_is_held_through_the_join(self):
        """Raised-cosine ramps sum to 1, so correlated material must not dip."""
        parts: list[np.ndarray] = []
        _crossfade_append(parts, _flat(0.5), SR)
        _crossfade_append(parts, _flat(0.5), SR)
        joined = np.concatenate(parts)

        assert np.allclose(joined, 0.5, atol=1e-6), "an equal-power dip would be audible as a hole"


class TestLengthAccounting:
    def test_the_return_value_is_exactly_the_growth(self):
        """Callers advance `offset_samples` by this. If it lies, every timing drifts."""
        parts: list[np.ndarray] = []
        before = 0
        for arr in (_flat(0.1, 1000), _flat(0.2, 800), _flat(0.3, 600)):
            before += _crossfade_append(parts, arr, SR)
        assert before == len(np.concatenate(parts))

    def test_the_overlap_is_deducted_once_per_join(self):
        parts: list[np.ndarray] = []
        _crossfade_append(parts, _flat(0.1, 1000), SR)
        added = _crossfade_append(parts, _flat(0.2, 800), SR)

        assert added == 800 - N
        assert len(np.concatenate(parts)) == 1000 + 800 - N

    def test_duration_is_essentially_unchanged(self):
        """3 ms per join: a shortening, but far below any pacing consequence."""
        parts: list[np.ndarray] = []
        _crossfade_append(parts, _flat(0.1), SR)
        _crossfade_append(parts, _flat(0.2), SR)
        shortfall = (2 * SR - len(np.concatenate(parts))) / SR
        assert shortfall == _CROSSFADE_SECONDS
        assert shortfall < 0.005


class TestEdges:
    def test_first_append_cannot_fade(self):
        parts: list[np.ndarray] = []
        assert _crossfade_append(parts, _flat(0.1, 500), SR) == 500
        assert len(parts) == 1

    def test_an_empty_array_appends_nothing(self):
        parts = [_flat(0.1, 500)]
        assert _crossfade_append(parts, np.zeros(0, dtype=np.float32), SR) == 0
        assert len(np.concatenate(parts)) == 500

    def test_an_array_shorter_than_the_fade_is_butt_joined(self):
        """All-or-nothing: a partial ramp would make the arithmetic unverifiable."""
        parts = [_flat(0.1, 500)]
        assert _crossfade_append(parts, _flat(0.2, N - 1), SR) == N - 1
        assert len(np.concatenate(parts)) == 500 + N - 1

    def test_a_running_buffer_shorter_than_the_fade_is_butt_joined(self):
        parts = [_flat(0.1, N - 1)]
        assert _crossfade_append(parts, _flat(0.2, 500), SR) == 500

    def test_fade_false_leaves_both_sides_byte_identical(self):
        """The authored-silence path. A ramp here would eat the pause."""
        speech, silence = _flat(0.6, 500), np.zeros(300, dtype=np.float32)
        parts: list[np.ndarray] = []
        _crossfade_append(parts, speech, SR)
        assert _crossfade_append(parts, silence, SR, fade=False) == 300

        joined = np.concatenate(parts)
        assert len(joined) == 800
        assert np.array_equal(joined[:500], speech)
        assert np.array_equal(joined[500:], silence), "authored silence must stay pure zeros"

    def test_a_nonsense_sample_rate_degrades_to_a_butt_join(self):
        parts = [_flat(0.1, 500)]
        assert _crossfade_append(parts, _flat(0.2, 500), 0) == 500

    def test_dtype_stays_float32(self):
        parts: list[np.ndarray] = []
        _crossfade_append(parts, _flat(0.1, 500), SR)
        _crossfade_append(parts, _flat(0.2, 500), SR)
        assert np.concatenate(parts).dtype == np.float32
