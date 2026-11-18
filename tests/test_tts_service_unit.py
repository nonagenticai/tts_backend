"""Unit tests for tts_service provider wiring + memory bounding."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import pytest

from services.ssml_processor import SsmlSegment
from services.tts_service import (
    _GAP_MAX_SECONDS,
    _GAP_MIN_SECONDS,
    _GAP_SECONDS,
    _PHONEMIZER_LINE_COUNT_ERROR,
    KokoroProvider,
    SynthesisResult,
    TTSService,
    _bounded_ort_session,
    _content_pacing_enabled,
    _gap_seconds_for,
    _normalize_for_phonemizer,
    _silence_phonemizer_words_mismatch,
    _trailing_punct,
)


class TestSilencePhonemizerWordsMismatch:
    """The real fix for the live /synthesize 500 flood.

    phonemizer-fork's words_mismatch processor raises
    'number of lines in input and output must be equal' whenever espeak emits a
    different line count than the input — which it does on ordinary narration
    text. The raise is purely diagnostic (phonemes are already computed), so the
    fix neutralizes it. This reproduces the exact production condition
    (input=4 lines, espeak output=3) and proves before/after behavior.
    """

    def test_neutralizes_the_fatal_line_count_raise(self):
        import logging

        from phonemizer.backend.espeak import words_mismatch
        from phonemizer.backend.espeak.words_mismatch import Ignore

        original = words_mismatch.BaseWordsMismatch._mismatched_lines
        try:

            def make():
                proc = Ignore(logging.getLogger("test"))
                proc._count_txt = [2, 3, 1, 2]  # input: 4 lines
                proc._count_phn = [2, 3, 1]  # espeak output: 3 lines
                return proc

            # BEFORE: the default processor raises on the line-count mismatch.
            with pytest.raises(RuntimeError, match="number of lines in input and output"):
                make().process(["a", "b", "c"])

            _silence_phonemizer_words_mismatch()

            # AFTER: same condition returns the phonemes unchanged, no raise.
            assert make().process(["a", "b", "c"]) == ["a", "b", "c"]
        finally:
            words_mismatch.BaseWordsMismatch._mismatched_lines = original


class TestBoundedOrtSession:
    def test_injects_arena_disabled_session_options(self, monkeypatch):
        """Regression: kokoro-onnx builds InferenceSession with no sess_options,
        so onnxruntime's CPU arena ratchets the process footprint up per long
        request and never releases it. _bounded_ort_session must force
        enable_cpu_mem_arena=False."""
        captured = {}

        def _fake_session(*args, **kwargs):
            captured["sess_options"] = kwargs.get("sess_options")
            return object()

        monkeypatch.setattr(ort, "InferenceSession", _fake_session)

        with _bounded_ort_session():
            # Simulate kokoro-onnx's exact call shape: positional model_path +
            # providers kwarg, no sess_options.
            ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

        so = captured["sess_options"]
        assert so is not None
        assert so.enable_cpu_mem_arena is False

    def test_restores_original_after_context(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(ort, "InferenceSession", sentinel)
        with _bounded_ort_session():
            assert ort.InferenceSession is not sentinel  # patched inside
        assert ort.InferenceSession is sentinel  # restored after

    def test_respects_explicit_sess_options(self, monkeypatch):
        """If a caller already passes sess_options, don't override it."""
        captured = {}
        mine = ort.SessionOptions()

        def _fake_session(*args, **kwargs):
            captured["sess_options"] = kwargs.get("sess_options")
            return object()

        monkeypatch.setattr(ort, "InferenceSession", _fake_session)
        with _bounded_ort_session():
            ort.InferenceSession("m.onnx", providers=["CPUExecutionProvider"], sess_options=mine)
        assert captured["sess_options"] is mine


class _FakeKokoro:
    """Stand-in for kokoro-onnx's Kokoro object; records every text it received.

    synthesize() now sentence-splits and calls create() once per piece, so we keep
    the full list; ``received_text`` stays as the last piece for convenience.
    """

    def __init__(self) -> None:
        self.received_texts: list[str] = []

    @property
    def received_text(self) -> str | None:
        return self.received_texts[-1] if self.received_texts else None

    def create(self, text, voice, speed):  # noqa: ANN001, ANN201 — match kokoro-onnx shape
        self.received_texts.append(text)
        return np.zeros(10, dtype=np.float32), 24000


def _loaded_provider(kokoro) -> KokoroProvider:  # noqa: ANN001
    prov = KokoroProvider()
    prov._kokoro = kokoro
    prov.voices = []  # empty registry → skip voice validation, exercise the text path
    return prov


class TestNormalizeForPhonemizer:
    def test_collapses_blank_lines_and_multiline(self):
        assert _normalize_for_phonemizer("Hello\n\nWorld") == "Hello World"

    def test_strips_trailing_and_interior_whitespace(self):
        assert _normalize_for_phonemizer("  line one \n\n  line two  \n") == "line one line two"

    def test_whitespace_only_becomes_empty(self):
        assert _normalize_for_phonemizer("\n\n   \t \n") == ""

    def test_single_clean_line_unchanged(self):
        assert _normalize_for_phonemizer("Just one line.") == "Just one line."


class TestKokoroSynthesizeNormalization:
    def test_splits_into_pieces_without_newlines(self):
        """Multi-line input is normalized and sentence-split; each piece reaching
        kokoro-onnx is a single line (no newlines) so espeak's per-line phoneme
        alignment can't desync, and no piece carries the truncation-prone tail."""
        fake = _FakeKokoro()
        prov = _loaded_provider(fake)
        prov.synthesize("First line.\n\nSecond line.\n", voice="af_heart", speed=1.0)
        assert fake.received_texts == ["First line.", "Second line."]
        assert all("\n" not in piece for piece in fake.received_texts)

    def test_normalizes_numbers_and_acronyms_before_create(self):
        """The pre-processor runs before kokoro: numbers become words and
        allow-listed initialisms are hyphenated."""
        fake = _FakeKokoro()
        prov = _loaded_provider(fake)
        prov.synthesize("The CAB met at 08:40.", voice="af_heart", speed=1.0)
        assert fake.received_texts == ["The C-A-B met at eight forty."]

    def test_inserts_gap_between_pieces_only(self):
        """A silence gap is inserted BETWEEN pieces but not after the last one, so
        total audio = sum(pieces) + (n-1) gaps."""
        fake = _FakeKokoro()
        prov = _loaded_provider(fake)
        result = prov.synthesize("One. Two. Three.", voice="af_heart", speed=1.0)
        gap = int(0.17 * 24000)
        assert len(fake.received_texts) == 3
        assert len(result.audio_pcm) == 3 * 10 + 2 * gap

    def test_rejects_whitespace_only_text_as_bad_request(self):
        prov = _loaded_provider(_FakeKokoro())
        with pytest.raises(ValueError, match="empty after whitespace normalization"):
            prov.synthesize("   \n\n  ", voice="af_heart", speed=1.0)

    def test_wraps_persistent_line_count_error_with_context(self):
        class _BadKokoro:
            def create(self, text, voice, speed):  # noqa: ANN001, ANN202
                raise RuntimeError(_PHONEMIZER_LINE_COUNT_ERROR)

        prov = _loaded_provider(_BadKokoro())
        with pytest.raises(RuntimeError, match="persisted after normalization"):
            prov.synthesize("hello", voice="af_heart", speed=1.0)

    def test_unrelated_runtime_error_is_not_swallowed(self):
        class _BoomKokoro:
            def create(self, text, voice, speed):  # noqa: ANN001, ANN202
                raise RuntimeError("model exploded")

        prov = _loaded_provider(_BoomKokoro())
        with pytest.raises(RuntimeError, match="model exploded"):
            prov.synthesize("hello", voice="af_heart", speed=1.0)


class _ConcurrencyProbeProvider:
    """Provider that records the peak number of threads inside synthesize at once.

    The real hazard is that the FastAPI route runs each request via
    asyncio.to_thread, so concurrent requests call the shared kokoro model from
    multiple OS threads simultaneously — espeak/kokoro-onnx are not thread-safe
    and garble the output. This stand-in makes that overlap observable: it sleeps
    briefly while holding a counter, so without the _SYNTH_LOCK serialization the
    peak overlap would be > 1.
    """

    name = "probe"
    default_voice = "af_heart"
    voices: list[str] = []

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak_in_flight = 0

    def is_loaded(self) -> bool:
        return True

    def synthesize(self, text: str, voice: str | None, speed: float, ssml: bool | None = None) -> SynthesisResult:
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            time.sleep(0.02)  # widen the window so any real overlap is caught
        finally:
            with self._lock:
                self._in_flight -= 1
        return SynthesisResult(
            audio_pcm=np.zeros(10, dtype=np.float32),
            sample_rate=24000,
            duration_sec=0.0,
            provider=self.name,
            voice=voice or self.default_voice,
        )


class TestSynthesisIsSerialized:
    """Regression for the promo "repeated half-sentences" defect (run 019e8d57).

    Concurrent /synthesize requests raced on the shared kokoro instance + the
    non-thread-safe espeak phonemizer, garbling the opening lines of a narration
    burst. TTSService.synthesize must hold a process-wide lock so the model is
    never entered by two threads at once.
    """

    def test_concurrent_synthesize_never_overlaps(self):
        probe = _ConcurrencyProbeProvider()
        svc = TTSService()
        svc.providers = {probe.name: probe}

        def _call(_i: int) -> None:
            svc.synthesize("Two sentences. Racing in threads.", "probe", None, 1.0)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_call, range(16)))

        # With the lock, the model is entered strictly one thread at a time.
        assert probe.peak_in_flight == 1


class TestKittenDefaultOff:
    def test_kitten_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("KITTEN_ENABLED", raising=False)
        svc = TTSService()
        assert svc.kitten_enabled is False

    def test_kitten_opt_in(self, monkeypatch):
        monkeypatch.setenv("KITTEN_ENABLED", "true")
        svc = TTSService()
        assert svc.kitten_enabled is True


class TestGapSecondsFor:
    """Mapping table + bounds for the content-derived inter-piece pause (TTSB-2)."""

    def test_none_and_empty_fall_back_to_default(self):
        assert _gap_seconds_for(None) == _GAP_SECONDS
        assert _gap_seconds_for("") == _GAP_SECONDS
        assert _gap_seconds_for("   ") == _GAP_SECONDS

    def test_terminal_marks_hold_longer_than_default(self):
        assert _gap_seconds_for(".") == 0.22
        assert _gap_seconds_for("!") == 0.22
        assert _gap_seconds_for(".") > _GAP_SECONDS

    def test_question_gets_a_distinct_longer_hold(self):
        assert _gap_seconds_for("?") == 0.25
        # A question wants a beat more than a plain statement.
        assert _gap_seconds_for("?") > _gap_seconds_for(".")

    def test_clause_marks_get_a_short_breath(self):
        assert _gap_seconds_for(",") == 0.09
        assert _gap_seconds_for(";") == 0.09
        assert _gap_seconds_for(":") == 0.09
        # The headline invariant: a sentence hold is longer than a clause breath.
        assert _gap_seconds_for(".") > _gap_seconds_for(",")

    def test_trailing_off_marks_get_the_floor(self):
        for mark in ("...", "…", "—", "–", "--"):
            assert _gap_seconds_for(mark) == 0.28, mark

    def test_ellipsis_beats_the_bare_period(self):
        # '...' ends with '.', so the ellipsis branch must win the ordering.
        assert _gap_seconds_for("...") == 0.28
        assert _gap_seconds_for("...") != _gap_seconds_for(".")

    def test_unknown_punctuation_uses_default(self):
        assert _gap_seconds_for("*") == _GAP_SECONDS

    def test_result_is_always_within_bounds(self):
        for mark in (None, "", ".", "!", "?", ",", ";", ":", "...", "…", "—", "–", "--", "*", "abc"):
            g = _gap_seconds_for(mark)
            assert _GAP_MIN_SECONDS <= g <= _GAP_MAX_SECONDS, mark


class TestTrailingPunct:
    def test_extracts_trailing_run(self):
        assert _trailing_punct("Hello.") == "."
        assert _trailing_punct("Wait...") == "..."
        assert _trailing_punct("Really?!") == "?!"

    def test_ignores_trailing_whitespace_and_closing_quotes(self):
        assert _trailing_punct('he said."  ') == "."
        assert _trailing_punct("(aside)") is None  # brackets stripped, no punct left

    def test_none_when_no_trailing_punctuation(self):
        assert _trailing_punct("plain word") is None
        assert _trailing_punct("   ") is None


class TestContentPacingKillSwitch:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        assert _content_pacing_enabled() is False

    def test_opt_in_truthy_values(self, monkeypatch):
        for val in ("true", "True", "1", "yes", " TRUE "):
            monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", val)
            assert _content_pacing_enabled() is True, val

    def test_falsey_values_stay_off(self, monkeypatch):
        for val in ("false", "0", "no", ""):
            monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", val)
            assert _content_pacing_enabled() is False, val


class TestRenderUnitsCarriesGaps:
    """_render_units threads a per-unit content gap alongside each create-unit."""

    def test_plain_pieces_carry_punctuation_derived_gaps(self):
        prov = KokoroProvider()
        units = prov._render_units([SsmlSegment(payload="Yes. No?")], 1.0)
        # Two sentence pieces; gap is the per-unit content pause regardless of the
        # misaki/espeak G2P branch taken.
        assert len(units) == 2
        assert [u.gap for u in units] == [0.22, 0.25]

    def test_phoneme_unit_falls_back_to_default_gap(self):
        prov = KokoroProvider()
        units = prov._render_units([SsmlSegment(payload="həˈloʊ", is_phonemes=True)], 1.0)
        assert len(units) == 1
        assert units[0].is_phonemes is True
        assert units[0].gap == _GAP_SECONDS  # IPA carries no recoverable sentence punctuation


class TestContentPacingSynthesize:
    """End-to-end gap behavior through KokoroProvider.synthesize."""

    _SR = 24000

    def test_kill_switch_off_is_byte_identical_flat_gap(self, monkeypatch):
        """The rollback guarantee: OFF ⇒ every gap is the flat _GAP_SECONDS,
        exactly the pre-TTSB-2 output on this no-CI live path."""
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)  # default OFF
        fake = _FakeKokoro()
        prov = _loaded_provider(fake)
        result = prov.synthesize("Yes. No? Maybe.", voice="af_heart", speed=1.0)
        flat = int(_GAP_SECONDS * self._SR)
        assert len(fake.received_texts) == 3
        # 3 pieces × 10 samples + 2 identical flat gaps — no variation by punctuation.
        assert len(result.audio_pcm) == 3 * 10 + 2 * flat

    def test_kill_switch_on_varies_gap_by_punctuation(self, monkeypatch):
        monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", "true")
        fake = _FakeKokoro()
        prov = _loaded_provider(fake)
        result = prov.synthesize("Yes. No? Maybe.", voice="af_heart", speed=1.0)
        gap_period = int(0.22 * self._SR)  # after "Yes."
        gap_question = int(0.25 * self._SR)  # after "No?"
        assert len(fake.received_texts) == 3
        assert len(result.audio_pcm) == 3 * 10 + gap_period + gap_question
        # The '?' gap is strictly longer than the '.' gap — pacing actually varies.
        assert gap_question > gap_period

    def test_on_differs_from_off(self, monkeypatch):
        fake_off = _FakeKokoro()
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        off = _loaded_provider(fake_off).synthesize("Yes. No? Maybe.", voice="af_heart", speed=1.0)

        fake_on = _FakeKokoro()
        monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", "true")
        on = _loaded_provider(fake_on).synthesize("Yes. No? Maybe.", voice="af_heart", speed=1.0)

        assert len(on.audio_pcm) != len(off.audio_pcm)

    def test_single_piece_has_no_trailing_gap_when_on(self, monkeypatch):
        """A standalone clip is never padded, pacing on or off."""
        monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", "true")
        fake = _FakeKokoro()
        prov = _loaded_provider(fake)
        result = prov.synthesize("Hello.", voice="af_heart", speed=1.0)
        assert len(fake.received_texts) == 1
        assert len(result.audio_pcm) == 10


class TestEncodeWavHeadroom:
    """An over-unity array is peak-normalized, not flat-topped.

    `np.clip(pcm, -1.0, 1.0)` alone bakes hard clipping into the samples BEFORE
    loudnorm's -1.5 dBTP ceiling can act; the ceiling then just turns the squared
    tops down with everything else. Scaling by `0.99 / peak` is the same headroom
    decision taken losslessly.
    """

    @staticmethod
    def _read(wav_bytes: bytes):  # noqa: ANN205
        import io

        import soundfile as sf

        return sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)

    def test_an_over_unity_sine_is_scaled_not_flat_topped(self):
        from services.tts_service import encode_wav

        sr = 24000
        t = np.linspace(0.0, 0.25, int(sr * 0.25), endpoint=False)
        pcm = (2.0 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        data, got_sr = self._read(encode_wav(pcm, sr))
        assert got_sr == sr
        # Clipping would park a large fraction of samples AT full scale; a scale
        # touches full scale essentially never.
        at_ceiling = int(np.count_nonzero(np.abs(data) >= 0.999))
        assert at_ceiling <= 2, at_ceiling
        # And the waveform must still be a sine: the clipped version's samples
        # near the peak are constant, the scaled version's are not.
        assert 0.95 <= float(np.abs(data).max()) <= 0.999, float(np.abs(data).max())
        # Shape preserved: ratio to the input is one constant gain.
        ratios = data[np.abs(pcm) > 0.5] / pcm[np.abs(pcm) > 0.5]
        assert float(ratios.std()) < 1e-3, float(ratios.std())

    def test_an_in_range_array_is_untouched(self):
        from services.tts_service import encode_wav

        sr = 24000
        t = np.linspace(0.0, 0.25, int(sr * 0.25), endpoint=False)
        pcm = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        data, _ = self._read(encode_wav(pcm, sr))
        assert abs(float(np.abs(data).max()) - 0.5) < 1e-3

    def test_a_non_finite_sample_does_not_scale_the_real_audio_to_silence(self):
        """A stray inf must not become the measured peak.

        `np.abs(pcm).max()` over an array containing inf is inf, so `0.99 / peak`
        would be 0 and the whole clip would go silent. The peak is measured over
        the FINITE samples only; `np.clip` remains the net for the inf itself.
        """
        from services.tts_service import encode_wav

        pcm = np.array([0.5, -0.5, np.inf, 0.5, np.nan], dtype=np.float32)
        data, _ = self._read(encode_wav(pcm, 24000))
        assert abs(float(data[0]) - 0.5) < 1e-3, float(data[0])

    def test_float_subtype_round_trips_as_float(self):
        """The premaster needs UN-quantized input; PCM_16 is the default elsewhere."""
        import io

        import soundfile as sf

        from services.tts_service import encode_wav

        pcm = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        assert sf.info(io.BytesIO(encode_wav(pcm, 24000, subtype="FLOAT"))).subtype == "FLOAT"
        assert sf.info(io.BytesIO(encode_wav(pcm, 24000))).subtype == "PCM_16"
