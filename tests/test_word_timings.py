"""Word-timing aggregation for the kokoro-torch engine (TTS_ENGINE=kokoro-torch).

The point of this feature is that the orchestrator can build word-timed captions
straight from /api/tts/synthesize instead of POSTing every narration line to
Whisper ASR. That only holds if the timings are correct in the FINAL audio
timebase — and synthesis is not one call. `_render_units` lowers a request into
several ordered units (sentence pieces, plus silence-only <break> units), each
piece is synthesized separately, and KPipeline itself may split ONE piece into
several chunks whose timestamps each restart near zero.

So every timestamp that reaches the caller has been re-based twice, and a bug in
either level is invisible at the start of a line and grows toward the end — a
caption that drifts late, which is exactly the defect this replaces. These tests
therefore pin EXACT sample-level offsets rather than asserting "roughly right",
and they run with no model: the engine is a scripted fake.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from models import SynthesizeResponse
from models import WordTiming as ResponseWordTiming
from routers.tts_routes import _word_timings_in_output_timebase
from services.tts_service import (
    _CROSSFADE_SECONDS,
    _ENGINE_ONNX,
    _ENGINE_TORCH,
    _GAP_SECONDS,
    _TORCH_SAMPLE_RATE,
    _TRIM_GAP_COMPENSATION_SECONDS,
    KokoroProvider,
    WordTiming,
    _configured_engine,
)
from services.tts_service import WordTiming as ServiceWordTiming

SR = _TORCH_SAMPLE_RATE
# What the concat loop inserts between two adjacent non-break pieces with the
# content-pacing kill-switch in its default OFF position, on a path where the
# torch edge-trim did NOT run: the ONNX engine (which silence-trims its own
# batches, so nothing was removed to compensate for) or torch with
# `TTS_TRIM_MODEL_PADDING=false`.
FLAT_GAP_SAMPLES = int(_GAP_SECONDS * SR)
# The same gap on the LIVE path — kokoro-torch with the edge-trim armed, which is
# the default. The trim takes ~0.66 s of kokoro padding out from between every
# pair of pieces, and `_TRIM_GAP_COMPENSATION_SECONDS` puts a deliberate part of
# it back; the two constants are a matched pair and neither moves alone. Derived
# from the module rather than hardcoded because the subject of the tests below is
# the OFFSET ARITHMETIC, not the value — the literal contract on the value itself
# lives in `test_content_pacing_read_path.py`.
TRIMMED_FLAT_GAP_SAMPLES = int((_GAP_SECONDS + _TRIM_GAP_COMPENSATION_SECONDS) * SR)
# The de-click overlap applied at every speech-to-speech join. It SHORTENS the
# buffer, so every offset downstream of a join moves earlier by exactly this much —
# which is why the expectations below are written against it rather than hardcoded.
CROSSFADE_SAMPLES = int(_CROSSFADE_SECONDS * SR)


class FakeToken:
    """Stand-in for a misaki MToken as KPipeline hands it back.

    `start_ts`/`end_ts` are None on a token with no phonemes — which is how real
    misaki reports punctuation (see KPipeline.join_timestamps, which skips them).
    """

    def __init__(self, text: str, start_ts: float | None, end_ts: float | None) -> None:
        self.text = text
        self.start_ts = start_ts
        self.end_ts = end_ts


class FakeResult:
    """One KPipeline.Result: a chunk of audio plus its CHUNK-RELATIVE tokens."""

    def __init__(self, n_samples: int, tokens: list[FakeToken] | None) -> None:
        self.audio = np.full(n_samples, 0.1, dtype=np.float32)
        self.tokens = tokens


class FakePipeline:
    """Scripted KPipeline: one entry in `program` per expected create call.

    Deliberately strict — an unexpected extra call fails loudly instead of
    silently yielding nothing, which would turn a units-lowering regression into
    a confusing timing assertion failure.
    """

    def __init__(self, program: list[list[FakeResult]]) -> None:
        self.program = program
        self.calls: list[tuple[str, str, float, bool]] = []

    def _next(self, payload: str, voice: str, speed: float, is_phonemes: bool) -> list[FakeResult]:
        idx = len(self.calls)
        self.calls.append((payload, voice, speed, is_phonemes))
        if idx >= len(self.program):
            raise AssertionError(f"unscripted synthesis call #{idx + 1} for payload {payload!r}")
        return self.program[idx]

    def __call__(self, text: str, voice: str, speed: float = 1.0) -> list[FakeResult]:
        return self._next(text, voice, speed, False)

    def generate_from_tokens(self, tokens: str, voice: str, speed: float = 1.0) -> list[FakeResult]:
        return self._next(tokens, voice, speed, True)


def _torch_provider(monkeypatch: pytest.MonkeyPatch, program: list[list[FakeResult]]) -> KokoroProvider:
    monkeypatch.setenv("TTS_ENGINE", _ENGINE_TORCH)
    prov = KokoroProvider()
    prov._pipeline = FakePipeline(program)
    prov.voices = []  # KPipeline has no registry; skips the membership check
    return prov


class TestEngineSelection:
    def test_defaults_to_onnx(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TTS_ENGINE", raising=False)
        assert _configured_engine() == _ENGINE_ONNX
        assert not KokoroProvider().emits_word_timings()

    def test_opt_in_to_torch(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TTS_ENGINE", "kokoro-torch")
        assert _configured_engine() == _ENGINE_TORCH
        assert KokoroProvider().emits_word_timings()

    def test_unknown_value_degrades_to_onnx(self, monkeypatch: pytest.MonkeyPatch):
        """A typo in the env must not take the live narrate path down."""
        monkeypatch.setenv("TTS_ENGINE", "kokoro-jax")
        assert _configured_engine() == _ENGINE_ONNX


class TestPieceRelativeAggregation:
    """Level 1: many KPipeline Results inside ONE piece."""

    def test_chunk_timestamps_are_rebased_by_real_sample_count(self, monkeypatch: pytest.MonkeyPatch):
        # KPipeline restarts join_timestamps at ~0 for every Result it yields, so
        # chunk B's 0.05 must land at 1.05 behind chunk A's one second of audio.
        prov = _torch_provider(
            monkeypatch,
            [
                [
                    FakeResult(SR, [FakeToken("Alpha", 0.0, 0.4), FakeToken("bravo", 0.45, 0.9)]),
                    FakeResult(SR // 2, [FakeToken("gamma", 0.05, 0.45)]),
                ]
            ],
        )
        audio, sr, words = prov._create_piece_torch("Alpha bravo gamma", "af_heart", 1.0, False)

        assert sr == SR
        # Chunk B is OVERLAPPED onto chunk A by the de-click crossfade, so the buffer
        # is one overlap shorter than the raw sum — and chunk B's timestamps are
        # re-based against where it REALLY starts, one overlap before 1.0s. Overlap,
        # not insertion, is what keeps that shift bounded at 3ms instead of growing.
        assert len(audio) == SR + SR // 2 - CROSSFADE_SAMPLES
        chunk_b = (SR - CROSSFADE_SAMPLES) / SR
        assert words == [
            WordTiming("Alpha", 0.0, 0.4),
            WordTiming("bravo", 0.45, 0.9),
            WordTiming("gamma", chunk_b + 0.05, chunk_b + 0.45),
        ]

    def test_offset_follows_audio_not_timestamps(self, monkeypatch: pytest.MonkeyPatch):
        """Predicted-duration error must stay inside its chunk, never accumulate.

        Chunk A claims a word ending at 2.0s but only emits 1.0s of audio. If the
        offset were taken from the timestamps, chunk B would be pushed to 2.0s and
        every later chunk would inherit the error. It is taken from real samples,
        so chunk B starts at exactly 1.0s.
        """
        prov = _torch_provider(
            monkeypatch,
            [
                [
                    FakeResult(SR, [FakeToken("overlong", 0.0, 2.0)]),
                    FakeResult(SR, [FakeToken("next", 0.0, 0.5)]),
                ]
            ],
        )
        _audio, _sr, words = prov._create_piece_torch("overlong next", "af_heart", 1.0, False)
        chunk_b = (SR - CROSSFADE_SAMPLES) / SR  # 1.0s of audio, less the de-click overlap
        assert words[1] == WordTiming("next", chunk_b, chunk_b + 0.5)

    def test_punctuation_only_tokens_are_dropped(self, monkeypatch: pytest.MonkeyPatch):
        """Rule 5: the consumer aligns against authored script WORDS."""
        prov = _torch_provider(
            monkeypatch,
            [
                [
                    FakeResult(
                        SR,
                        [
                            FakeToken("Hello", 0.0, 0.4),
                            FakeToken(",", None, None),  # misaki: no phonemes -> no ts
                            FakeToken("—", 0.4, 0.45),  # timestamped but not a word
                            FakeToken("   ", 0.45, 0.5),  # whitespace token
                            FakeToken("world", 0.5, 0.9),
                        ],
                    )
                ]
            ],
        )
        _audio, _sr, words = prov._create_piece_torch("Hello, world", "af_heart", 1.0, False)
        assert [w.word for w in words] == ["Hello", "world"]

    def test_phoneme_passthrough_reports_no_timings(self, monkeypatch: pytest.MonkeyPatch):
        """generate_from_tokens yields tokens=None — report nothing, don't guess."""
        prov = _torch_provider(monkeypatch, [[FakeResult(SR, None)]])
        audio, _sr, words = prov._create_piece_torch("hˈɛloʊ", "af_heart", 1.0, True)
        assert len(audio) == SR
        assert words == []
        assert prov._pipeline.calls[0][3] is True  # took the generate_from_tokens path

    def test_audio_none_subchunk_raises_instead_of_silently_dropping_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sub-chunk with no audio must fail loudly, not vanish.

        `if audio is None: continue` used to drop the whole sub-chunk AND its word
        timings with no log, no retry, no error — a 200 with a shorter file and
        text that was never spoken. This pins the fix: log + raise.
        """

        no_audio_result = FakeResult(SR, None)
        no_audio_result.audio = None  # ty: ignore[invalid-assignment] # simulate KPipeline's real failure mode
        prov = _torch_provider(
            monkeypatch,
            [[FakeResult(SR, [FakeToken("first", 0.0, 0.4)]), no_audio_result]],
        )
        with pytest.raises(RuntimeError, match="no audio"):
            prov._create_piece_torch("first second", "af_heart", 1.0, False)


class TestCrossPieceAggregation:
    """Level 2: THE test — offsets across pieces, an inserted gap, and silence.

    Input lowers to four ordered units:

        [0] text  "One two."      -> next unit is text  => flat gap inserted AFTER
        [1] text  "Three four."   -> next unit is break => gap SUPPRESSED
        [2] break 500ms           -> silence only, no create(), gap suppressed
        [3] text  "Five six."     -> last unit, no trailing gap

    which exercises every branch that can move the running offset: a real piece,
    an inserted gap, a suppressed gap, and authored silence.
    """

    TEXT = "<speak>One two. Three four. <break time='500ms'/>Five six.</speak>"

    def _provider(self, monkeypatch: pytest.MonkeyPatch) -> KokoroProvider:
        return _torch_provider(
            monkeypatch,
            [
                # unit 0 — two chunks: 1.0s + 0.5s
                [
                    FakeResult(SR, [FakeToken("Alpha", 0.0, 0.4), FakeToken("bravo", 0.45, 0.9)]),
                    FakeResult(SR // 2, [FakeToken("gamma", 0.05, 0.45)]),
                ],
                # unit 1 — one chunk, 1.0s
                [FakeResult(SR, [FakeToken("delta", 0.1, 0.5)])],
                # unit 3 — one chunk, 1.0s (unit 2 is the break; it calls nothing)
                [FakeResult(SR, [FakeToken("epsilon", 0.2, 0.9)])],
            ],
        )

    def test_offsets_are_exact_across_pieces_gaps_and_silence(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)  # flat-gap default
        prov = self._provider(monkeypatch)

        result = prov.synthesize(self.TEXT, "af_heart", 1.0, ssml=True)

        assert len(prov._pipeline.calls) == 3, "the <break> unit must not call the model"

        # Audio layout, in samples: piece0 | gap | piece1 | break | piece3
        piece0 = SR + SR // 2 - CROSSFADE_SAMPLES  # two chunks, joined by one crossfade
        break_samples = int(0.5 * SR)
        assert len(result.audio_pcm) == piece0 + TRIMMED_FLAT_GAP_SAMPLES + SR + break_samples + SR

        # Offsets in seconds of the start of each unit.
        off1 = (piece0 + TRIMMED_FLAT_GAP_SAMPLES) / SR
        off3 = (piece0 + TRIMMED_FLAT_GAP_SAMPLES + SR + break_samples) / SR
        # gamma is in unit 0's SECOND chunk, so it sits one de-click overlap earlier
        # than the raw 1.0s of chunk audio ahead of it.
        chunk_b = (SR - CROSSFADE_SAMPLES) / SR
        assert result.word_timings == [
            WordTiming("Alpha", 0.0, 0.4),
            WordTiming("bravo", 0.45, 0.9),
            WordTiming("gamma", chunk_b + 0.05, chunk_b + 0.45),
            WordTiming("delta", off1 + 0.1, off1 + 0.5),
            WordTiming("epsilon", off3 + 0.2, off3 + 0.9),
        ]

    def test_timings_are_monotonic_and_inside_the_clip(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        prov = self._provider(monkeypatch)

        result = prov.synthesize(self.TEXT, "af_heart", 1.0, ssml=True)

        previous_end = 0.0
        for w in result.word_timings:
            assert w.start >= previous_end, f"{w.word} starts before the previous word ended"
            assert w.end > w.start, f"{w.word} has a non-positive duration"
            previous_end = w.end
        assert result.word_timings[-1].end <= result.duration_sec

    def test_content_pacing_gap_also_advances_the_offset(self, monkeypatch: pytest.MonkeyPatch):
        """With pacing ON the gap after '.' is 0.22s, not 0.17s — and the words move.

        This is the branch the offset accounting is most likely to miss: the audio
        and the offset come from two different expressions unless they are derived
        from the same array.
        """
        monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", "true")
        prov = self._provider(monkeypatch)

        result = prov.synthesize(self.TEXT, "af_heart", 1.0, ssml=True)

        # 0.22 from the table + the trim compensation the live (torch, trim-armed)
        # path adds to EVERY gap. The table literal is unchanged by this feature.
        paced_gap = int((0.22 + _TRIM_GAP_COMPENSATION_SECONDS) * SR)  # _gap_seconds_for(".")
        assert paced_gap != TRIMMED_FLAT_GAP_SAMPLES
        piece0 = SR + SR // 2 - CROSSFADE_SAMPLES  # two chunks, joined by one crossfade
        expected_off1 = (piece0 + paced_gap) / SR
        delta = next(w for w in result.word_timings if w.word == "delta")
        assert delta.start == pytest.approx(expected_off1 + 0.1)
        # And the audio really is that long — offset and audio cannot disagree.
        assert len(result.audio_pcm) == piece0 + paced_gap + SR + int(0.5 * SR) + SR

    def test_end_is_clamped_to_the_clip(self, monkeypatch: pytest.MonkeyPatch):
        """A predicted end past the last sample must not escape into a caption."""
        prov = _torch_provider(monkeypatch, [[FakeResult(SR, [FakeToken("solo", 0.0, 99.0)])]])
        result = prov.synthesize("Solo.", "af_heart", 1.0, ssml=False)
        assert result.word_timings == [WordTiming("solo", 0.0, result.duration_sec)]


class TestSpliceDeClickKeepsTimingsHonest:
    """The de-click overlap must move timings WITH the audio, never against it.

    `_crossfade_append` shortens the buffer at every speech-to-speech join. That is
    fine — and only fine — if `offset_samples` is advanced by what the buffer
    actually grew by. Advancing by the raw length instead leaves every timing after
    the join pointing one overlap late, growing with each join toward the end of the
    line: exactly the caption drift the word-timing feature exists to remove.
    """

    def test_a_word_in_a_later_chunk_is_where_the_audio_really_puts_it(self, monkeypatch: pytest.MonkeyPatch):
        """Derived from the DELIVERED buffer, not from the arrays that went in.

        The last chunk is a full 1.0s and is the last thing in the clip, so whatever
        else the crossfade did, its word must start exactly 1.0s before the end.

        ⚠️ THREE chunks, not two, and that is load-bearing. Advancing the offset by
        the raw length instead of the growth is SELF-CANCELLING across a single join
        (the re-base subtracts the same overlap the offset over-counted) and only
        starts drifting from the SECOND join onward. A two-chunk version of this test
        passes with the bug in place.
        """
        prov = _torch_provider(
            monkeypatch,
            [
                [
                    FakeResult(SR, [FakeToken("first", 0.0, 0.5)]),
                    FakeResult(SR, [FakeToken("second", 0.0, 0.5)]),
                    FakeResult(SR, [FakeToken("third", 0.0, 0.5)]),
                ]
            ],
        )
        audio, _sr, words = prov._create_piece_torch("first second third", "af_heart", 1.0, False)

        assert len(audio) == 3 * SR - 2 * CROSSFADE_SAMPLES
        for i, word in enumerate(words):
            start = (i * (SR - CROSSFADE_SAMPLES)) / SR
            assert word.start == pytest.approx(start), f"{word.word} drifted"
            assert word.end == pytest.approx(start + 0.5)
        assert words[-1].start == pytest.approx(len(audio) / SR - 1.0)

    def test_a_mid_sentence_piece_join_is_crossfaded_and_still_exact(self, monkeypatch: pytest.MonkeyPatch):
        """`<emphasis>` splits a phrase into pieces with NO gap — the worst splice.

        Three pieces, two speech-to-speech joins, so two overlaps; and the last
        piece's word still sits exactly its own length before the end of the clip.
        """
        prov = _torch_provider(
            monkeypatch,
            [
                [FakeResult(SR, [FakeToken("Say", 0.0, 0.3)])],
                [FakeResult(SR, [FakeToken("this", 0.0, 0.3)])],
                [FakeResult(SR, [FakeToken("now", 0.0, 0.4)])],
            ],
        )
        result = prov.synthesize("Say <emphasis level='strong'>this</emphasis> now.", "af_heart", 1.0, ssml=True)

        assert len(prov._pipeline.calls) == 3
        assert len(result.audio_pcm) == 3 * SR - 2 * CROSSFADE_SAMPLES, "no gap, two crossfades"
        last = next(w for w in result.word_timings if w.word == "now")
        assert last.start == pytest.approx(result.duration_sec - 1.0)

    def test_authored_break_silence_is_not_ramped(self, monkeypatch: pytest.MonkeyPatch):
        """A `<break>` is a deliberate pause: no overlap on either side of it, and
        its zeros stay zeros. Fading into it would both shorten the authored pause
        and blur an ending that is meant to stop clean."""
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        prov = _torch_provider(
            monkeypatch,
            [
                [FakeResult(SR, [FakeToken("before", 0.0, 0.4)])],
                [FakeResult(SR, [FakeToken("after", 0.0, 0.4)])],
            ],
        )
        result = prov.synthesize("<speak>Before.<break time='500ms'/>After.</speak>", "af_heart", 1.0, ssml=True)

        break_samples = int(0.5 * SR)
        # Not one sample of overlap anywhere: piece | break | piece, full lengths.
        assert len(result.audio_pcm) == SR + break_samples + SR
        assert np.array_equal(result.audio_pcm[SR : SR + break_samples], np.zeros(break_samples, dtype=np.float32))
        # And the piece after the silence starts exactly where the silence ended.
        after = next(w for w in result.word_timings if w.word == "after")
        assert after.start == pytest.approx((SR + break_samples) / SR)

    def test_an_inserted_pacing_gap_is_not_ramped_either(self, monkeypatch: pytest.MonkeyPatch):
        """Same rule for the inter-sentence gap: it is silence the service chose to
        insert, so the surrounding pieces keep their full length."""
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        prov = _torch_provider(
            monkeypatch,
            [
                [FakeResult(SR, [FakeToken("one", 0.0, 0.4)])],
                [FakeResult(SR, [FakeToken("two", 0.0, 0.4)])],
            ],
        )
        result = prov.synthesize("One. Two.", "af_heart", 1.0, ssml=False)

        assert len(result.audio_pcm) == SR + TRIMMED_FLAT_GAP_SAMPLES + SR
        two = next(w for w in result.word_timings if w.word == "two")
        assert two.start == pytest.approx((SR + TRIMMED_FLAT_GAP_SAMPLES) / SR)

    def test_no_step_survives_a_mid_sentence_join(self, monkeypatch: pytest.MonkeyPatch):
        """End to end: the click itself, measured on the delivered buffer."""

        class Loud(FakeResult):
            def __init__(self, value: float, tokens):
                super().__init__(SR, tokens)
                self.audio = np.full(SR, value, dtype=np.float32)

        prov = _torch_provider(
            monkeypatch,
            [
                [Loud(0.7, [FakeToken("Say", 0.0, 0.3)])],
                [Loud(-0.7, [FakeToken("this", 0.0, 0.3)])],
            ],
        )
        result = prov.synthesize("Say <emphasis level='strong'>this</emphasis>", "af_heart", 1.0, ssml=True)

        # Butt-joined, this pair steps by 1.4 in one sample. Ramped, by ~0.03.
        assert float(np.abs(np.diff(result.audio_pcm)).max()) < 0.05


class TestOnnxPathUnaffected:
    def test_onnx_engine_emits_no_timings(self, monkeypatch: pytest.MonkeyPatch):
        """The default path must be untouched: audio as before, timings empty."""
        monkeypatch.delenv("TTS_ENGINE", raising=False)
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        prov = KokoroProvider()

        class FakeKokoro:
            def create(self, payload, voice, speed, is_phonemes=False):
                return np.zeros(SR, dtype=np.float32), SR

        prov._kokoro = FakeKokoro()
        prov.voices = ["af_heart"]

        result = prov.synthesize("One two. Three four.", "af_heart", 1.0, ssml=False)

        assert result.word_timings == []
        assert len(result.audio_pcm) == SR + FLAT_GAP_SAMPLES + SR


class TestOutputTimebase:
    """The route maps engine-timebase offsets onto the WAV the caller receives.

    premaster is length-exact (it re-encodes at the INPUT sample rate and
    `linear=true` loudnorm is pure gain — measured: 0 frame delta on 1s/4.4s/12s
    clips), so it needs no compensation. `encode_wav` with a `sample_rate`
    OVERRIDE does move things: soundfile does not resample, it relabels the same
    samples, so the clip's real duration scales by engine_sr/target_sr.
    """

    def test_absent_when_the_engine_produced_none(self):
        assert _word_timings_in_output_timebase([], 24000, 24000) is None

    def test_unchanged_when_the_rate_matches(self):
        given = [ServiceWordTiming("hello", 1.0, 1.5)]
        assert _word_timings_in_output_timebase(given, 24000, 24000) == [
            ResponseWordTiming(word="hello", start=1.0, end=1.5)
        ]

    def test_rescaled_when_the_caller_overrides_the_sample_rate(self):
        # Same samples relabelled at 48 kHz play twice as fast: 1.0s -> 0.5s.
        given = [ServiceWordTiming("hello", 1.0, 1.5)]
        assert _word_timings_in_output_timebase(given, 24000, 48000) == [
            ResponseWordTiming(word="hello", start=0.5, end=0.75)
        ]

    def test_json_response_omits_the_field_on_the_onnx_default(self):
        """Absent, not null — existing consumers see exactly today's payload."""
        payload = SynthesizeResponse(
            audio_base64="",
            sample_rate=24000,
            duration_sec=1.0,
            provider="kokoro",
            voice="af_heart",
            format="wav",
            word_timings=None,
        )
        assert "word_timings" not in json.loads(payload.model_dump_json(exclude_none=True))
