"""Tests for SSML lowering (ssml_processor) + its integration into synthesize."""

from __future__ import annotations

import numpy as np
import pytest

from services.ssml_processor import (
    SsmlSegment,
    effective_speed,
    is_ssml,
    ssml_to_segments,
    strip_ssml,
    validate_ssml,
)
from services.tts_service import KokoroProvider


# --- pure lowering -----------------------------------------------------------
def test_is_ssml_detects_tags() -> None:
    assert is_ssml("Hello <break time='500ms'/> world")
    assert is_ssml('<say-as interpret-as="characters">CAB</say-as>')
    assert not is_ssml("Just plain text, no tags.")


def test_plain_text_is_single_segment() -> None:
    segs = ssml_to_segments("Hello world.")
    assert segs == [SsmlSegment(payload="Hello world.")]


def test_say_as_characters_hyphenates() -> None:
    segs = ssml_to_segments('The <say-as interpret-as="characters">CAB</say-as> arrived.')
    joined = "".join(s.payload for s in segs)
    assert "C-A-B" in joined
    assert all(not s.is_phonemes for s in segs)


def test_phoneme_becomes_phoneme_segment() -> None:
    segs = ssml_to_segments('Say <phoneme alphabet="ipa" ph="kˈɑstəɹ">Koster</phoneme> now.')
    phon = [s for s in segs if s.is_phonemes]
    assert len(phon) == 1
    assert phon[0].payload == "kˈɑstəɹ"
    # Surrounding text stays plain and is not merged into the phoneme span.
    assert any(not s.is_phonemes and "Say" in s.payload for s in segs)


def test_break_becomes_timed_silence_segment() -> None:
    # <break time="750ms"/> → a pure-silence segment carrying the exact ms (no fake
    # punctuation), text unchanged around it.
    segs = ssml_to_segments('Wait <break time="750ms"/> then go.')
    breaks = [s for s in segs if s.break_ms is not None]
    assert len(breaks) == 1
    assert breaks[0].break_ms == 750
    assert breaks[0].payload == ""
    # Seconds unit normalizes to ms.
    segs_s = ssml_to_segments('Wait <break time="1s"/> then go.')
    assert [s.break_ms for s in segs_s if s.break_ms is not None] == [1000]


def test_emphasis_strong_reduces_speed_not_uppercase() -> None:
    # kokoro can't do acoustic emphasis → emphasis = a per-segment speed reduction.
    # Text must be UNCHANGED (not uppercased); the span's factor ≈ 0.88.
    segs = ssml_to_segments("This is <emphasis level='strong'>vital</emphasis>.")
    emph = [s for s in segs if "vital" in s.payload]
    assert len(emph) == 1
    # A FACTOR, not an absolute speed: `speed` stays None so the request speed
    # still applies, and 0.88 scales it (see test_emphasis_scales_request_speed).
    assert emph[0].speed is None
    assert emph[0].speed_factor == pytest.approx(0.88)
    assert effective_speed(emph[0], 1.0) == pytest.approx(0.88)
    joined = "".join(s.payload for s in segs)
    assert "vital" in joined  # unchanged
    assert "VITAL" not in joined  # NOT the old .upper() fake


def test_escaped_ampersand_survives_parse() -> None:
    # The orchestrator escapes text before wrapping it in SSML; a properly-escaped
    # R&amp;D (and &lt;) must parse without error and round-trip to R&D.
    segs = ssml_to_segments("<speak>R&amp;D and 1 &lt; 2</speak>")
    joined = "".join(s.payload for s in segs)
    assert "R&D" in joined
    assert "1 < 2" in joined


def test_prosody_rate_sets_speed() -> None:
    segs = ssml_to_segments('<prosody rate="slow">slowly</prosody> and normal.')
    slow = [s for s in segs if s.speed == 0.75]
    assert slow and "slowly" in slow[0].payload
    # The trailing text outside prosody inherits the request default (None here).
    assert any(s.speed is None for s in segs)


def test_prosody_percent_rate_clamped() -> None:
    segs = ssml_to_segments('<prosody rate="200%">fast</prosody>')
    assert any(s.speed == 2.0 for s in segs)  # 200% clamps to kokoro max


def test_adjacent_text_coalesces_but_phoneme_splits() -> None:
    # Same-speed adjacent text stays one segment (no mid-sentence gap); a phoneme
    # span forces a boundary. (Emphasis now changes speed, so it deliberately does
    # NOT coalesce — covered by test_emphasis_forces_speed_boundary below.)
    segs = ssml_to_segments("Hello world! <phoneme ph='X'>Q</phoneme>")
    text_segs = [s for s in segs if not s.is_phonemes]
    phon_segs = [s for s in segs if s.is_phonemes]
    assert len(phon_segs) == 1
    assert len(text_segs) == 1
    assert "Hello world!" in text_segs[0].payload


def test_emphasis_forces_speed_boundary() -> None:
    # An emphasized span carries a reduced speed, so it must NOT merge with the
    # surrounding request-speed text — the reduction only applies to that word.
    segs = ssml_to_segments("Hello <emphasis>world</emphasis>!")
    text_segs = [s for s in segs if not s.is_phonemes]
    assert len(text_segs) == 3  # "Hello ", "world" (×0.94), "!"
    emph = [s for s in text_segs if s.payload == "world"]
    assert emph and emph[0].speed_factor == pytest.approx(0.94)  # default level → moderate


def test_emphasis_scales_request_speed_rather_than_replacing_it() -> None:
    # The defect this pins: the emphasis level used to be written into `speed` as an
    # ABSOLUTE value, so an emphasized word in a speed=1.5 request rendered at 0.88 —
    # the request speed was discarded for exactly the span the author cared about.
    segs = ssml_to_segments("Say <emphasis level='strong'>this</emphasis> now.")
    emph = [s for s in segs if s.payload == "this"][0]
    plain = [s for s in segs if s.payload.startswith("Say")][0]
    assert effective_speed(emph, 1.5) == pytest.approx(1.5 * 0.88)
    assert effective_speed(plain, 1.5) == pytest.approx(1.5)


def test_emphasis_inside_prosody_compounds_and_only_the_product_is_clamped() -> None:
    # <prosody> sets the absolute base; <emphasis> scales it. 0.5 * 0.88 = 0.44,
    # which is both below kokoro's own 0.5 floor AND far outside the narrow
    # ±15% band around the x-slow base (0.5) — the absolute intelligibility
    # floor (0.85) wins over both, since it sits above the band's own ceiling
    # (0.575) for a base this slow. Emphasis must never make an already-slow
    # span render slower still (root cause B, A/V quality wave 8).
    segs = ssml_to_segments("<prosody rate='x-slow'><emphasis level='strong'>careful</emphasis></prosody>")
    emph = [s for s in segs if "careful" in s.payload][0]
    assert emph.speed == pytest.approx(0.5)  # x-slow
    assert emph.speed_factor == pytest.approx(0.88)
    assert effective_speed(emph, 1.0) == pytest.approx(0.85)  # 0.44 clamped up to the absolute floor


def test_emphasis_cannot_drop_a_floored_beat_below_the_absolute_floor() -> None:
    # Root cause B (A/V quality wave 8): on a beat already stretched to the
    # orchestrator's vo_rate floor (0.80), "strong" emphasis used to render at
    # 0.80 * 0.88 = 0.704 — a further 12% instantaneous rate drop mid-sentence.
    # The absolute floor now wins: emphasis on an already-slow beat renders no
    # slower than 0.85, i.e. FASTER than the unclamped product would have been.
    segs = ssml_to_segments("Say <emphasis level='strong'>vital</emphasis> now.")
    emph = [s for s in segs if s.payload == "vital"][0]
    assert effective_speed(emph, 0.80) == pytest.approx(0.85)
    assert effective_speed(emph, 0.80) > 0.80 * 0.88  # the old, unclamped defect value


def test_emphasis_band_leaves_normal_speed_requests_unaffected() -> None:
    # At a normal (non-floored) request speed the band is wide enough that
    # ordinary "strong"/"moderate" emphasis is untouched — this clamp targets
    # only the floored-beat case, not emphasis in general.
    segs = ssml_to_segments("Say <emphasis level='strong'>this</emphasis> now.")
    emph = [s for s in segs if s.payload == "this"][0]
    assert effective_speed(emph, 1.0) == pytest.approx(0.88)
    assert effective_speed(emph, 1.5) == pytest.approx(1.5 * 0.88)


def test_malformed_ssml_falls_back_to_strip() -> None:
    segs = ssml_to_segments("Broken <break time='500ms'> unclosed <foo")
    assert len(segs) == 1
    assert not segs[0].is_phonemes


def test_strip_ssml_plain() -> None:
    assert strip_ssml("<speak>Hi <emphasis>there</emphasis></speak>") == "Hi there"


def test_validate_ssml_flags_bad_rate() -> None:
    assert validate_ssml('<prosody rate="turbo">x</prosody>')
    assert not validate_ssml('<prosody rate="fast">x</prosody>')


# --- integration into KokoroProvider.synthesize ------------------------------
class _CapturingKokoro:
    """Records (text, speed, is_phonemes) for every create() call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, bool]] = []

    def create(self, text, voice, speed, is_phonemes=False):  # noqa: ANN001, ANN201 — kokoro shape
        self.calls.append((text, speed, is_phonemes))
        return np.zeros(10, dtype=np.float32), 24000


def _provider(kokoro: _CapturingKokoro) -> KokoroProvider:
    prov = KokoroProvider()
    prov._kokoro = kokoro
    prov.voices = []
    return prov


def test_synthesize_plain_text_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # No SSML → single normalized/split path, is_phonemes stays False.
    fake = _CapturingKokoro()
    _provider(fake).synthesize("The CAB met at 08:40.", voice="af_heart", speed=1.0)
    assert [c[0] for c in fake.calls] == ["The C-A-B met at eight forty."]
    assert all(is_phon is False for _, _, is_phon in fake.calls)


def test_synthesize_phoneme_span_passthrough() -> None:
    fake = _CapturingKokoro()
    _provider(fake).synthesize(
        'Meet <phoneme alphabet="ipa" ph="kˈɑstəɹ">Koster</phoneme> today.',
        voice="af_heart",
        speed=1.0,
    )
    # Exactly one call carries the raw phonemes with is_phonemes=True.
    phon_calls = [c for c in fake.calls if c[2] is True]
    assert len(phon_calls) == 1
    assert phon_calls[0][0] == "kˈɑstəɹ"


def test_synthesize_say_as_reaches_create_hyphenated() -> None:
    fake = _CapturingKokoro()
    _provider(fake).synthesize(
        'The <say-as interpret-as="characters">CAB</say-as> arrived.',
        voice="af_heart",
        speed=1.0,
    )
    joined = " ".join(c[0] for c in fake.calls)
    assert "C-A-B" in joined


def test_synthesize_prosody_rate_applies_speed() -> None:
    fake = _CapturingKokoro()
    _provider(fake).synthesize(
        '<prosody rate="slow">Read this slowly.</prosody>',
        voice="af_heart",
        speed=1.0,
    )
    assert fake.calls and all(speed == 0.75 for _, speed, _ in fake.calls)


def test_synthesize_ssml_false_strips_tags_not_speaks_them(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit ssml=False → the markup is STRIPPED to plain text, NOT spoken. The
    # break silence is not inserted AND the tag itself is never voiced.
    fake = _CapturingKokoro()
    _provider(fake).synthesize('Hi <break time="1s"/> there.', voice="af_heart", speed=1.0, ssml=False)
    joined = " ".join(c[0] for c in fake.calls)
    assert "..." not in joined  # no break silence
    assert "break" not in joined.lower() and "1s" not in joined  # tag NOT spoken
    assert "Hi there." in joined


def test_synthesize_ssml_killswitch_strips_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    # TTS_SSML_ENABLED=false is the panic switch. A caller (the orchestrator) may
    # still emit <speak> when only the backend switch is flipped; the disabled path
    # must STRIP the markup, never phonemize the tags aloud.
    monkeypatch.setattr("services.tts_service._SSML_ENABLED", False)
    fake = _CapturingKokoro()
    _provider(fake).synthesize(
        '<speak><break time="500ms"/>Hello <emphasis level="strong">world</emphasis>.</speak>',
        voice="af_heart",
        speed=1.0,
    )
    joined = " ".join(c[0] for c in fake.calls)
    assert "speak" not in joined.lower() and "emphasis" not in joined.lower()
    assert "break" not in joined.lower()
    assert "Hello world." in joined


def test_synthesize_break_inserts_real_silence() -> None:
    # A 1000ms <break> appends >=24000 zero samples (24kHz) and is NOT sent to
    # kokoro (no create() call for the break itself).
    fake = _CapturingKokoro()
    result = _provider(fake).synthesize(
        'Wait <break time="1000ms"/> then go.',
        voice="af_heart",
        speed=1.0,
    )
    # Two spoken pieces reach create(); the break does not.
    assert len(fake.calls) == 2
    # 1000ms → exactly 24000 silence samples spliced in (plus the 2×10 stub samples).
    assert result.audio_pcm.shape[0] >= 24000


def test_synthesize_break_capped_at_max() -> None:
    # A runaway break is capped at _BREAK_MAX_SECONDS (1.2s → 28800 samples).
    # Lowered from 2.5s (A/V quality wave 8, plan item 2e): 2.5s of dead air mid-
    # narration reads as a dropped connection, not an authored pause.
    fake = _CapturingKokoro()
    result = _provider(fake).synthesize('A<break time="10s"/>B', voice="af_heart", speed=1.0)
    # 1.2s cap → 28800 silence samples, not the old 2.5s/60000 and well under a
    # raw 10s (240000).
    assert 28800 <= result.audio_pcm.shape[0] < 40000


def test_synthesize_emphasis_reaches_create_with_reduced_speed() -> None:
    fake = _CapturingKokoro()
    _provider(fake).synthesize(
        "Say <emphasis level='strong'>this word</emphasis> now.",
        voice="af_heart",
        speed=1.0,
    )
    speeds = [speed for _, speed, _ in fake.calls]
    assert any(speed == pytest.approx(0.88) for speed in speeds)  # emphasized span
    assert any(speed == 1.0 for speed in speeds)  # surrounding text at request speed


def test_synthesize_emphasis_honours_a_non_default_request_speed() -> None:
    # End-to-end form of test_emphasis_scales_request_speed: at speed=1.5 the
    # emphasized span must reach create() at 1.32, NOT at the bare level factor.
    fake = _CapturingKokoro()
    _provider(fake).synthesize(
        "Say <emphasis level='strong'>this word</emphasis> now.",
        voice="af_heart",
        speed=1.5,
    )
    speeds = [speed for _, speed, _ in fake.calls]
    assert any(speed == pytest.approx(1.5 * 0.88) for speed in speeds)  # emphasized span
    assert any(speed == pytest.approx(1.5) for speed in speeds)  # surrounding text
    assert not any(speed == pytest.approx(0.88) for speed in speeds)  # the old absolute bug


def test_synthesize_inserts_no_gap_at_a_mid_sentence_emphasis_boundary() -> None:
    # <emphasis> forces a unit boundary (different speed), and every boundary used to
    # get an inter-piece silence — ~340ms of dead air straddling every emphasized
    # word, because such a span is bounded on BOTH sides. A boundary inside a phrase
    # is not a place to breathe: only a piece that ended on punctuation earns a gap.
    fake = _CapturingKokoro()
    result = _provider(fake).synthesize(
        "Say <emphasis level='strong'>this word</emphasis> now.",
        voice="af_heart",
        speed=1.0,
    )
    assert len(fake.calls) == 3  # "Say" / "this word" / "now."
    # 3 stub pieces × 10 samples and NOT one sample more: no gap anywhere.
    assert result.audio_pcm.shape[0] == 30


def test_synthesize_still_gaps_between_real_sentences() -> None:
    # Control for the test above — the pacing gap must survive where it belongs.
    fake = _CapturingKokoro()
    result = _provider(fake).synthesize("First one. Second one.", voice="af_heart", speed=1.0)
    assert len(fake.calls) == 2
    assert result.audio_pcm.shape[0] > 20  # a real gap sits between the sentences


def test_synthesize_break_only_input_is_rejected_not_billed_as_silence() -> None:
    # A request whose only content is authored silence rendered a valid, perfectly
    # silent clip and returned it as a metered success. Silence is not synthesis.
    fake = _CapturingKokoro()
    with pytest.raises(ValueError, match="no speakable content"):
        _provider(fake).synthesize('<speak><break time="500ms"/></speak>', voice="af_heart", speed=1.0)
    assert fake.calls == []
