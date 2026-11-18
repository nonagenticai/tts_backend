"""`pause_seconds` / `speech_seconds`: the denominator a speed correction must use.

WHY THIS EXISTS — measured on three delivered narration WAVs (run
01a09e7d-c6da-7450-a7cf-91c43a4aecf3) and re-measured live on mini2 against
kokoro-torch: **40-43% of every returned clip is silence, at every speed.** The
caller (w2v_orchestrator) corrects narration speed with ``s* = (d * s) / A``,
computed from TOTAL duration. Dividing a words-per-minute target into a number
that is ~42% silence over-drives the speech.

Only part of that silence is this service's doing, and the split is what these
tests pin:

* **Inserted pacing silence** — the flat ``_GAP_SECONDS`` (or a content-derived
  gap) between pieces, plus any authored SSML ``<break>``. It is a fixed number
  of zero samples: it does NOT shrink when ``speed`` goes up, so it must come
  OUT of the denominator. This is what ``pause_seconds`` reports.
* **kokoro's own per-utterance head/tail padding** — measured live on mini2 at
  ~0.30 s lead + ~0.44 s trail per piece at ``speed=1.0``, falling to ~0.23 s /
  ~0.24 s at ``speed=1.81``. That padding IS (near-)proportionally scaled by
  ``speed``, so it behaves like speech for the purposes of the correction and is
  deliberately NOT counted here. ``test_model_emitted_silence_is_not_counted``
  is the executable statement of that decision — if a future change starts
  waveform-measuring instead of sample-counting, it goes red.

The accounting is summed from the exact arrays the concat loop appended, never
re-measured, so ``speech_seconds + pause_seconds == duration_sec`` to the sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.tts_service import _GAP_SECONDS, _TRIM_GAP_COMPENSATION_SECONDS, _gap_seconds_for

from .test_word_timings import (
    CROSSFADE_SAMPLES,
    SR,
    TRIMMED_FLAT_GAP_SAMPLES,
    FakeResult,
    FakeToken,
    _torch_provider,
)

# The fixture text lowers to: text | flat gap | text | break(500ms) | text.
# One inserted pacing gap, one authored break, one SUPPRESSED gap (before the
# break) — i.e. every branch that can add, or decline to add, pause samples.
TEXT = "<speak>One two. Three four. <break time='500ms'/>Five six.</speak>"
BREAK_SAMPLES = int(0.5 * SR)

PROGRAM = [
    [
        FakeResult(SR, [FakeToken("Alpha", 0.0, 0.4)]),
        FakeResult(SR // 2, [FakeToken("gamma", 0.05, 0.45)]),
    ],
    [FakeResult(SR, [FakeToken("delta", 0.1, 0.5)])],
    [FakeResult(SR, [FakeToken("epsilon", 0.2, 0.9)])],
]
# Speech audio actually emitted by the (fake) model: two chunks joined by one
# de-click overlap, plus two single-chunk pieces.
SPEECH_SAMPLES = (SR + SR // 2 - CROSSFADE_SAMPLES) + SR + SR


class TestTheSplitIsExact:
    def test_flat_gap_and_authored_break_are_both_counted(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        prov = _torch_provider(monkeypatch, PROGRAM)

        result = prov.synthesize(TEXT, "af_heart", 1.0, ssml=True)

        assert result.pause_seconds == pytest.approx((TRIMMED_FLAT_GAP_SAMPLES + BREAK_SAMPLES) / SR)
        assert result.speech_seconds == pytest.approx(SPEECH_SAMPLES / SR)

    def test_speech_plus_pause_reconstructs_the_duration(self, monkeypatch: pytest.MonkeyPatch):
        """The whole contract in one line: the two halves ARE the clip."""
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        prov = _torch_provider(monkeypatch, PROGRAM)

        result = prov.synthesize(TEXT, "af_heart", 1.0, ssml=True)

        assert result.duration_sec == pytest.approx(len(result.audio_pcm) / result.sample_rate)
        assert result.speech_seconds + result.pause_seconds == pytest.approx(result.duration_sec)

    def test_a_content_paced_gap_is_counted_at_ITS_length(self, monkeypatch: pytest.MonkeyPatch):
        """Pacing ON makes the gap after '.' 0.22s. The count must follow the audio.

        The branch most likely to rot: the audio and the tally come from two
        different expressions unless both are derived from the same array.
        """
        monkeypatch.setenv("TTS_CONTENT_PACING_ENABLED", "true")
        prov = _torch_provider(monkeypatch, PROGRAM)

        result = prov.synthesize(TEXT, "af_heart", 1.0, ssml=True)

        # The live path is torch WITH the edge-trim armed, so every gap — flat or
        # content-paced — carries `_TRIM_GAP_COMPENSATION_SECONDS` on top. The tally
        # must follow the audio through that term too, which is the whole point here.
        paced = int((_gap_seconds_for(".") + _TRIM_GAP_COMPENSATION_SECONDS) * SR)
        assert paced != TRIMMED_FLAT_GAP_SAMPLES, "fixture no longer distinguishes the two modes"
        assert result.pause_seconds == pytest.approx((paced + BREAK_SAMPLES) / SR)
        assert result.speech_seconds + result.pause_seconds == pytest.approx(result.duration_sec)

    def test_a_suppressed_gap_adds_nothing(self, monkeypatch: pytest.MonkeyPatch):
        """A mid-sentence boundary inserts no silence, so it must tally none.

        `<emphasis>` splits one sentence into two units with `sentence_final`
        False. Counting a gap there — or inserting one — would both be wrong.
        """
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        prov = _torch_provider(
            monkeypatch,
            [
                [FakeResult(SR, [FakeToken("One", 0.0, 0.4)])],
                [FakeResult(SR, [FakeToken("two", 0.0, 0.4)])],
            ],
        )

        result = prov.synthesize("<speak>One <emphasis>two</emphasis></speak>", "af_heart", 1.0, ssml=True)

        assert len(prov._pipeline.calls) == 2, "fixture must really split into two units"
        assert result.pause_seconds == 0.0
        assert result.speech_seconds == pytest.approx(result.duration_sec)

    def test_a_single_piece_clip_has_no_pause_at_all(self, monkeypatch: pytest.MonkeyPatch):
        """No inter-piece boundary exists, so there is nothing to subtract."""
        prov = _torch_provider(monkeypatch, [[FakeResult(SR, [FakeToken("solo", 0.0, 0.9)])]])

        result = prov.synthesize("Solo.", "af_heart", 1.0, ssml=False)

        assert result.pause_seconds == 0.0
        assert result.speech_seconds == pytest.approx(result.duration_sec)


class TestWhatIsDeliberatelyNotCounted:
    def test_model_emitted_silence_is_not_counted(self, monkeypatch: pytest.MonkeyPatch):
        """Silence kokoro itself emitted is speech-side, not `pause_seconds`.

        Measured on mini2: kokoro-torch's head/tail padding SHRINKS with `speed`
        (~0.30s/~0.44s per piece at 1.0 -> ~0.23s/~0.24s at 1.81), so it belongs
        in the denominator a speed correction divides. Only the fixed zero-sample
        runs this service splices in are unscaled.

        Fixture: a piece whose audio is ENTIRELY zeros. A waveform-measuring
        implementation would count all of it; a sample-counting one counts none.
        """
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        silent = FakeResult(SR, [FakeToken("mumble", 0.0, 0.9)])
        silent.audio = np.zeros(SR, dtype=np.float32)
        prov = _torch_provider(monkeypatch, [[silent]])

        result = prov.synthesize("Mumble.", "af_heart", 1.0, ssml=False)

        assert result.duration_sec == pytest.approx(1.0)
        assert result.pause_seconds == 0.0, "model output must never be reported as inserted padding"
        assert result.speech_seconds == pytest.approx(1.0)

    def test_kitten_reports_no_inserted_pause(self):
        """Kitten does one `generate` call and splices nothing — honestly 0.0."""
        from services.tts_service import KittenProvider

        prov = KittenProvider()

        class _Pipe:
            def generate(self, text: str, voice: str) -> np.ndarray:
                return np.full(SR, 0.1, dtype=np.float32)

        prov._pipeline = _Pipe()

        result = prov.synthesize("Hello there.", "expr-voice-2-f", 1.0, ssml=False)

        assert result.pause_seconds == 0.0
        assert result.speech_seconds == pytest.approx(result.duration_sec)


class TestTheCallerCanActuallyUseIt:
    def test_the_denominator_is_smaller_than_the_duration(self, monkeypatch: pytest.MonkeyPatch):
        """The point of the feature: `speech_seconds` < `duration_sec` on real copy.

        A caller computing `s* = (d * s) / A` against `duration_sec` over-drives
        the speech by exactly the ratio asserted here.
        """
        monkeypatch.delenv("TTS_CONTENT_PACING_ENABLED", raising=False)
        prov = _torch_provider(monkeypatch, PROGRAM)

        result = prov.synthesize(TEXT, "af_heart", 1.0, ssml=True)

        assert 0.0 < result.speech_seconds < result.duration_sec
        assert result.pause_seconds >= _GAP_SECONDS, "at minimum the one inserted pacing gap"

    def test_never_negative_even_if_the_two_disagree(self):
        """A clamped floor, so a caller can divide by it without a guard."""
        from services.tts_service import SynthesisResult

        r = SynthesisResult(
            audio_pcm=np.zeros(0, dtype=np.float32),
            sample_rate=SR,
            duration_sec=0.1,
            provider="kokoro",
            voice="af_heart",
            model="x",
            pause_seconds=0.5,
        )

        assert r.speech_seconds == 0.0


class TestItReachesTheWire:
    """Both response branches carry it, and an old caller is unaffected.

    ⚠️ Merging this repo DEPLOYS it (mini2 host-config-sync, StartInterval=300),
    so the backward-compatibility half is not a nicety: a consumer that has not
    shipped the reader yet must see byte-identical audio and an unchanged
    `duration_sec`.
    """

    PCM = np.full(int(2.5 * SR), 0.1, dtype=np.float32)

    @pytest.fixture
    def client(self, test_client):  # noqa: ANN001, ANN201
        from services.tts_service import SynthesisResult

        class _Svc:
            def loaded_models(self) -> list[str]:
                return ["kokoro"]

            def synthesize(self, text, provider, voice, speed, ssml=None):  # noqa: ANN001, ANN202
                return SynthesisResult(
                    audio_pcm=TestItReachesTheWire.PCM,
                    sample_rate=SR,
                    duration_sec=float(len(TestItReachesTheWire.PCM) / SR),
                    provider="kokoro",
                    voice=voice or "af_heart",
                    model="fake",
                    pause_seconds=0.51,
                )

        test_client.app.state.tts_service = _Svc()
        return test_client

    def test_json_branch_carries_both_fields(self, client):  # noqa: ANN001
        resp = client.post(
            "/api/tts/synthesize",
            json={"text": "Hello world."},
            headers={"accept": "application/json"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["pause_seconds"] == pytest.approx(0.51)
        assert body["speech_seconds"] == pytest.approx(2.5 - 0.51)
        # The pre-existing field is untouched — an old caller reads the same number.
        assert body["duration_sec"] == pytest.approx(2.5)

    def test_wav_branch_carries_both_headers(self, client):  # noqa: ANN001
        resp = client.post("/api/tts/synthesize", json={"text": "Hello world."})
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "audio/wav"
        assert float(resp.headers["x-tts-pause-seconds"]) == pytest.approx(0.51, abs=1e-3)
        assert float(resp.headers["x-tts-speech-seconds"]) == pytest.approx(1.99, abs=1e-3)
        # Unchanged for a caller that reads only this one.
        assert float(resp.headers["x-tts-duration-sec"]) == pytest.approx(2.5, abs=1e-3)
