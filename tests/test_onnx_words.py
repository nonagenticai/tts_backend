"""Native duration alignment and portable profile validation, without inference."""

from types import SimpleNamespace

import numpy as np
import pytest

from services.onnx_words import words_from_timings
from services.tts_service import KokoroProvider, ProviderUnavailableError


def token(text, phonemes, whitespace=""):
    return SimpleNamespace(text=text, phonemes=phonemes, whitespace=whitespace)


def timings(phonemes):
    return [SimpleNamespace(phoneme=p, start=i / 10, end=(i + 1) / 10) for i, p in enumerate(phonemes)]


def test_words_consume_spaces_punctuation_and_repeated_words():
    tokens = [token("go", "go", " "), token("go", "go"), token("!", "!")]
    assert words_from_timings(tokens, timings("go go!"), lambda x: x) == [
        ("go", 0.0, 0.2),
        ("go", 0.3, 0.5),
    ]


def test_unknown_phonemes_filtered_exactly_as_model():
    tokens = [token("hello", "h?i", " "), token("world", "w")]
    assert words_from_timings(tokens, timings("hi w"), lambda x: x.replace("?", "")) == [
        ("hello", 0.0, 0.2),
        ("world", 0.3, 0.4),
    ]


def test_alignment_mismatch_is_not_silently_guessed():
    with pytest.raises(ValueError, match="do not match"):
        words_from_timings([token("hello", "hi")], timings("ho"), lambda x: x)


def test_custom_profile_is_loaded_and_retains_public_id(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_ENGINE", "kokoro-onnx")
    monkeypatch.setenv("KOKORO_CUSTOM_VOICES_DIR", str(tmp_path))
    pack = np.ones((510, 1, 256), dtype=np.float32)
    np.save(tmp_path / "am_local_v1.npy", pack)
    provider = KokoroProvider()
    provider.voices = ["af_heart"]
    provider._load_custom_voices()
    assert provider.voices == ["af_heart", "am_local_v1"]
    np.testing.assert_array_equal(provider._resolve_voice("am_local_v1"), pack)
    assert provider._resolve_voice("af_heart") == "af_heart"


@pytest.mark.parametrize("kind", ["nan", "shape", "dtype", "shadow"])
def test_bad_custom_profiles_refuse_load(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("TTS_ENGINE", "kokoro-onnx")
    monkeypatch.setenv("KOKORO_CUSTOM_VOICES_DIR", str(tmp_path))
    pack = np.ones((510, 1, 256), dtype=np.float32)
    name = "am_local_v1"
    if kind == "nan":
        pack[0, 0, 0] = np.nan
    elif kind == "shape":
        pack = pack[:1]
    elif kind == "dtype":
        pack = pack.astype(np.float64)
    elif kind == "shadow":
        name = "af_heart"
    np.save(tmp_path / f"{name}.npy", pack)
    provider = KokoroProvider()
    provider.voices = ["af_heart"]
    with pytest.raises(ProviderUnavailableError):
        provider._load_custom_voices()


def test_onnx_trim_rebases_real_native_word_offsets(monkeypatch):
    from services import onnx_words

    monkeypatch.setenv("TTS_ENGINE", "kokoro-onnx")
    monkeypatch.setenv("TTS_TRIM_MODEL_PADDING", "true")
    monkeypatch.setattr(onnx_words, "tokenize", lambda text, british: ("hi", [token("Hi", "hi")]))
    audio = np.concatenate([np.zeros(4800), np.ones(12000) * 0.1, np.zeros(4800)]).astype(np.float32)

    def create_timed(*args, **kwargs):
        assert kwargs["trim"] is False
        assert kwargs["sentence_pause"] == kwargs["clause_pause"] == 0
        return (
            audio,
            24000,
            [SimpleNamespace(phoneme="h", start=0.2, end=0.4), SimpleNamespace(phoneme="i", start=0.4, end=0.7)],
        )

    provider = KokoroProvider()
    provider._kokoro = SimpleNamespace(create_timed=create_timed, tokenizer=SimpleNamespace(known=lambda x: x))
    samples, sr, words = provider._create_piece_onnx_timed("Hi", "af_heart", 1, False)
    assert len(samples) == 12000 + 1920
    assert words[0].word == "Hi"
    assert words[0].start == pytest.approx(0.04)
    assert words[0].end == pytest.approx(0.54)
