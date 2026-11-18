"""Map native ONNX phoneme spans to misaki words without an ASR/model pass."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=2)
def _g2p(british: bool) -> Any:
    from misaki import en, espeak

    return en.G2P(trf=False, british=british, fallback=espeak.EspeakFallback(british=british), unk="")


def tokenize(text: str, british: bool) -> tuple[str, list[Any]]:
    phonemes, tokens = _g2p(british)(text)
    return (phonemes or "").strip(), tokens


def words_from_timings(tokens: list[Any], timings: list[Any], known: Any) -> list[tuple[str, float, float]]:
    """Consume the exact surviving phoneme stream; never guess a text alignment.

    Punctuation and whitespace advance the stream but do not become caption words.
    A tokenizer mismatch fails the request rather than returning plausible bad spans.
    """
    actual = "".join(t.phoneme for t in timings)
    annotated: list[tuple[str, str | None]] = []
    for token in tokens:
        word = token.text.strip() if any(c.isalnum() for c in token.text) else None
        annotated.extend((p, word) for p in known(token.phonemes or ""))
        if token.whitespace:
            annotated.extend((p, None) for p in known(" "))
    while annotated and annotated[-1][0].isspace():
        annotated.pop()
    while annotated and annotated[0][0].isspace():
        annotated.pop(0)
    if "".join(p for p, _ in annotated) != actual:
        raise ValueError("ONNX phoneme timings do not match the word-token stream")
    words: list[tuple[str, float, float]] = []
    previous_owner: str | None = None
    for (_, owner), timing in zip(annotated, timings, strict=True):
        if owner is not None:
            if owner == previous_owner and words:
                words[-1] = (owner, words[-1][1], float(timing.end))
            else:
                words.append((owner, float(timing.start), float(timing.end)))
        previous_owner = owner
    return words
