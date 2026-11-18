"""misaki grapheme-to-phoneme wrapper for the kokoro path (Apple-Silicon only).

kokoro-onnx phonemizes with **espeak** — its out-of-distribution fallback G2P —
not **misaki**, the dictionary-based G2P the kokoro model was actually trained
on. espeak mis-reads proper nouns, initialisms and homographs; misaki reads them
from its lexicon (+ num2words + spaCy) and only falls back to espeak for truly
unknown words. Feeding misaki's phonemes to ``kokoro.create(..., is_phonemes=
True)`` keeps the light ONNX model (no torch/KPipeline migration) while lifting
pronunciation quality.

This module is deliberately kept OUT of ``text_preproc`` (which stays
``re``+``num2words`` only so it imports on Linux CI). misaki + spaCy are
darwin-gated deps; every import here is lazy and optional, so a missing/broken
misaki degrades to ``None`` and the caller falls back to the espeak path — misaki
must never turn a working narration into a 500.

Gated by the ``TTS_G2P`` env var (``espeak`` default | ``misaki``) so a bad
rollout is a one-env-flip rollback with no code change.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

# ``espeak`` (today's behaviour) unless explicitly opted into misaki. Read once
# at import; flipping requires a service restart, which is the deploy boundary.
_G2P_ENGINE = os.getenv("TTS_G2P", "espeak").strip().lower()

_g2p = None  # lazy misaki en.G2P singleton
_load_lock = threading.Lock()
_load_failed = False


def misaki_enabled() -> bool:
    """True when the operator has opted into the misaki G2P path."""
    return _G2P_ENGINE == "misaki"


def _get_g2p():
    """Lazily build the misaki G2P singleton, or return ``None`` if unavailable.

    Construction is guarded so the first synth after a restart pays the spaCy
    warmup once. Any import/build failure is logged once and latched to ``None``
    so we don't retry on every call.
    """
    global _g2p, _load_failed
    if _g2p is not None:
        return _g2p
    if _load_failed:
        return None
    with _load_lock:
        if _g2p is not None:
            return _g2p
        if _load_failed:
            return None
        try:
            from misaki import en, espeak

            # unk='' → an unresolved word contributes no phonemes (silence) rather
            # than a '❓' glyph that kokoro's tokenizer would drop anyway; the
            # EspeakFallback resolves almost everything before that can happen.
            _g2p = en.G2P(
                trf=False,
                british=False,
                fallback=espeak.EspeakFallback(british=False),
                unk="",
            )
            logger.info("misaki G2P loaded (TTS_G2P=misaki)")
        except Exception:
            _load_failed = True
            logger.exception("misaki G2P unavailable — falling back to espeak path")
            return None
    return _g2p


def phonemize_sentence(text: str) -> str | None:
    """Return misaki phonemes for ``text``, or ``None`` to use the espeak path.

    ``None`` when misaki is disabled (``TTS_G2P != misaki``), unavailable, or
    raised on this input. The returned string is fed to
    ``kokoro.create(phonemes, is_phonemes=True)``.
    """
    if not misaki_enabled():
        return None
    g2p = _get_g2p()
    if g2p is None:
        return None
    try:
        phonemes, _ = g2p(text)
    except Exception:  # noqa: BLE001 — a misaki/spaCy/espeak hiccup on one piece
        # must never fail the request; the caller falls back to espeak for it.
        logger.warning("misaki phonemize failed for a piece; falling back to espeak", exc_info=True)
        return None
    phonemes = (phonemes or "").strip()
    return phonemes or None
