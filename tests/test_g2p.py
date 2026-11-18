"""Unit tests for the misaki G2P wrapper (services/g2p.py).

The disabled-path tests run everywhere (Linux CI included) because
``phonemize_sentence`` short-circuits before importing misaki when
``TTS_G2P != misaki``. The enabled-path tests need misaki, which is darwin-gated,
so they skip when it can't be imported.

We monkeypatch the module-level ``_G2P_ENGINE`` directly (auto-restored by the
fixture) rather than reloading the module — reloading would leave the engine
global mutated in the shared module dict and pollute other tests.
"""

from __future__ import annotations

import pytest

import services.g2p as g2p


def _set_engine(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setattr(g2p, "_G2P_ENGINE", value)
    # Reset the lazy singleton state so each test starts clean.
    monkeypatch.setattr(g2p, "_g2p", None)
    monkeypatch.setattr(g2p, "_load_failed", False)


def test_disabled_by_default() -> None:
    # The module default (no TTS_G2P set at import) is espeak.
    assert g2p._G2P_ENGINE == "espeak"
    assert g2p.misaki_enabled() is False


def test_disabled_when_engine_is_espeak(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_engine(monkeypatch, "espeak")
    assert g2p.misaki_enabled() is False
    # Returns None without importing misaki — the espeak fallback path.
    assert g2p.phonemize_sentence("The C-A-B met the C-F-O.") is None


def test_enabled_flag_reads_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_engine(monkeypatch, "misaki")
    assert g2p.misaki_enabled() is True


def test_missing_misaki_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled but misaki unavailable → None (caller falls back to espeak)."""
    _set_engine(monkeypatch, "misaki")
    monkeypatch.setattr(g2p, "_load_failed", True)  # simulate a prior import failure
    assert g2p.phonemize_sentence("Anything at all.") is None


@pytest.mark.parametrize(
    "text",
    ["The C-A-B met the C-F-O about I-T.", "Koster and de Jong visited the café."],
)
def test_misaki_phonemizes_when_available(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    pytest.importorskip("misaki", reason="misaki is darwin-gated; not on Linux CI")
    _set_engine(monkeypatch, "misaki")
    phon = g2p.phonemize_sentence(text)
    assert isinstance(phon, str) and phon.strip(), f"expected phonemes for {text!r}"


def test_misaki_disambiguates_homograph(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real misaki win espeak can't do: 'read' present vs past differ."""
    pytest.importorskip("misaki", reason="misaki is darwin-gated; not on Linux CI")
    _set_engine(monkeypatch, "misaki")
    present = g2p.phonemize_sentence("I read books every day.")
    past = g2p.phonemize_sentence("I read that book last year.")
    assert present is not None and past is not None
    assert present != past
