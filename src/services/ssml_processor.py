"""SSML → kokoro segment lowering for the live /synthesize path.

Ported from tts_service's ssml_processor, but with the kokoro renderer rewritten
from "SSML → one string" to "SSML → an ordered list of typed segments"
(:class:`SsmlSegment`). The string form could not carry a ``<phoneme>`` span
through to ``kokoro.create(is_phonemes=True)``; the segment form can. Each
segment is either plain text (lowered further through normalize/split/misaki) or
a phoneme string fed straight to kokoro, optionally with a prosody-rate speed
override.

Supported tags: ``speak, break, emphasis, prosody, say-as, phoneme, p, s``.
- ``break``     → a real timed silence segment (``break_ms``) spliced into audio
- ``emphasis``  → a per-segment speed *factor* (kokoro's only prosody lever;
  slower = weightier — the closest faithful approximation to acoustic emphasis).
  It is a FACTOR, not an absolute speed, so it scales whatever speed the request
  (or an enclosing ``<prosody rate>``) asked for instead of replacing it.
- ``say-as interpret-as="characters"|"spell-out"`` → HYPHENATED letters
  (``CAB`` → ``C-A-B``) — the form the user's ear-test picked as crisp; spaced /
  dotted over-pause.
- ``phoneme``   → its IPA ``ph`` passed through as a phoneme segment
- ``prosody rate`` → per-segment kokoro ``speed`` (pitch/volume unsupported)
- ``p`` / ``s`` → append a sentence break
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET  # nosec B405 — type reference only; parsing uses defusedxml
from collections.abc import Iterator
from dataclasses import dataclass, replace

import defusedxml.ElementTree as DefusedET

logger = logging.getLogger(__name__)

SUPPORTED_TAGS = {"speak", "break", "emphasis", "prosody", "say-as", "phoneme", "p", "s"}

_SSML_PATTERN = re.compile(r"<(?:speak|break|emphasis|prosody|say-as|phoneme)\b", re.IGNORECASE)
_BREAK_TIME_PATTERN = re.compile(r"(\d+)(ms|s)")
_PROSODY_RATE = {"x-slow": 0.5, "slow": 0.75, "medium": 1.0, "fast": 1.25, "x-fast": 1.5}
# kokoro-onnx asserts 0.5 <= speed <= 2.0.
_SPEED_MIN, _SPEED_MAX = 0.5, 2.0
# Per-level speed factor for <emphasis>. kokoro's ONLY prosody lever is speed, so
# acoustic emphasis is impossible — we approximate it by slowing the emphasized
# span (slower reads as weightier / more deliberate). ``none``/``reduced`` are
# no-ops; a missing/unknown level is treated as ``moderate``. Ear-test-tunable.
#
# ⚠️ These are FACTORS, applied multiplicatively in :func:`effective_speed`. They
# were once written into ``SsmlSegment.speed`` as if absolute, which meant an
# emphasized word in a ``speed=1.5`` request rendered at 0.88 — the request speed
# was silently discarded for exactly the span the author cared most about.
_EMPHASIS_SPEED = {"strong": 0.88, "moderate": 0.94, "reduced": 1.0, "none": 1.0}

# ⚠️ Root cause B (A/V quality wave 8): on a narration beat already stretched to
# the orchestrator's vo_rate floor (0.80), a "strong" emphasis word rendered at
# 0.80 * 0.88 = 0.704 — a further 12% instantaneous rate drop, mid-sentence, on
# top of the fact that <emphasis> also forces its own kokoro utterance boundary
# (a fresh prosody/pitch-contour reset and a join). Measured locally
# (scratchpad/m3_measure.py): the emphasized word rendered at ~0.93 words/sec vs.
# ~1.45-2.86 words/sec for the surrounding plain words in the SAME phrase — the
# "weirdly slows down" the owner heard.
#
# `effective_speed` used to clamp only at kokoro's bare assertion range
# (0.5-2.0), which does nothing to stop a beat's own resolved speed from being
# dragged down further by emphasis. Now the emphasis factor may only move a
# segment's speed within a narrow band of its own resolved (pre-emphasis) speed,
# and never below an absolute intelligibility floor — so a beat that is already
# slow cannot be made audibly slower by emphasis; it renders at the floor
# instead, still slower than plain but no longer alarmingly so. This is the
# belt-side clamp; orchestrator plan-item 1b (not emitting <emphasis> at all on
# an already-floored line) is the braces.
_EMPHASIS_MIN_ABS = 0.85
_EMPHASIS_BAND = 0.15  # max fractional deviation from the segment's own resolved speed


@dataclass
class SsmlSegment:
    """One render unit lowered from SSML.

    ``is_phonemes`` segments bypass normalize/split and go straight to
    ``kokoro.create(is_phonemes=True)``. ``speed`` is an absolute prosody-rate
    override (``None`` = "use the request speed"); ``speed_factor`` is a relative
    ``<emphasis>`` multiplier applied on top of it. Resolve both with
    :func:`effective_speed` rather than reading the fields directly.
    """

    payload: str
    is_phonemes: bool = False
    speed: float | None = None
    break_ms: int | None = None  # pure-silence segment of exactly N ms (no text/create)
    speed_factor: float | None = None  # relative <emphasis> multiplier; None = 1.0


def effective_speed(seg: SsmlSegment, request_speed: float) -> float:
    """Resolve a segment's render speed against the speed the request asked for.

    An absolute ``<prosody rate>`` wins over the request speed; an ``<emphasis>``
    factor then scales whichever of the two applies ("base"). Without an emphasis
    factor, `base` is returned unclamped (a bare ``<prosody rate>`` is already
    clamped to kokoro's range at parse time — see ``_prosody_speed``).

    With an emphasis factor, the product is bounded to a narrow band around
    `base` itself (never more than ``_EMPHASIS_BAND`` fractionally slower/faster
    than the segment would have rendered without emphasis), with an unconditional
    absolute floor of ``_EMPHASIS_MIN_ABS`` that wins even when it sits above the
    band's own ceiling (e.g. a very slow `base`) — emphasis must never be the
    thing that makes an already-slow beat render slower still. See root-cause-B
    note above ``_EMPHASIS_SPEED``.
    """
    base = seg.speed if seg.speed is not None else request_speed
    if seg.speed_factor is None:
        return base
    raw = base * seg.speed_factor
    ceiling = min(_SPEED_MAX, base * (1 + _EMPHASIS_BAND))
    floor = max(_EMPHASIS_MIN_ABS, base * (1 - _EMPHASIS_BAND))
    return min(_SPEED_MAX, max(floor, min(ceiling, raw)))


def is_ssml(text: str) -> bool:
    """Auto-detect whether the input contains SSML tags."""
    return bool(_SSML_PATTERN.search(text))


def _wrap(text: str) -> str:
    return text if text.strip().startswith("<speak") else f"<speak>{text}</speak>"


def _localname(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def validate_ssml(text: str) -> list[str]:
    """Validate SSML markup; return a list of error messages (empty = valid)."""
    errors: list[str] = []
    try:
        root = DefusedET.fromstring(_wrap(text))
    except ET.ParseError as exc:
        return [f"XML parse error: {exc}"]

    for elem in root.iter():
        tag = _localname(elem.tag)
        if tag not in SUPPORTED_TAGS:
            errors.append(f"Unsupported SSML tag: <{tag}>")
    for break_elem in root.iter("break"):
        time_attr = break_elem.get("time")
        if time_attr and not _BREAK_TIME_PATTERN.match(time_attr):
            errors.append(f"Invalid break time format: '{time_attr}' (expected e.g. '500ms' or '1s')")
    for prosody in root.iter("prosody"):
        rate = prosody.get("rate")
        if rate and rate not in _PROSODY_RATE and not re.match(r"^\d+%$", rate):
            errors.append(f"Invalid prosody rate: '{rate}'")
    return errors


def strip_ssml(text: str) -> str:
    """Remove all SSML tags, returning plain text (fallback path)."""
    try:
        root = DefusedET.fromstring(_wrap(text))
        return "".join(root.itertext()).strip()
    except ET.ParseError:
        return re.sub(r"<[^>]+>", "", text).strip()


def _break_ms(elem: ET.Element) -> int | None:
    """Parse a <break time="…"> into a millisecond count (``None`` if unparseable)."""
    match = _BREAK_TIME_PATTERN.match(elem.get("time", "500ms"))
    if not match:
        return None
    val, unit = int(match.group(1)), match.group(2)
    return val if unit == "ms" else val * 1000


def _hyphenate(text: str) -> str:
    """``CAB`` → ``C-A-B`` — crisp letter-by-letter reading (ear-test verdict)."""
    letters = [c for c in text.strip() if not c.isspace()]
    return "-".join(letters)


def _prosody_speed(elem: ET.Element, inherited: float | None) -> float | None:
    rate = elem.get("rate")
    if not rate:
        return inherited
    if rate in _PROSODY_RATE:
        return _PROSODY_RATE[rate]
    match = re.match(r"^(\d+)%$", rate)
    if match:
        return max(_SPEED_MIN, min(_SPEED_MAX, int(match.group(1)) / 100))
    return inherited


def _walk(elem: ET.Element, speed: float | None, factor: float | None = None) -> Iterator[SsmlSegment]:
    """Depth-first walk yielding ordered segments, threading prosody speed + emphasis factor."""
    if elem.text:
        yield SsmlSegment(payload=elem.text, speed=speed, speed_factor=factor)
    for child in elem:
        tag = _localname(child.tag)
        if tag == "phoneme":
            ph = child.get("ph", "")
            if ph.strip():
                yield SsmlSegment(payload=ph, is_phonemes=True, speed=speed, speed_factor=factor)
            elif child.text:  # no ph attr → fall back to reading the inner text
                yield SsmlSegment(payload=child.text, speed=speed, speed_factor=factor)
        elif tag == "break":
            ms = _break_ms(child)
            if ms is not None:
                yield SsmlSegment(payload="", break_ms=ms, speed=speed, speed_factor=factor)
        elif tag == "say-as":
            inner = "".join(child.itertext())
            if child.get("interpret-as", "") in ("characters", "spell-out"):
                yield SsmlSegment(payload=_hyphenate(inner), speed=speed, speed_factor=factor)
            else:
                yield SsmlSegment(payload=inner, speed=speed, speed_factor=factor)
        elif tag == "emphasis":
            # kokoro has no acoustic emphasis; approximate it by slowing the span
            # (slower = weightier). Reuse the same `speed` channel as <prosody rate>.
            level = child.get("level", "moderate")
            level_factor = _EMPHASIS_SPEED.get(level, _EMPHASIS_SPEED["moderate"])
            yield from _walk(child, speed, (factor or 1.0) * level_factor)
        elif tag == "prosody":
            yield from _walk(child, _prosody_speed(child, speed), factor)
        elif tag in ("p", "s"):
            yield from _walk(child, speed, factor)
            yield SsmlSegment(payload=". ", speed=speed, speed_factor=factor)
        else:  # speak / unknown → descend
            yield from _walk(child, speed, factor)
        if child.tail:
            yield SsmlSegment(payload=child.tail, speed=speed, speed_factor=factor)


def _coalesce(segments: list[SsmlSegment]) -> list[SsmlSegment]:
    """Merge adjacent same-speed text segments so a sentence stays contiguous.

    Without this, ``Hello <say-as ...>CAB</say-as>!`` would render as three
    gap-separated pieces (an unnatural pause mid-sentence). Phoneme segments,
    timed ``break_ms`` silences, and speed changes (incl. ``<emphasis>``) each
    force a boundary; everything else joins so ``split_for_tts`` later decides the
    real sentence breaks.
    """
    out: list[SsmlSegment] = []
    for seg in segments:
        if not seg.payload and not seg.is_phonemes and seg.break_ms is None:
            continue
        mergeable = (
            out
            and not seg.is_phonemes
            and seg.break_ms is None
            and not out[-1].is_phonemes
            and out[-1].break_ms is None
            and out[-1].speed == seg.speed
            and out[-1].speed_factor == seg.speed_factor
        )
        if mergeable:
            out[-1] = replace(out[-1], payload=out[-1].payload + seg.payload)
        else:
            out.append(seg)
    return out


def ssml_to_segments(text: str) -> list[SsmlSegment]:
    """Lower an SSML string to an ordered list of render segments."""
    try:
        root = DefusedET.fromstring(_wrap(text))
    except ET.ParseError:
        logger.warning("SSML parse failed; stripping tags and treating as plain text")
        return [SsmlSegment(payload=strip_ssml(text))]
    return _coalesce(list(_walk(root, None)))
