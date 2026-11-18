"""Text pre-processing for kokoro synthesis: normalize + sentence/clause split.

Kokoro-onnx phonemizes the whole input in one espeak pass, batches the phoneme
string at ``MAX_PHONEME_LENGTH = 510`` splitting **only** at ``. , ! ? ;``, and
**hard-truncates** any batch past 510 — dropping its tail (a warning only). Em-dashes,
colons, ellipses and parens are not split points, so a long dash/colon span runs past
510 and loses its end: the "swallowed words" defect. Kokoro also silence-trims each
batch (librosa top_db=60) and butts them together with **zero gap**, which strips
natural pauses (rushed) and can clip soft consonants at the seam.

Feeding kokoro pre-split, normalized sentences — and inserting our own inter-sentence
silence downstream — fixes both. Reading numbers/times/initialisms as words fixes
espeak's weak spots (kokoro-onnx uses espeak, not misaki, as its G2P).

This module is deliberately kokoro-free (only ``re`` + ``num2words``, both pure
Python) so it imports on Linux CI where the darwin-gated kokoro-onnx is absent.
"""

from __future__ import annotations

import re

from num2words import num2words

# Curated initialism allow-list, rendered HYPHENATED (``CAB`` -> ``C-A-B``). The
# user's ear-test picked hyphen as best: crisp letter-by-letter articulation with NO
# over-pause — spaced ``C A B`` and dotted ``C.A.B.`` halt between letters ("C … F …
# O"). Hyphenation also forces letter reading, so it is safe against word collisions
# (raw ``IT`` -> "it", ``CAB`` -> "cab", depending on espeak's mood). Extend this set
# per book/domain; do NOT auto-spell every all-caps token — that mangles stylistic
# capitals and shouted words.
_ACRONYMS = frozenset({"IT", "CAB", "COO", "CFO", "CTO", "CEO", "UX", "AI", "AGA", "API", "SLA", "KPI"})

_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})

_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
# ONE alternation for every numeric form, longest-first, so a number is expanded by
# exactly one pass. Running separate decimal → thousands → integer passes let the
# decimal pattern bite the tail off a grouped number: "1,234.56" matched "234.56"
# first, leaving "one,two hundred and thirty-four point five six". Grouped-with-
# decimal must therefore be tried before bare-decimal, and bare-decimal before
# bare-integer.
_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b|\b\d+\b")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,5})\b")
_DASH_RE = re.compile(r"\s*[—–]\s*")  # em/en dash
_WS_RE = re.compile(r"\s+")

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")

# Titles/abbreviations whose trailing period is NOT a sentence end. Splitting there
# costs twice: each piece is synthesized as its own kokoro utterance with
# ``sentence_final=True``, so "Dr. Smith" gets a ~0.17 s inter-sentence hole AND a
# prosody restart in the middle of a name. Case-sensitive on purpose — the
# lowercase forms that really occur (``vs.``, ``etc.``, ``al.``) are listed
# explicitly, and matching case-insensitively would suppress real sentence ends
# ("…say no. Then we left.").
_ABBREVIATIONS = (
    # personal / honorific
    "Dr|Mr|Mrs|Ms|Mx|Prof|Rev|Hon|Sr|Jr|St|Capt|Col|Gen|Lt|Sgt|Messrs"
    # reference / editorial
    "|Fig|Figs|No|Nos|Vol|Vols|Ch|Sec|Eq|Ref|Refs|pp|Ed|Eds|Op|Mt"
    # organisational
    "|Inc|Ltd|Co|Corp|Dept|Univ|Est|Assn|Bros"
    # latin / measure, lowercase as they actually appear in prose
    "|vs|etc|al|cf|ca|approx|viz|ibid"
)

# A boundary candidate is suppressed when the text to its LEFT ends in one of:
#
#   1. a listed abbreviation, itself preceded by start-of-text/space/open-bracket
#      (so "Waterloo." does not match the listed "Co.", nor "Disco." "Co.");
#   2. a dotted letter chain — "e.g.", "i.e.", "U.S.", "A.B.C." — which is the
#      generic form and covers every multi-dot abbreviation without listing them;
#   3. a single capital initial ("J." in "J. Smith").
#
# A decimal ("3.5") needs no rule: ``_SENT_RE`` only fires on a period followed by
# WHITESPACE, so a decimal point was never a boundary candidate in the first place
# (asserted in the tests so a future rewrite of _SENT_RE cannot lose that).
_NO_SPLIT_AFTER_RE = re.compile(
    r"(?:^|[\s(\[\"'])(?:" + _ABBREVIATIONS + r")\.$"
    r"|(?:[A-Za-z]\.){2,}$"
    r"|(?:^|[\s(\[\"'])[A-Z]\.$"
)


def _split_sentences(text: str) -> list[str]:
    """``_SENT_RE.split`` minus the boundaries that are abbreviations, not sentences.

    Deliberately still a pure-``re`` decision: ``_SENT_RE`` finds the candidates and
    ``_NO_SPLIT_AFTER_RE`` vetoes them. A negative lookbehind inside ``_SENT_RE``
    itself is not an option — Python's ``re`` requires every lookbehind alternative
    to be the same fixed width, and "Dr" / "approx" / "e.g" are not.

    A suppressed boundary merely makes the piece longer; ``split_for_tts`` still
    sub-splits anything over ``max_chars``, so an over-eager entry in the
    abbreviation set degrades to a longer utterance, never to a dropped one.
    """
    pieces: list[str] = []
    start = 0
    for m in _SENT_RE.finditer(text):
        left = text[start : m.start()]
        if _NO_SPLIT_AFTER_RE.search(left):
            continue
        pieces.append(left)
        start = m.end()
    pieces.append(text[start:])
    return pieces


def _number_to_words(m: re.Match[str]) -> str:
    """Expand one number token — grouped, decimal, or plain — to speakable words.

    The fractional part is spoken DIGIT BY DIGIT, which is both the ordinary reading
    ("ten point five zero") and the only one that survives a trailing zero. Routing
    it through ``num2words(float(...))`` silently dropped them — ``float("10.50")``
    is ``10.5``, so a price rendered as "ten point five" and ``1,999.90`` as
    "…point nine". The integer part keeps ``num2words`` so grouped numbers still
    read naturally.
    """
    raw = m.group(0).replace(",", "")
    if "." not in raw:
        return num2words(int(raw))
    whole, _, frac = raw.partition(".")
    spoken_frac = " ".join(num2words(int(d)) for d in frac)
    return f"{num2words(int(whole or 0))} point {spoken_frac}"


def _time_to_words(m: re.Match[str]) -> str:
    h, mm = int(m.group(1)), int(m.group(2))
    if h > 23 or mm > 59:
        # Not a clock time (e.g. a ratio/score); leave the digits for int expansion.
        return m.group(0)
    # Read on a 12-hour clock so 24h times sound natural: 17:00 -> "five o'clock",
    # 00:15 -> "twelve fifteen", 13:30 -> "one thirty". No am/pm — the surrounding
    # prose carries that, and "five o'clock" reads cleaner than "seventeen o'clock".
    h12 = h % 12 or 12
    if mm == 0:
        return f"{num2words(h12)} o'clock"
    if mm < 10:
        return f"{num2words(h12)} oh {num2words(mm)}"
    return f"{num2words(h12)} {num2words(mm)}"


def normalize_text_for_tts(text: str) -> str:
    """Expand numbers, times, decimals, dashes, symbols and initialisms to speakable text.

    Pure + idempotent-ish on already-clean prose. Ordering matters: dashes become
    commas (a pause AND a kokoro split point) before number expansion; times run
    before the number pass so a clock reading wins its span; every other numeric
    form is expanded by a SINGLE alternation (see ``_NUMBER_RE``) so no pass can
    eat another's digits; acronyms hyphenate last so num2words' own hyphens
    (``twenty-four``) are untouched (they are lowercase, not ``[A-Z]{2,5}``).
    """
    text = text.translate(_SMART_QUOTES)
    text = text.replace("…", "...")  # ellipsis
    text = _DASH_RE.sub(", ", text)
    text = _TIME_RE.sub(_time_to_words, text)
    text = _NUMBER_RE.sub(_number_to_words, text)
    text = text.replace("&", " and ").replace("%", " percent")
    text = _ACRONYM_RE.sub(lambda m: "-".join(m.group(1)) if m.group(1) in _ACRONYMS else m.group(1), text)
    return _WS_RE.sub(" ", text).strip()


def _word_pack(piece: str, max_chars: int) -> list[str]:
    """Last resort for a clause still longer than max_chars: pack whole words."""
    out: list[str] = []
    cur = ""
    for w in piece.split():
        if cur and len(cur) + len(w) + 1 > max_chars:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def _subsplit(sentence: str, max_chars: int) -> list[str]:
    """Split an over-long sentence at clause boundaries (``, ; :``), then word-pack."""
    parts: list[str] = []
    cur = ""
    for piece in _CLAUSE_RE.split(sentence):
        if cur and len(cur) + len(piece) + 1 > max_chars:
            parts.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        parts.append(cur)
    out: list[str] = []
    for p in parts:
        out.extend([p] if len(p) <= max_chars else _word_pack(p, max_chars))
    return out


def split_for_tts(text: str, max_chars: int = 320) -> list[str]:
    """Sentence-split, sub-splitting any over-long sentence to well under 510 phonemes.

    ~320 chars is a safe target: kokoro truncates at 510 *phonemes*, and English
    prose is well under one phoneme per character, so a 320-char piece never
    approaches the cap even for dense passages.
    """
    out: list[str] = []
    for s in _split_sentences(text):
        s = s.strip()
        if not s:
            continue
        out.extend([s] if len(s) <= max_chars else _subsplit(s, max_chars))
    return out
