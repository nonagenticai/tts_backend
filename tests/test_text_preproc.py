"""Unit tests for the kokoro TTS text pre-processor (normalize + split).

These are pure-function tests — no kokoro-onnx (darwin-gated), so they run on Linux
CI. They pin the behaviour the user's ear-test settled on: hyphenated initialisms,
numbers/times/decimals read as words, em-dash as a comma pause, and sentence/clause
splitting that keeps every piece well under kokoro-onnx's 510-phoneme truncation cap.
"""

from __future__ import annotations

import pytest

from services.text_preproc import normalize_text_for_tts, split_for_tts


class TestNormalizeNumbers:
    def test_bare_integer_to_words(self):
        assert normalize_text_for_tts("There were 43 tickets.") == "There were forty-three tickets."

    def test_thousands_separator(self):
        assert normalize_text_for_tts("1,240 more") == "one thousand, two hundred and forty more"

    def test_grouped_number_with_decimal_expands_as_one_number(self):
        # The decimal pass used to run BEFORE the thousands pass, so "1,234.56"
        # matched "234.56" first and the grouped number was torn in half:
        # "one,two hundred and thirty-four point five six". One alternation now
        # claims the whole token.
        out = normalize_text_for_tts("Revenue was 1,234.56 last year")
        assert "one thousand" in out
        assert "point five six" in out
        assert "one,two" not in out

    def test_a_trailing_zero_survives_the_decimal_expansion(self):
        # Routing the fraction through num2words(float(...)) DROPPED trailing zeros:
        # float("10.50") is 10.5, so a price was spoken as "ten point five" and
        # 1,999.90 as "...point nine". The fraction is now read digit by digit,
        # which is both the ordinary reading and the only one that keeps the zero.
        assert normalize_text_for_tts("The price is 10.50 per unit") == ("The price is ten point five zero per unit")
        assert "point nine zero" in normalize_text_for_tts("Total: 1,999.90 euros")
        # A number with no trailing zero is unchanged.
        assert normalize_text_for_tts("Policy 11.4 applies") == "Policy eleven point four applies"

    def test_decimal_reads_as_point(self):
        assert normalize_text_for_tts("Policy 11.4 applies") == "Policy eleven point four applies"

    def test_time_on_the_hour(self):
        assert normalize_text_for_tts("at 09:00 sharp") == "at nine o'clock sharp"

    def test_time_single_digit_minute_uses_oh(self):
        assert normalize_text_for_tts("at 08:05") == "at eight oh five"

    def test_time_regular(self):
        assert normalize_text_for_tts("at 08:40") == "at eight forty"

    def test_time_24h_converted_to_12h(self):
        assert normalize_text_for_tts("by 17:00") == "by five o'clock"

    def test_time_24h_with_minutes(self):
        assert normalize_text_for_tts("at 13:30") == "at one thirty"

    def test_time_midnight_reads_as_twelve(self):
        assert normalize_text_for_tts("at 00:15") == "at twelve fifteen"

    def test_time_noon_reads_as_twelve(self):
        assert normalize_text_for_tts("at 12:00") == "at twelve o'clock"

    def test_non_time_ratio_left_for_int_expansion(self):
        # 88:12 is not a clock time (hour > 23) → not read as a time.
        assert "o'clock" not in normalize_text_for_tts("ratio 88:12")


class TestNormalizeAcronyms:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("CAB", "C-A-B"), ("CFO", "C-F-O"), ("IT", "I-T"), ("KPI", "K-P-I")],
    )
    def test_allowlisted_acronym_hyphenated(self, raw, expected):
        assert normalize_text_for_tts(f"The {raw} decided.") == f"The {expected} decided."

    def test_non_allowlisted_caps_untouched(self):
        # Stylistic/unknown all-caps must NOT be auto-spelled.
        assert normalize_text_for_tts("He yelled STOP loudly.") == "He yelled STOP loudly."

    def test_single_capital_letter_untouched(self):
        assert normalize_text_for_tts("I Understand the risk.") == "I Understand the risk."


class TestNormalizeSymbolsAndDashes:
    def test_em_dash_becomes_comma_pause(self):
        assert normalize_text_for_tts("wait — then go") == "wait, then go"

    def test_ampersand_and_percent(self):
        assert normalize_text_for_tts("50% more & better") == "fifty percent more and better"

    def test_smart_quotes_normalized(self):
        assert normalize_text_for_tts("“quote”") == '"quote"'

    def test_clean_prose_is_stable(self):
        assert normalize_text_for_tts("Nothing to change here.") == "Nothing to change here."

    def test_collapses_whitespace(self):
        assert normalize_text_for_tts("a\n\n  b   c") == "a b c"


class TestSplit:
    def test_splits_on_sentence_boundaries(self):
        assert split_for_tts("First. Second! Third?") == ["First.", "Second!", "Third?"]

    def test_short_text_is_one_piece(self):
        assert split_for_tts("Just one line.") == ["Just one line."]

    def test_drops_empty_fragments(self):
        assert split_for_tts("   ") == []

    def test_long_sentence_subsplit_at_clauses(self):
        # A single over-long sentence (no terminal punctuation) must still be broken
        # into pieces each within max_chars, split at clause boundaries.
        clause = "and it kept going on and on"
        text = ", ".join([clause] * 12) + "."
        pieces = split_for_tts(text, max_chars=80)
        assert len(pieces) > 1
        assert all(len(p) <= 80 for p in pieces)

    def test_over_long_clause_word_packed(self):
        # No clause/sentence punctuation at all → must fall back to word-packing.
        text = "word " * 60
        pieces = split_for_tts(text.strip(), max_chars=50)
        assert len(pieces) > 1
        assert all(len(p) <= 50 for p in pieces)


class TestAbbreviationsDoNotEndASentence:
    """`(?<=[.!?])\\s+` alone split "Dr. Smith" into TWO kokoro utterances.

    Each piece is synthesized with `sentence_final=True`, so the listener gets a
    ~0.17 s hole AND a prosody restart in the middle of a name. Both directions
    matter: an abbreviation must NOT split, and a genuine sentence boundary must.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Smith arrived early.",
            "Mr. and Mrs. Jones left.",
            "See e.g. the second chart.",
            "That is i.e. the whole point.",
            "Ford vs. Ferrari was close.",
            "Fig. 4 shows the trend.",
            "No. 7 comes next.",
            "Bread, milk, etc. went in the bag.",
            "J. Smith spoke first.",
            "The U.S. economy grew.",
            "Acme Co. shipped it.",
            "(Dr. Smith) arrived.",
        ],
    )
    def test_an_abbreviation_stays_in_one_segment(self, text):
        assert split_for_tts(text) == [text]

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("First sentence. Second sentence.", ["First sentence.", "Second sentence."]),
            ("Really? Yes! Okay.", ["Really?", "Yes!", "Okay."]),
            # The abbreviation set must not swallow a REAL boundary that merely ends
            # in the same letters: "Waterloo." is not the listed "Co.", and
            # "disco." is not "Co." either — the left-boundary requirement is what
            # makes that hold.
            ("He fought at Waterloo. Then he rested.", ["He fought at Waterloo.", "Then he rested."]),
            ("We danced to disco. It was loud.", ["We danced to disco.", "It was loud."]),
            # An abbreviation mid-sentence must not suppress the boundary at the END.
            ("Dr. Smith arrived. He sat down.", ["Dr. Smith arrived.", "He sat down."]),
        ],
    )
    def test_a_real_sentence_boundary_still_splits(self, text, expected):
        assert split_for_tts(text) == expected

    def test_a_decimal_point_is_not_a_boundary_candidate(self):
        # No suppression rule is needed for this (the boundary regex requires
        # WHITESPACE after the period) — asserted so a future rewrite cannot lose it.
        assert split_for_tts("The rate was 3.5 percent.") == ["The rate was 3.5 percent."]

    def test_a_suppressed_boundary_is_still_length_capped(self):
        # An over-eager abbreviation entry must degrade to a LONGER piece, never to
        # a piece over max_chars: `_subsplit` still runs on the merged sentence.
        text = "Dr. " + "and it kept going on and on, " * 10 + "end."
        pieces = split_for_tts(text, max_chars=80)
        assert all(len(p) <= 80 for p in pieces), [len(p) for p in pieces]
