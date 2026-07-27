"""Unit tests for the frozen A+ OSpan strict parser (grammar v4).

Covers the re-verification cases: verbose natural-sentence answers,
parenthetical and bracketed sequences, colon-suffix rejection, compact
sequences, enumerated one-letter-per-line answers, punctuation, prompt
echoes, refusals, markup fragments, empty responses, conjunction-bridged
sequences, quoted sequences, and final-recall-line ranking.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'reanalysis'))
from aplus_rescore_20260718 import strict_parse, positional_credit


def toks(text):
    return strict_parse(text)[0]


def status(text):
    return strict_parse(text)[1]


def test_plain_comma_list():
    assert toks("M, J, F") == ["M", "J", "F"]


def test_plain_space_list():
    assert toks("M J F") == ["M", "J", "F"]


def test_verbose_sentence_no_colon():
    assert toks("The letters are N, K, G.") == ["N", "K", "G"]


def test_verbose_prefix_with_colon():
    assert toks("The letters are: M, J, F.") == ["M", "J", "F"]


def test_sequence_with_trailing_parenthetical():
    assert toks("N K G (letters recalled from the trials)") == ["N", "K", "G"]


def test_bracketed_sequence():
    assert toks("[T F R E]") == ["T", "F", "R", "E"]


def test_colon_suffix_single_letter_rejected():
    assert toks("Item 2: D") == []
    assert status("Item 2: D") == 'none'


def test_compact_allcaps():
    assert toks("MJF") == ["M", "J", "F"]


def test_compact_yes_no_excluded():
    assert toks("YES") == []
    assert toks("NO") == []


def test_enumerated_one_letter_per_line():
    assert toks("1. M\n2. J\n3. F") == ["M", "J", "F"]
    assert status("1. M\n2. J\n3. F") == 'multiline'


def test_verbose_then_answer_line():
    assert toks("Sure! Here are the letters I remember:\nM, J, F") == ["M", "J", "F"]


def test_repeated_letters_kept_for_positional_scoring():
    got = toks("M, M, J, S, F.")
    assert got == ["M", "M", "J", "S", "F"]
    assert positional_credit(got, ["M", "J", "F"]) == 1 / 3


def test_prompt_echo_marker_rejected():
    assert toks("Now recall ALL the letters you were asked to remember") == []


def test_refusal_prose():
    assert toks("I cannot recall the letters from this conversation.") == []


def test_markup_fragment():
    assert toks("</s>") == []


def test_empty_and_whitespace():
    assert toks("") == []
    assert toks("\n  \n") == []
    assert status("") == 'none'


def test_prose_single_a_and_i_not_letters():
    assert toks("I saw a cat and a dog in the garden.") == []


def test_extra_letters_beyond_gold_ignored_by_scoring():
    got = toks("M J F K Q")
    assert positional_credit(got, ["M", "J", "F"]) == 1.0


def test_full_credit_and_zero_credit():
    assert positional_credit(["M", "J", "F"], ["M", "J", "F"]) == 1.0
    assert positional_credit([], ["M", "J", "F"]) == 0.0


def test_conjunction_in_answer_only_line():
    assert toks("N, K, and G") == ["N", "K", "G"]


def test_conjunction_in_anchored_sentence():
    assert toks("The letters were Z, B, X, W, and Y.") == ["Z", "B", "X", "W", "Y"]


def test_quoted_sequence():
    assert toks('"Z B X W Y"') == ["Z", "B", "X", "W", "Y"]


def test_final_recall_line_outranks_interim_fragment():
    text = "C, K\nNow I can give the final recalled letters: C, K, W, M, Y"
    assert toks(text) == ["C", "K", "W", "M", "Y"]
    assert status(text) == 'final'
