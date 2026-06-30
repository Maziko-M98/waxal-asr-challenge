from waxal_asr.metrics import char_error_rate, waxal_score, word_error_rate
from waxal_asr.normalization import normalize_text


def test_normalize_text_removes_punctuation_and_extra_spaces():
    assert normalize_text("  Hello,   Waxal!  ") == "hello waxal"


def test_word_error_rate_exact_match():
    assert word_error_rate(["waxal nlp"], ["waxal nlp"]) == 0.0


def test_word_error_rate_single_substitution():
    assert word_error_rate(["waxal nlp"], ["waxal asr"]) == 0.5


def test_char_error_rate_single_deletion():
    assert char_error_rate(["abc"], ["ab"]) == 1 / 3


def test_waxal_score_is_average_of_wer_and_cer():
    score = waxal_score(["abc"], ["ab"])
    assert score["error"] == 0.5 * score["wer"] + 0.5 * score["cer"]
    assert score["score"] == 1.0 - score["error"]
