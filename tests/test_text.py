import pytest

from axomiya_ocr.data.synthetic import extract_lines, stable_split
from axomiya_ocr.data.text import (
    assamese_script_ratio,
    assert_assamese_source,
    ctc_required_steps,
    normalize_label,
    validate_assamese_label,
)


def test_normalize_preserves_assamese_specific_characters() -> None:
    assert normalize_label("  অসমৰ   ৰাইজৰ ৱেব  ") == "অসমৰ ৰাইজৰ ৱেব"


def test_assamese_ratio() -> None:
    assert assamese_script_ratio("অসমীয়া ১২৩") == 1.0
    assert assamese_script_ratio("English only") == 0.0


def test_ctc_required_steps_counts_repeated_symbols() -> None:
    assert ctc_required_steps("কলা") == 3
    assert ctc_required_steps("কক") == 3


def test_rejects_wrong_dataset_config() -> None:
    assert_assamese_source("assamese")
    with pytest.raises(ValueError, match="only accepts 'assamese'"):
        assert_assamese_source("bengali")


def test_validates_assamese_and_rejects_english() -> None:
    assert validate_assamese_label("মই অসমীয়া কওঁ")[0]
    assert not validate_assamese_label("only English")[0]


def test_synthetic_text_stays_assamese_and_split_is_stable() -> None:
    lines = list(extract_lines("অসমীয়া আমাৰ মাতৃভাষা। This sentence is English."))
    assert lines == ["অসমীয়া আমাৰ মাতৃভাষা।"]
    assert stable_split("article-1") == stable_split("article-1")
