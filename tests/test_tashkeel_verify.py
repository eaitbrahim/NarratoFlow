import pytest

from narratoflow.text.tashkeel.verify import (
    SkeletonMismatchError,
    assert_skeleton_preserved,
    strip_diacritics,
)


def test_strip_removes_harakat_and_tatweel() -> None:
    assert strip_diacritics("ٱلْعِلْمُ نُورٌ") == "ٱلعلم نور"
    assert strip_diacritics("كـتـاب") == "كتاب"


def test_adding_diacritics_is_allowed() -> None:
    assert_skeleton_preserved("العلم نور", "اَلْعِلْمُ نُورٌ")


def test_changed_word_is_rejected() -> None:
    with pytest.raises(SkeletonMismatchError):
        assert_skeleton_preserved("العلم نور", "اَلْعِلْمُ ضِيَاءٌ")


def test_english_passes_through() -> None:
    assert strip_diacritics("Hello") == "Hello"
