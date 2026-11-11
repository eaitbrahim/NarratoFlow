from pathlib import Path

import pytest
from conftest import FIXTURES

from narratoflow.errors import CorruptInputError
from narratoflow.models import Language
from narratoflow.text.quality import (
    IssueKind,
    check_input,
    normalize_input,
    require_clean,
)


def read(name: str) -> str:
    return (FIXTURES / "input" / name).read_text(encoding="utf-8")


def test_clean_arabic_passes() -> None:
    assert check_input(read("clean_ar.txt"), Language.ARABIC).ok


def test_clean_english_passes() -> None:
    assert check_input(read("clean_en.txt"), Language.ENGLISH).ok


def test_corrupt_arabic_flagged_with_locations() -> None:
    report = check_input(read("corrupt_ar.txt"), Language.ARABIC)
    kinds = {i.kind for i in report.issues}
    assert IssueKind.STRETCHED_LETTERS in kinds and IssueKind.PAGE_NUMBER in kinds
    assert all(i.line >= 1 for i in report.issues)
    assert all(len(i.excerpt) <= 40 for i in report.issues)


def test_tatweel_is_normalized_not_flagged() -> None:
    assert normalize_input("ي\u0640\u0640\u0640\u0640\u0640\u0640\u0640\u0640وحنا") == "يوحنا"  # noqa: RUF001
    assert check_input("الحم\u0640\u0640\u0640\u0640د لله", Language.ARABIC).ok


def test_vowel_interleaved_stretching_flagged() -> None:
    assert not check_input("يَيَيَ المعمدان", Language.ARABIC).ok


def test_bom_crlf_and_bidi_marks_not_flagged() -> None:
    text = "\ufeff" + read("clean_ar.txt").replace("\n", "\r\n").replace(" ", " \u200f", 5)
    assert check_input(text, Language.ARABIC).ok


def test_bidi_embeddings_and_isolates_removed() -> None:
    marks = "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
    assert normalize_input(f"\u0642\u0627\u0644{marks} abc") == "\u0642\u0627\u0644 abc"
    text = read("clean_ar.txt").replace(" ", " \u2067", 5).replace(".", ".\u2069", 3)
    assert check_input(text, Language.ARABIC).ok


def test_normalize_keeps_diacritics() -> None:
    assert normalize_input("اَلْعِلْمُ\r\n") == "اَلْعِلْمُ\n"


def test_legit_doubled_letters_not_stretched() -> None:
    assert check_input("إنّ الله مع الصابرين، واللّيل إذا يسر", Language.ARABIC).ok


def test_require_clean_raises_with_line_numbers() -> None:
    with pytest.raises(CorruptInputError, match=r"line \d+: STRETCHED_LETTERS"):
        require_clean(read("corrupt_ar.txt"), Language.ARABIC)


def test_require_clean_returns_normalized() -> None:
    raw = "\ufeff\u0645\u0631\u062d\u0628\u0627\r\n\u0628\u0627\u0628\u0627\u0628"
    assert (
        require_clean(raw, Language.ARABIC)
        == "\u0645\u0631\u062d\u0628\u0627\n\u0628\u0627\u0628\u0627\u0628"
    )


def test_latin_stretching_and_replacement_char() -> None:
    report = check_input("fine\nheeeello\nbad \ufffd char\nprivate \ue000", Language.ENGLISH)
    assert [(i.kind, i.line) for i in report.issues] == [
        (IssueKind.STRETCHED_LETTERS, 2),
        (IssueKind.REPLACEMENT_CHAR, 3),
        (IssueKind.REPLACEMENT_CHAR, 4),
    ]


def test_page_number_lines() -> None:
    text = "hello world\n31\n\u0663\u0667\n- 12 -\n\u06f4\u06f5\nbye now"
    report = check_input(text, Language.ENGLISH)
    pages = [(i.kind, i.line) for i in report.issues if i.kind is IssueKind.PAGE_NUMBER]
    assert pages == [
        (IssueKind.PAGE_NUMBER, 2),
        (IssueKind.PAGE_NUMBER, 3),
        (IssueKind.PAGE_NUMBER, 4),
        (IssueKind.PAGE_NUMBER, 5),
    ]


@pytest.mark.parametrize("scene_break", ["* * *", "---", "...", "\u06de", "- ", "***"])
def test_scene_break_lines_pass(scene_break: str) -> None:
    for name, lang in (("clean_en.txt", Language.ENGLISH), ("clean_ar.txt", Language.ARABIC)):
        # Two copies of the script so one short break line stays under the fragment ratio.
        text = f"{read(name).rstrip()}\n\n{scene_break}\n\n{read(name)}"
        assert check_input(text, lang).ok, (name, scene_break)


def test_fragmented_lines_reported_once() -> None:
    text = "ab\ncd\nef\n" + "\n".join(f"this is long line number {n}" for n in range(8))
    report = check_input(text, Language.ENGLISH)
    frag = [i for i in report.issues if i.kind is IssueKind.FRAGMENTED_LINES]
    assert len(frag) == 1 and frag[0].line == 1


def test_require_clean_caps_message_at_ten_issues() -> None:
    text = "\n".join(f"line {n} aaaaa" for n in range(15))
    with pytest.raises(CorruptInputError) as exc:
        require_clean(text, Language.ENGLISH)
    assert str(exc.value).count("line ") >= 10
    assert str(exc.value).count("STRETCHED_LETTERS") == 10


def test_quality_source_is_ascii() -> None:
    import narratoflow.text.quality as module

    assert module.__file__ is not None
    assert Path(module.__file__).read_text(encoding="utf-8").isascii()


def test_short_clean_script_with_one_short_line_passes() -> None:
    text = "\n".join(
        [
            "\u0628\u062f\u0623\u062a \u0627\u0644\u0642\u0635\u0629 \u0647\u0646\u0627",
            "\u0646\u0639\u0645",
            "\u062b\u0645 \u0627\u0646\u062a\u0647\u062a \u0627\u0644\u0642\u0635\u0629",
            "\u0648\u0634\u0643\u0631\u0627 \u0644\u0643\u0645 \u062c\u0645\u064a\u0639\u0627",
        ]
    )
    assert check_input(text, Language.ARABIC).ok
