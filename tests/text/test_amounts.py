import pytest

from narratoflow.models import Language
from narratoflow.text.amounts import spell_amount

EN = Language.ENGLISH
AR = Language.ARABIC


@pytest.mark.parametrize(
    ("prefix", "core", "suffix", "expected"),
    [
        ("$", "5", "", ("", "five dollars", "")),
        ("$", "1", ",", ("", "one dollar", ",")),
        ("$", "1.50", "", ("", "one dollar and fifty cents", "")),
        ("$", "0.99", "", ("", "ninety-nine cents", "")),
        ("$", "3.00", "", ("", "three dollars", "")),
        ("$", "1.5", "", ("", "one point five dollars", "")),
        ("\u00a3", "2.01", "", ("", "two pounds and one penny", "")),
        ("\u00a3", "2.05", "", ("", "two pounds and five pence", "")),
        ("\u20ac", "1453", "", ("", "one thousand four hundred and fifty-three euros", "")),
        ("", "50", "%.", ("", "fifty percent", ".")),
        ("(-", "5", ")", ("(", "minus five", ")")),
        ("-$", "5", "", ("", "minus five dollars", "")),
        ("$-", "5", "", ("", "minus five dollars", "")),
        ("\u2212", "5", "", ("", "minus five", "")),
        ("+", "3", "", ("", "plus three", "")),
        ("", "5", "$", ("", "five dollars", "")),
        ("-", "1999", "%", ("", "minus one thousand nine hundred and ninety-nine percent", "")),
    ],
)
def test_english(prefix: str, core: str, suffix: str, expected: tuple[str, str, str]) -> None:
    assert spell_amount(prefix, core, suffix, EN) == expected


@pytest.mark.parametrize(
    ("prefix", "core", "suffix", "expected"),
    [
        ("$", "1", "", "\u062f\u0648\u0644\u0627\u0631 \u0648\u0627\u062d\u062f"),
        ("$", "2", "", "\u062f\u0648\u0644\u0627\u0631\u0627\u0646"),
        ("$", "5", "", "\u062e\u0645\u0633\u0629 \u062f\u0648\u0644\u0627\u0631\u0627\u062a"),
        (
            "$",
            "1005",
            "",
            "\u0623\u0644\u0641 "
            "\u0648\u062e\u0645\u0633\u0629 \u062f\u0648\u0644\u0627\u0631\u0627\u062a",
        ),
        ("$", "20", "", "\u0639\u0634\u0631\u0648\u0646 \u062f\u0648\u0644\u0627\u0631\u0627"),
        ("$", "100", "", "\u0645\u0627\u0626\u0629 \u062f\u0648\u0644\u0627\u0631"),
        (
            "$",
            "2.5",
            "",
            "\u0627\u062b\u0646\u0627\u0646 \u0641\u0627\u0635\u0644\u0629 "
            "\u062e\u0645\u0633\u0629 \u062f\u0648\u0644\u0627\u0631",
        ),
        ("\u20ac", "5", "", "\u062e\u0645\u0633\u0629 \u064a\u0648\u0631\u0648\u0647\u0627\u062a"),
        ("\u00a3", "2", "", "\u062c\u0646\u064a\u0647\u0627\u0646"),
        (
            "",
            "\u0665\u0660",
            "\u066a",
            "\u062e\u0645\u0633\u0648\u0646 \u0628\u0627\u0644\u0645\u0627\u0626\u0629",
        ),
        ("-", "\u0665", "", "\u0633\u0627\u0644\u0628 \u062e\u0645\u0633\u0629"),
    ],
)
def test_arabic(prefix: str, core: str, suffix: str, expected: str) -> None:
    assert spell_amount(prefix, core, suffix, AR) == ("", expected, "")


@pytest.mark.parametrize(
    ("prefix", "core", "suffix"),
    [
        ("$", "5", "%"),
        ("$\u20ac", "5", ""),
        ("$", "5", "$"),
        ("", "5", ""),
        ("$", "abc", ""),
        ("- ", "5", ""),
        ("-", "5th", ""),
    ],
)
def test_not_an_amount(prefix: str, core: str, suffix: str) -> None:
    assert spell_amount(prefix, core, suffix, EN) is None
