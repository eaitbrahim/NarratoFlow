import pytest

from narratoflow.models import Language
from narratoflow.text.numbers import spell_number

EN = Language.ENGLISH
AR = Language.ARABIC


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("0", "zero"),
        ("7", "seven"),
        ("13", "thirteen"),
        ("21", "twenty-one"),
        ("100", "one hundred"),
        ("101", "one hundred and one"),
        ("2005", "two thousand and five"),
        ("1,000,000", "one million"),
        (
            "999999999999",
            "nine hundred and ninety-nine billion nine hundred and ninety-nine million"
            " nine hundred and ninety-nine thousand nine hundred and ninety-nine",
        ),
        ("1453", "fourteen fifty-three"),
        ("1900", "nineteen hundred"),
        ("1905", "nineteen oh five"),
        ("1000", "one thousand"),
        ("2000", "two thousand"),
        ("1,453", "one thousand four hundred and fifty-three"),
        ("3.14", "three point one four"),
        ("1st", "first"),
        ("22nd", "twenty-second"),
        ("19th", "nineteenth"),
        ("100th", "one hundredth"),
        ("20th", "twentieth"),
        ("12th", "twelfth"),
        ("101st", "one hundred and first"),
        ("1453rd", "one thousand four hundred and fifty-third"),
    ],
)
def test_english(token: str, expected: str) -> None:
    assert spell_number(token, EN) == expected


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("0", "\u0635\u0641\u0631"),
        ("1", "\u0648\u0627\u062d\u062f"),
        ("2", "\u0627\u062b\u0646\u0627\u0646"),
        ("11", "\u0623\u062d\u062f \u0639\u0634\u0631"),
        ("12", "\u0627\u062b\u0646\u0627 \u0639\u0634\u0631"),
        ("21", "\u0648\u0627\u062d\u062f \u0648\u0639\u0634\u0631\u0648\u0646"),
        ("100", "\u0645\u0627\u0626\u0629"),
        ("200", "\u0645\u0627\u0626\u062a\u0627\u0646"),
        ("300", "\u062b\u0644\u0627\u062b\u0645\u0627\u0626\u0629"),
        ("1000", "\u0623\u0644\u0641"),
        ("2000", "\u0623\u0644\u0641\u0627\u0646"),
        ("3000", "\u062b\u0644\u0627\u062b\u0629 \u0622\u0644\u0627\u0641"),
        ("10000", "\u0639\u0634\u0631\u0629 \u0622\u0644\u0627\u0641"),
        ("11000", "\u0623\u062d\u062f \u0639\u0634\u0631 \u0623\u0644\u0641\u0627"),
        ("100000", "\u0645\u0627\u0626\u0629 \u0623\u0644\u0641"),
        (
            "1453",
            "\u0623\u0644\u0641 \u0648\u0623\u0631\u0628\u0639\u0645\u0627\u0626\u0629"
            " \u0648\u062b\u0644\u0627\u062b\u0629 \u0648\u062e\u0645\u0633\u0648\u0646",
        ),
        ("20", "\u0639\u0634\u0631\u0648\u0646"),
        ("90", "\u062a\u0633\u0639\u0648\u0646"),
        ("50000", "\u062e\u0645\u0633\u0648\u0646 \u0623\u0644\u0641\u0627"),
        ("1000000", "\u0645\u0644\u064a\u0648\u0646"),
        ("2000000", "\u0645\u0644\u064a\u0648\u0646\u0627\u0646"),
        ("5000000", "\u062e\u0645\u0633\u0629 \u0645\u0644\u0627\u064a\u064a\u0646"),
        ("11000000", "\u0623\u062d\u062f \u0639\u0634\u0631 \u0645\u0644\u064a\u0648\u0646\u0627"),
        ("1000000000", "\u0645\u0644\u064a\u0627\u0631"),
        ("2000000000", "\u0645\u0644\u064a\u0627\u0631\u0627\u0646"),
        ("3000000000", "\u062b\u0644\u0627\u062b\u0629 \u0645\u0644\u064a\u0627\u0631\u0627\u062a"),
        (
            "11000000000",
            "\u0623\u062d\u062f \u0639\u0634\u0631 \u0645\u0644\u064a\u0627\u0631\u0627",
        ),
        (
            "\u0663\u066b\u0665",
            (
                "\u062b\u0644\u0627\u062b\u0629 \u0641\u0627\u0635\u0644\u0629"
                " \u062e\u0645\u0633\u0629"
            ),
        ),
    ],
)
def test_arabic(token: str, expected: str) -> None:
    assert spell_number(token, AR) == expected


@pytest.mark.parametrize("token", ["\u0661\u0664\u0665\u0663", "\u06f1\u06f4\u06f5\u06f3"])
def test_arabic_indic_and_persian_digits_equal_ascii(token: str) -> None:
    assert spell_number(token, AR) == spell_number("1453", AR)
    assert spell_number(token, AR) is not None


def test_english_reads_arabic_indic_digits() -> None:
    assert spell_number("\u0661\u0664\u0665\u0663", EN) == "fourteen fifty-three"


def test_arabic_thousands_separator() -> None:
    assert spell_number("1\u066c000", AR) == spell_number("1000", AR)


def test_ordinals_are_english_only() -> None:
    assert spell_number("1st", AR) is None


@pytest.mark.parametrize(
    "token",
    ["1234567890123", "12a", "1,23", "abc", "", "1.2.3", "1th", "11st", "3.", ".5", "1,2345"],
)
@pytest.mark.parametrize("language", [EN, AR])
def test_rejected(token: str, language: Language) -> None:
    assert spell_number(token, language) is None


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("1099", "one thousand and ninety-nine"),
        ("1100", "eleven hundred"),
        ("1999", "nineteen ninety-nine"),
        ("2000", "two thousand"),
    ],
)
def test_english_year_boundaries(token: str, expected: str) -> None:
    assert spell_number(token, EN) == expected


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("101000", "\u0645\u0627\u0626\u0629 \u0623\u0644\u0641 \u0648\u0623\u0644\u0641"),
        (
            "102000",
            "\u0645\u0627\u0626\u0629 \u0623\u0644\u0641 \u0648\u0623\u0644\u0641\u0627\u0646",
        ),
        (
            "101000000",
            "\u0645\u0627\u0626\u0629 \u0645\u0644\u064a\u0648\u0646"
            " \u0648\u0645\u0644\u064a\u0648\u0646",
        ),
        (
            "202000000000",
            "\u0645\u0627\u0626\u062a\u0627 \u0645\u0644\u064a\u0627\u0631"
            " \u0648\u0645\u0644\u064a\u0627\u0631\u0627\u0646",
        ),
    ],
)
def test_arabic_count_ending_in_one_or_two_before_scale(token: str, expected: str) -> None:
    assert spell_number(token, AR) == expected


@pytest.mark.parametrize("token", ["1\u0664", "\u0661\u06f4", "1\u066b\u0665\u0664"])
def test_mixed_digit_scripts_rejected(token: str) -> None:
    assert spell_number(token, EN) is None
    assert spell_number(token, AR) is None


def test_years_false_reads_cardinal() -> None:
    assert spell_number("1453", EN, years=False) == "one thousand four hundred and fifty-three"
    assert spell_number("1453", EN) == "fourteen fifty-three"
