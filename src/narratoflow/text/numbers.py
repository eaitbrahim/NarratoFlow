"""Spell numbers out as words, in English and Arabic, for narration.

Only plain integers up to 999,999,999,999 (with an optional thousands separator
and decimal part, in any Unicode decimal digits) and English ordinals such as
`22nd` are handled; anything else is left for the voice to read as written.
Arabic uses the plain pausal form: masculine, no case endings, no tashkeel.
"""

from __future__ import annotations

import re
import unicodedata

from narratoflow.models import Language

MAX_NUMBER = 999_999_999_999

_THOUSANDS_SEPS = ",\u066c"
_DECIMAL_SEPS = ".\u066b"
_NUMBER = re.compile(
    rf"(?P<int>\d+|\d{{1,3}}(?:[{_THOUSANDS_SEPS}]\d{{3}})+)(?:[{_DECIMAL_SEPS}](?P<frac>\d+))?"
)
_ORDINAL = re.compile(r"(?P<int>\d+|\d{1,3}(?:[,\u066c]\d{3})+)(?P<suffix>st|nd|rd|th)", re.I)

_EN_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_EN_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_EN_SCALES = ((10**9, "billion"), (10**6, "million"), (1000, "thousand"))
_EN_ORDINAL_IRREGULAR = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}


def _digits(text: str) -> int:
    return int("".join(str(unicodedata.decimal(ch)) for ch in text))


def _en_below_100(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    tens, ones = divmod(n, 10)
    return _EN_TENS[tens] + (f"-{_EN_ONES[ones]}" if ones else "")


def _en_below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    if not hundreds:
        return _en_below_100(rest)
    head = f"{_EN_ONES[hundreds]} hundred"
    return f"{head} and {_en_below_100(rest)}" if rest else head


def _en_cardinal(n: int) -> str:
    if n == 0:
        return "zero"
    parts: list[str] = []
    for size, name in _EN_SCALES:
        count, n = divmod(n, size)
        if count:
            parts.append(f"{_en_below_1000(count)} {name}")
    if n:
        tail = _en_below_1000(n)
        parts.append(f"and {tail}" if parts and n < 100 else tail)
    return " ".join(parts)


def _en_year(n: int) -> str:
    high, low = divmod(n, 100)
    if low == 0:
        return f"{_en_below_100(high)} hundred"
    if low < 10:
        return f"{_en_below_100(high)} oh {_EN_ONES[low]}"
    return f"{_en_below_100(high)} {_en_below_100(low)}"


def _en_ordinal(n: int) -> str:
    cardinal = _en_cardinal(n)
    head, sep, last = cardinal.rpartition(" ")
    prefix, hyphen, word = last.rpartition("-")
    if word in _EN_ORDINAL_IRREGULAR:
        word = _EN_ORDINAL_IRREGULAR[word]
    elif word.endswith("y"):
        word = word[:-1] + "ieth"
    else:
        word += "th"
    return head + sep + prefix + hyphen + word


def _ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


_AR_AND = " \u0648"
_AR_ONES = [
    "\u0635\u0641\u0631",
    "\u0648\u0627\u062d\u062f",
    "\u0627\u062b\u0646\u0627\u0646",
    "\u062b\u0644\u0627\u062b\u0629",
    "\u0623\u0631\u0628\u0639\u0629",
    "\u062e\u0645\u0633\u0629",
    "\u0633\u062a\u0629",
    "\u0633\u0628\u0639\u0629",
    "\u062b\u0645\u0627\u0646\u064a\u0629",
    "\u062a\u0633\u0639\u0629",
    "\u0639\u0634\u0631\u0629",
]
_AR_TEENS = {
    11: "\u0623\u062d\u062f \u0639\u0634\u0631",
    12: "\u0627\u062b\u0646\u0627 \u0639\u0634\u0631",
}
_AR_TENS = [
    "_",
    "_",
    "\u0639\u0634\u0631\u0648\u0646",
    "\u062b\u0644\u0627\u062b\u0648\u0646",
    "\u0623\u0631\u0628\u0639\u0648\u0646",
    "\u062e\u0645\u0633\u0648\u0646",
    "\u0633\u062a\u0648\u0646",
    "\u0633\u0628\u0639\u0648\u0646",
    "\u062b\u0645\u0627\u0646\u0648\u0646",
    "\u062a\u0633\u0639\u0648\u0646",
]
_AR_HUNDREDS = [
    "_",
    "\u0645\u0627\u0626\u0629",
    "\u0645\u0627\u0626\u062a\u0627\u0646",
    "\u062b\u0644\u0627\u062b\u0645\u0627\u0626\u0629",
    "\u0623\u0631\u0628\u0639\u0645\u0627\u0626\u0629",
    "\u062e\u0645\u0633\u0645\u0627\u0626\u0629",
    "\u0633\u062a\u0645\u0627\u0626\u0629",
    "\u0633\u0628\u0639\u0645\u0627\u0626\u0629",
    "\u062b\u0645\u0627\u0646\u0645\u0627\u0626\u0629",
    "\u062a\u0633\u0639\u0645\u0627\u0626\u0629",
]
_AR_HUNDREDS_CONSTRUCT_200 = "\u0645\u0627\u0626\u062a\u0627"
_AR_COMMA = "\u0641\u0627\u0635\u0644\u0629"
# (size, singular, dual, plural for 3..10, singular accusative for 11 and more)
_AR_SCALES = (
    (10**9, "\u0645\u0644\u064a\u0627\u0631", "\u0645\u0644\u064a\u0627\u0631\u0627\u0646",
     "\u0645\u0644\u064a\u0627\u0631\u0627\u062a", "\u0645\u0644\u064a\u0627\u0631\u0627"),
    (10**6, "\u0645\u0644\u064a\u0648\u0646", "\u0645\u0644\u064a\u0648\u0646\u0627\u0646",
     "\u0645\u0644\u0627\u064a\u064a\u0646", "\u0645\u0644\u064a\u0648\u0646\u0627"),
    (1000, "\u0623\u0644\u0641", "\u0623\u0644\u0641\u0627\u0646",
     "\u0622\u0644\u0627\u0641", "\u0623\u0644\u0641\u0627"),
)  # fmt: skip


def _ar_below_100(n: int) -> str:
    if n <= 10:
        return _AR_ONES[n]
    if n < 20:
        return _AR_TEENS.get(n) or f"{_AR_ONES[n - 10]} \u0639\u0634\u0631"
    tens, ones = divmod(n, 10)
    if not ones:
        return _AR_TENS[tens]
    return f"{_AR_ONES[ones]}{_AR_AND}{_AR_TENS[tens]}"


def _ar_below_1000(n: int, *, construct: bool = False) -> str:
    hundreds, rest = divmod(n, 100)
    parts: list[str] = []
    if hundreds:
        word = _AR_HUNDREDS[hundreds]
        if construct and hundreds == 2 and not rest:
            word = _AR_HUNDREDS_CONSTRUCT_200
        parts.append(word)
    if rest:
        parts.append(_ar_below_100(rest))
    return _AR_AND.join(parts)


def _ar_cardinal(n: int) -> str:
    if n == 0:
        return _AR_ONES[0]
    parts: list[str] = []
    for size, single, dual, plural, accusative in _AR_SCALES:
        count, n = divmod(n, size)
        if not count:
            continue
        if count == 1:
            parts.append(single)
        elif count == 2:
            parts.append(dual)
        else:
            tail = count % 100
            if count > 100 and tail in (1, 2):
                # MSA: "mi'a alf wa-alf", "mi'a alf wa-alfan"
                parts.append(f"{_ar_below_1000(count - tail, construct=True)} {single}")
                parts.append(single if tail == 1 else dual)
                continue
            noun = plural if 3 <= tail <= 10 else accusative if tail > 10 else single
            parts.append(f"{_ar_below_1000(count, construct=noun == single)} {noun}")
    if n:
        parts.append(_ar_below_1000(n))
    return _AR_AND.join(parts)


def spell_number(token: str, language: Language) -> str | None:
    """The number core of `token` spelled out, or None when it is not one we handle.

    `token` has no surrounding punctuation. Handles integers up to
    999,999,999,999 with optional thousands separators and decimal part, in any
    Unicode decimal digits, plus English ordinals (`22nd`). English reads a bare
    4-digit integer from 1100 to 1999 as a year.
    """
    if len({ord(c) - unicodedata.decimal(c) for c in token if c.isdecimal()}) > 1:
        return None  # mixed digit scripts
    english = language is Language.ENGLISH
    if english and (ordinal := _ORDINAL.fullmatch(token)):
        value = _digits(re.sub(f"[{_THOUSANDS_SEPS}]", "", ordinal["int"]))
        if value > MAX_NUMBER or ordinal["suffix"].lower() != _ordinal_suffix(value):
            return None
        return _en_ordinal(value)
    match = _NUMBER.fullmatch(token)
    if match is None:
        return None
    raw = match["int"]
    value = _digits(re.sub(f"[{_THOUSANDS_SEPS}]", "", raw))
    if value > MAX_NUMBER:
        return None
    if english:
        spoken = (
            _en_year(value)
            if match["frac"] is None and raw.isdigit() and (1100 <= value <= 1999 and len(raw) == 4)
            else _en_cardinal(value)
        )
    else:
        spoken = _ar_cardinal(value)
    frac = match["frac"]
    if frac is None:
        return spoken
    digit_words = [(_EN_ONES if english else _AR_ONES)[unicodedata.decimal(c)] for c in frac]
    point = "point" if english else _AR_COMMA
    return f"{spoken} {point} {' '.join(digit_words)}"
