"""Read signs, currency symbols and percent signs that touch a number.

`spell_amount` takes the edges of one author token (as `spoken._split_edges`
splits them) and consumes only the sign, currency and percent characters that
touch the number core. Anything else stays in the returned prefix and suffix.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from narratoflow.models import Language
from narratoflow.text.numbers import _en_cardinal, spell_number

_SIGNS = {"-": "minus", "\u2212": "minus", "+": "plus"}
_AR_SIGNS = {"minus": "\u0633\u0627\u0644\u0628", "plus": "\u0632\u0627\u0626\u062f"}
_PERCENTS = "%\u066a"
_EN_PERCENT = "percent"
_AR_PERCENT = "\u0628\u0627\u0644\u0645\u0627\u0626\u0629"
_AR_ONE = "\u0648\u0627\u062d\u062f"
_PLAIN_NUMBER = re.compile(r"(?P<int>\d+|\d{1,3}(?:[,\u066c]\d{3})+)(?:[.\u066b](?P<frac>\d+))?")
_TWO_DECIMALS = re.compile(r"(?P<int>\d+|\d{1,3}(?:[,\u066c]\d{3})+)[.\u066b](?P<cents>\d{2})")


@dataclass(frozen=True)
class _Currency:
    en_one: str
    en_many: str
    en_minor: tuple[str, str]
    ar: tuple[str, str, str, str]  # singular, dual, plural, accusative


_CURRENCIES = {
    "$": _Currency(
        "dollar",
        "dollars",
        ("cent", "cents"),
        (
            "\u062f\u0648\u0644\u0627\u0631",
            "\u062f\u0648\u0644\u0627\u0631\u0627\u0646",
            "\u062f\u0648\u0644\u0627\u0631\u0627\u062a",
            "\u062f\u0648\u0644\u0627\u0631\u0627",
        ),
    ),
    "\u20ac": _Currency(
        "euro",
        "euros",
        ("cent", "cents"),
        (
            "\u064a\u0648\u0631\u0648",
            "\u064a\u0648\u0631\u0648\u0627\u0646",
            "\u064a\u0648\u0631\u0648\u0647\u0627\u062a",
            "\u064a\u0648\u0631\u0648",
        ),
    ),
    "\u00a3": _Currency(
        "pound",
        "pounds",
        ("penny", "pence"),
        (
            "\u062c\u0646\u064a\u0647",
            "\u062c\u0646\u064a\u0647\u0627\u0646",
            "\u062c\u0646\u064a\u0647\u0627\u062a",
            "\u062c\u0646\u064a\u0647\u0627",
        ),
    ),
}


def _int_value(raw: str) -> int:
    return int("".join(str(unicodedata.decimal(c)) for c in raw if c.isdecimal()))


def _arabic_unit(number: str, value: int, has_frac: bool, cur: _Currency, construct: str) -> str:
    single, dual, plural, accusative = cur.ar
    if has_frac:
        return f"{number} {single}"
    if value == 1:
        return f"{single} {_AR_ONE}"
    if value == 2:
        return dual
    tail = value % 100
    if 3 <= tail <= 10:
        return f"{number} {plural}"
    if tail > 10:
        return f"{number} {accusative}"
    return f"{construct} {single}"  # a trailing dual takes its construct form before the noun


def _english_unit(core: str, number: str, value: int, has_frac: bool, cur: _Currency) -> str:
    two = _TWO_DECIMALS.fullmatch(core)
    if two is None:
        return f"{number} {cur.en_one if value == 1 and not has_frac else cur.en_many}"
    cents = _int_value(two["cents"])
    major_text = _en_cardinal(value)
    cents_text = _en_cardinal(cents)
    major_phrase = f"{major_text} {cur.en_one if value == 1 else cur.en_many}"
    if cents == 0:
        return major_phrase
    cents_phrase = f"{cents_text} {cur.en_minor[0] if cents == 1 else cur.en_minor[1]}"
    return cents_phrase if value == 0 else f"{major_phrase} and {cents_phrase}"


def spell_amount(
    prefix: str, core: str, suffix: str, language: Language
) -> tuple[str, str, str] | None:
    """`(remaining_prefix, phrase, remaining_suffix)` or None.

    None when no sign, currency or percent touches `core`, when a token has two
    currencies, two signs, or both a currency and a percent, or when `core` is not a plain
    number the speller handles.
    """
    sign: str | None = None
    currency: str | None = None
    percent = False
    rest = prefix
    while rest:
        ch = rest[-1]
        if ch in _SIGNS and sign is None:
            sign = _SIGNS[ch]
        elif ch in _CURRENCIES and currency is None:
            currency = ch
        else:
            break
        rest = rest[:-1]
    tail = suffix
    if tail and tail[0] in _CURRENCIES and currency is None:
        currency = tail[0]
        tail = tail[1:]
    elif tail and tail[0] in _PERCENTS and currency is None:
        percent = True
        tail = tail[1:]
    if sign is None and currency is None and not percent:
        return None
    if (rest and (rest[-1] in _CURRENCIES or rest[-1] in _SIGNS)) or (
        tail and (tail[0] in _CURRENCIES or tail[0] in _PERCENTS)
    ):
        return None  # two currencies, or a currency with a percent
    match = _PLAIN_NUMBER.fullmatch(core)
    number = spell_number(core, language, years=False)
    if match is None or number is None:
        return None
    value = _int_value(match["int"])
    has_frac = match["frac"] is not None
    english = language is Language.ENGLISH
    construct = spell_number(core, language, years=False, construct_dual=True) or number
    body = number
    if currency is not None:
        cur = _CURRENCIES[currency]
        unit = (
            _english_unit(core, number, value, has_frac, cur)
            if english
            else _arabic_unit(number, value, has_frac, cur, construct)
        )
        body = unit
    elif percent:
        body = f"{number} {_EN_PERCENT if english else _AR_PERCENT}"
    if sign is not None:
        body = f"{sign if english else _AR_SIGNS[sign]} {body}"
    return rest, body, tail
