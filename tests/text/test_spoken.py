from pathlib import Path

import pytest

from narratoflow.channels.config import load_channel
from narratoflow.models import Language
from narratoflow.text.spoken import Lexicon, SpokenText, channel_lexicon, spoken_text

EN = Language.ENGLISH
AR = Language.ARABIC


def test_plain_text_unchanged() -> None:
    assert spoken_text("hello world.", EN, Lexicon.EMPTY) == SpokenText("hello world.", (1, 1))


def test_number_token_expands_with_punctuation_kept() -> None:
    result = spoken_text("In 1453, Constantinople fell.", EN, Lexicon.EMPTY)
    assert result.text == "In fourteen fifty-three, Constantinople fell."
    assert result.groups == (1, 2, 1, 1)


def test_wrapping_punctuation_and_symbols_stay_on_edges() -> None:
    result = spoken_text('("22nd")', EN, Lexicon.EMPTY)
    assert result == SpokenText('("twenty-second")', (1,))
    assert spoken_text("(1453)", EN, Lexicon.EMPTY) == SpokenText("(fourteen fifty-three)", (2,))


def test_arabic_number_uses_arabic_speller() -> None:
    result = spoken_text("\u0633\u0646\u0629 \u0661\u0664\u0665\u0663.", AR, Lexicon.EMPTY)
    assert result.text == (
        "\u0633\u0646\u0629 \u0623\u0644\u0641"
        " \u0648\u0623\u0631\u0628\u0639\u0645\u0627\u0626\u0629"
        " \u0648\u062b\u0644\u0627\u062b\u0629 \u0648\u062e\u0645\u0633\u0648\u0646."
    )
    assert result.groups == (1, 4)


def test_lexicon_beats_number() -> None:
    lexicon = Lexicon.from_entries({"1453": "the year of the conquest"})
    result = spoken_text("1453,", EN, lexicon)
    assert result == SpokenText("the year of the conquest,", (5,))


def test_lexicon_matches_ignoring_diacritics() -> None:
    lexicon = Lexicon.from_entries(
        {"\u0645\u062d\u0645\u062f": "\u0645\u064f\u062d\u064e\u0645\u0651\u064e\u062f"}
    )
    word = "\u0645\u064f\u062d\u064e\u0645\u0651\u064e\u062f\u064c"
    result = spoken_text(word, AR, lexicon)
    assert result.text == "\u0645\u064f\u062d\u064e\u0645\u0651\u064e\u062f"
    assert lexicon.lookup(word) == "\u0645\u064f\u062d\u064e\u0645\u0651\u064e\u062f"


def test_lexicon_lookup_misses_return_none() -> None:
    lexicon = Lexicon.from_entries({"NASA": "nasa"})
    assert lexicon.lookup("NASA,") == "nasa"
    assert lexicon.lookup("Nasa,") is None
    assert lexicon.lookup("other") is None
    assert lexicon.lookup("...") is None
    assert Lexicon.EMPTY.lookup("x") is None


def test_mixed_token_left_alone() -> None:
    assert spoken_text("COVID-19", EN, Lexicon.EMPTY) == SpokenText("COVID-19", (1,))


def test_mixed_token_forced_by_lexicon() -> None:
    lexicon = Lexicon.from_entries({"COVID-19": "covid nineteen"})
    assert spoken_text("COVID-19.", EN, lexicon) == SpokenText("covid nineteen.", (2,))


def test_blank_and_punctuation_only_text() -> None:
    assert spoken_text("", EN, Lexicon.EMPTY) == SpokenText("", ())
    assert spoken_text("... 5", EN, Lexicon.EMPTY) == SpokenText("... five", (1, 1))


def test_groups_invariant() -> None:
    with pytest.raises(ValueError, match="groups"):
        SpokenText("a b", (1,))
    with pytest.raises(ValueError, match="groups"):
        SpokenText("a", (0, 1))


def test_lexicon_rejects_bad_entries() -> None:
    with pytest.raises(ValueError, match="bad key") as space:
        Lexicon.from_entries({"bad key": "SECRETREPLACEMENT"})
    assert "SECRETREPLACEMENT" not in str(space.value)
    with pytest.raises(ValueError, match="empty"):
        Lexicon.from_entries({"  ": "x"})
    with pytest.raises(ValueError, match="NASA") as blank:
        Lexicon.from_entries({"NASA": "   "})
    assert "   " not in str(blank.value).replace("NASA", "")
    with pytest.raises(ValueError, match="duplicate") as dup:
        Lexicon.from_entries({"nasa": "SECRETONE", "nasa.": "SECRETTWO"})
    assert "SECRET" not in str(dup.value)
    with pytest.raises(ValueError, match="empty"):
        Lexicon.from_entries({"--": "x"})


def test_channel_lexicon(channel_file: Path) -> None:
    text = channel_file.read_text(encoding="utf-8")
    channel_file.write_text(text + '\n[pronunciation]\n"NASA" = "nasa"\n', encoding="utf-8")
    assert channel_lexicon(load_channel(channel_file)).lookup("NASA") == "nasa"


def test_format_characters_at_token_edges_are_edges() -> None:
    rlm = "\u200f"
    arabic = rlm + "\u0661\u0664\u0665\u0663" + rlm
    result = spoken_text(arabic, AR, Lexicon.EMPTY)
    assert result.text.startswith(rlm + "\u0623\u0644\u0641 ")
    assert result.text.endswith("\u0648\u062e\u0645\u0633\u0648\u0646" + rlm)
    assert result.groups == (4,)
    assert spoken_text("5\u200b", EN, Lexicon.EMPTY) == SpokenText("five\u200b", (1,))


def test_lexicon_is_hashable() -> None:
    a = Lexicon.from_entries({"NASA": "nasa"})
    assert hash(a) == hash(Lexicon.from_entries({"NASA": "nasa"}))
    assert {a, Lexicon.EMPTY, Lexicon.EMPTY} == {a, Lexicon.EMPTY}


def test_lexicon_rejects_symbol_keys_but_allows_punctuation() -> None:
    with pytest.raises(ValueError, match=r"C\+\+") as exc:
        Lexicon.from_entries({"C++": "SECRETPHRASE"})
    assert "SECRETPHRASE" not in str(exc.value)
    assert Lexicon.from_entries({"COVID-19": "covid nineteen"}).lookup("COVID-19") is not None


def test_unchanged_text_is_identical() -> None:
    raw = "The city fell that spring. \n\n"
    result = spoken_text(raw, EN, Lexicon.EMPTY)
    assert result.text == raw
    assert result.groups == (1,) * 5
    changed = spoken_text("In  1453\n\nit fell. ", EN, Lexicon.EMPTY)
    assert changed.text == "In  fourteen fifty-three\n\nit fell. "
    assert changed.groups == (1, 2, 1, 1)


def test_acronym_key_does_not_rewrite_common_word() -> None:
    lexicon = Lexicon.from_entries({"U.S.": "you ess"})
    result = spoken_text("Tell us about the U.S.", EN, lexicon)
    assert result.text == "Tell us about the you ess."


def test_uppercase_key_is_case_sensitive() -> None:
    lexicon = Lexicon.from_entries({"WHO": "double you aitch oh"})
    assert spoken_text("who knew", EN, lexicon).text == "who knew"
    assert spoken_text("(WHO) knew", EN, lexicon).text == "(double you aitch oh) knew"


def test_lowercase_key_stays_loose() -> None:
    lexicon = Lexicon.from_entries({"nasa": "naa saa"})
    assert lexicon.lookup("NASA") == "naa saa"
    assert lexicon.lookup("Nasa.") == "naa saa"


def test_arabic_key_stays_loose() -> None:
    lexicon = Lexicon.from_entries({"\u0645\u062d\u0645\u062f": "x"})
    assert lexicon.lookup("\u0645\u064f\u062d\u064e\u0645\u0651\u064e\u062f\u060c") == "x"


def test_exact_key_ignores_tashkeel_in_token() -> None:
    lexicon = Lexicon.from_entries({"Cafe": "kaf"})
    assert lexicon.lookup("Cafe\u0301") == "kaf"


def test_loose_and_exact_keys_can_coexist_and_exact_wins() -> None:
    lexicon = Lexicon.from_entries({"nasa": "loose", "NASA": "exact"})
    assert lexicon.lookup("NASA") == "exact"
    assert lexicon.lookup("Nasa") == "loose"


def test_exact_duplicates_rejected_without_replacement() -> None:
    with pytest.raises(ValueError, match="duplicate") as dup:
        Lexicon.from_entries({"U.S.": "SECRETONE", "U.S": "SECRETTWO"})
    assert "SECRET" not in str(dup.value)
