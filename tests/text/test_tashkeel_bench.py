import hashlib
import re
from pathlib import Path

import pytest

from narratoflow.errors import ProviderError
from narratoflow.text.tashkeel import bench as B
from narratoflow.text.tashkeel.verify import SkeletonMismatchError, strip_diacritics

KAF, TA, BA, WAW, LAM, DAL, ALEF, MEEM = (
    chr(0x0643),
    chr(0x062A),
    chr(0x0628),
    chr(0x0648),
    chr(0x0644),
    chr(0x062F),
    chr(0x0627),
    chr(0x0645),
)
FATHA, DAMMA, KASRA, SUKUN, SHADDA = chr(0x064E), chr(0x064F), chr(0x0650), chr(0x0652), chr(0x0651)
TATWEEL, SUPER_ALEF = chr(0x0640), chr(0x0670)

# kataba: 3 letters, all marked. alwaladu: 5 letters, alef and first lam bare.
WORD1 = KAF + FATHA + TA + FATHA + BA + FATHA
WORD2 = ALEF + LAM + WAW + FATHA + LAM + FATHA + DAL + DAMMA
GOLD = f"{WORD1} {WORD2}"


def test_perfect_prediction_zero_error() -> None:
    score = B.score_line(GOLD, GOLD)
    assert score.rejected is False
    assert score.with_case == B.Counts(letters=8, letter_errors=0, words=2, word_errors=0)
    assert score.without_case == B.Counts(letters=6, letter_errors=0, words=2, word_errors=0)


def test_undiacritized_prediction() -> None:
    score = B.score_line(GOLD, strip_diacritics(GOLD))
    assert score.with_case == B.Counts(letters=8, letter_errors=6, words=2, word_errors=2)
    assert score.without_case == B.Counts(letters=6, letter_errors=4, words=2, word_errors=2)


def test_bare_gold_letter_predicted_bare_is_correct() -> None:
    # only the alef and the first lam of WORD2 are bare in gold: a bare prediction is right there
    pred = f"{WORD1} {ALEF}{LAM}{WAW}{LAM}{DAL}"
    assert B.score_line(GOLD, pred).with_case.letter_errors == 3  # waw, lam2 (fatha), dal (damma)


def test_extra_mark_on_bare_gold_letter_is_an_error() -> None:
    pred = f"{WORD1} {ALEF}{FATHA}{LAM}{WAW}{FATHA}{LAM}{FATHA}{DAL}{DAMMA}"
    score = B.score_line(GOLD, pred)
    assert score.with_case.letter_errors == 1
    assert score.with_case.word_errors == 1


def test_case_ending_variant_ignores_last_letter() -> None:
    pred = f"{KAF}{FATHA}{TA}{FATHA}{BA}{DAMMA} {ALEF}{LAM}{WAW}{FATHA}{LAM}{FATHA}{DAL}{KASRA}"
    score = B.score_line(GOLD, pred)
    assert score.without_case.letter_errors == 0
    assert score.without_case.word_errors == 0
    assert score.with_case.letter_errors == score.with_case.word_errors == 2


def test_one_letter_word_without_case_ending_is_always_correct() -> None:
    gold = f"{WAW}{FATHA} {WORD1}"
    score = B.score_line(gold, strip_diacritics(gold))
    assert score.without_case.words == 2
    assert score.without_case.letters == 2  # the one-letter word scores nothing
    assert score.without_case.word_errors == 1
    assert score.with_case.word_errors == 2


def test_non_arabic_excluded() -> None:
    noisy = f"2024 abc, {GOLD}! (x)"
    assert B.score_line(noisy, noisy) == B.score_line(GOLD, GOLD)
    assert B.score_line(noisy, strip_diacritics(noisy)) == B.score_line(
        GOLD, strip_diacritics(GOLD)
    )


def test_line_without_arabic_letters_scores_nothing() -> None:
    score = B.score_line("2024 abc", "2024 abc")
    assert score.with_case == B.Counts(0, 0, 0, 0)
    assert score.rejected is False


def test_shadda_order_insensitive() -> None:
    gold = f"{DAL}{SHADDA}{FATHA}{BA}"
    pred = f"{DAL}{FATHA}{SHADDA}{BA}"
    assert B.score_line(gold, pred).with_case.letter_errors == 0
    assert B.score_line(gold, f"{DAL}{FATHA}{BA}").with_case.letter_errors == 1


def test_superscript_alef_and_tatweel_stripped() -> None:
    gold = f"{DAL}{FATHA}{TATWEEL}{BA}{SUPER_ALEF}{KASRA}"
    pred = f"{DAL}{FATHA}{BA}{KASRA}"
    assert B.score_line(gold, pred).with_case.letter_errors == 0


def test_other_marks_in_the_arabic_mark_block_are_ignored() -> None:
    maddah = chr(0x0653)
    gold = f"{DAL}{maddah}{FATHA}{BA}"
    assert B.score_line(gold, f"{DAL}{FATHA}{BA}").with_case.letter_errors == 0


def test_changed_base_letters_rejected() -> None:
    score = B.score_line(GOLD, f"{KAF}{FATHA}{TA}{FATHA}{MEEM} {WORD2}")
    assert score.rejected is True
    assert score.with_case == B.Counts(letters=8, letter_errors=8, words=2, word_errors=2)
    assert score.without_case == B.Counts(letters=6, letter_errors=6, words=2, word_errors=2)


def test_respacing_that_keeps_base_letters_is_not_rejected() -> None:
    pred = f"{WORD1}{WORD2}"
    score = B.score_line(GOLD, pred)
    assert score.rejected is False
    assert score.with_case.letter_errors == 0


def test_run_engine_scores_and_times() -> None:
    ticks = iter([10.0, 12.5])
    result = B.run_engine("bare", lambda text: text, [GOLD], clock=lambda: next(ticks))
    # the two bare gold letters are right even unmarked: 6 of 8 letters, 4 of 6 without endings
    assert result.der == pytest.approx(75.0)
    assert result.der_no_case == pytest.approx(4 / 6 * 100)
    assert (result.wer, result.wer_no_case) == (100.0, 100.0)
    assert (result.lines, result.rejected_lines, result.seconds) == (1, 0, 2.5)


def test_run_engine_input_is_stripped_gold() -> None:
    seen: list[str] = []

    def engine(text: str) -> str:
        seen.append(text)
        return GOLD

    result = B.run_engine("oracle", engine, [GOLD])
    assert seen == [strip_diacritics(GOLD)]
    assert (result.der, result.wer, result.rejected_lines) == (0.0, 0.0, 0)


def test_run_engine_aggregates_over_lines() -> None:
    perfect_then_bare = iter([GOLD, strip_diacritics(GOLD)])
    result = B.run_engine("mixed", lambda _t: next(perfect_then_bare), [GOLD, GOLD])
    assert result.der == pytest.approx(6 / 16 * 100)
    assert result.der_no_case == pytest.approx(4 / 12 * 100)
    assert result.wer == pytest.approx(50.0)


def test_run_engine_catches_skeleton_mismatch() -> None:
    def engine(_text: str) -> str:
        raise SkeletonMismatchError("changed")

    result = B.run_engine("bad", engine, [GOLD, GOLD])
    assert result.rejected_lines == 2
    assert result.der == 100.0
    assert result.wer == 100.0


def test_run_engine_catches_provider_error() -> None:
    def engine(_text: str) -> str:
        raise ProviderError("down")

    result = B.run_engine("down", engine, [GOLD])
    assert result.rejected_lines == 1
    assert result.der_no_case == 100.0


def test_run_engine_counts_changed_base_output_as_rejected() -> None:
    result = B.run_engine("wrong", lambda _t: f"{MEEM}{MEEM}", [GOLD])
    assert result.rejected_lines == 1
    assert result.der == 100.0


def test_run_engine_on_no_scored_letters_is_zero() -> None:
    result = B.run_engine("none", lambda t: t, ["2024"])
    assert (result.der, result.wer, result.der_no_case, result.wer_no_case) == (0, 0, 0, 0)


def test_not_run_row_has_reason() -> None:
    row = B.not_run("llm", "not run (paid)")
    assert row.engine == "llm"
    assert row.not_run == "not run (paid)"
    assert row.lines == 0


def test_results_markdown_lists_not_run_reason() -> None:
    ran = B.EngineResult("identity", 300, 2, 61.25, 70.0, 80.5, 90.0, 1.5)
    md = B.results_markdown([ran, B.not_run("catt", "needs onnxruntime-gpu")])
    lines = md.splitlines()
    assert lines[0].startswith("| engine | lines | rejected | DER |")
    assert "| identity | 300 | 2 | 61.25 | 70.00 | 80.50 | 90.00 | 1.5 s |" in lines
    assert "| catt | not run | not run | not run | not run | not run | not run | - |" in lines
    assert "catt: needs onnxruntime-gpu" in md


def _manifest_for(text: str) -> B.GoldManifest:
    return B.make_manifest(
        text, source_url="https://example.test/x", commit="abc", licence_notes="n"
    )


def _sha(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()


def test_make_manifest_rule() -> None:
    rows = [
        "one two three four",  # 1: too short
        "a b c d e",  # 2: 5 tokens, in
        "a b c d e f g",  # 3: in
        "",  # 4: empty
        " ".join(["w"] * 61),  # 5: too long
        " ".join(["w"] * 60),  # 6: 60 tokens, in
        "a b c d e f g h",  # 7: in
        "a b c d e f g h i",  # 8: count reached before this
    ]
    manifest = B.make_manifest(
        "\n".join(rows),
        source_url="https://example.test/x",
        commit="c0ffee",
        licence_notes="MIT repo, GPLv2 text",
        count=3,
    )
    assert [n for n, _ in manifest.lines] == [2, 3, 6]
    assert manifest.lines[0] == (2, _sha(rows[1]))
    assert (manifest.source_url, manifest.commit) == ("https://example.test/x", "c0ffee")
    assert manifest.file_sha256 == _sha("\n".join(rows))
    assert "5 to 60" in manifest.rule


def test_manifest_roundtrip_and_ascii(tmp_path_factory: pytest.TempPathFactory) -> None:
    text = "\n".join(f"w{i} a b c d e" for i in range(10))
    manifest = _manifest_for(text)
    path = tmp_path_factory.mktemp("m") / "m.toml"
    path.write_text(B.manifest_toml(manifest), encoding="utf-8")
    assert path.read_text(encoding="utf-8").isascii()
    assert B.load_manifest(path) == manifest


def test_select_gold_returns_lines_in_order() -> None:
    text = "\n".join(f"w{i} a b c d e" for i in range(10))
    manifest = _manifest_for(text)
    assert B.select_gold(text, manifest) == text.split("\n")


def test_select_gold_checks_hashes() -> None:
    rows = [f"w{i} a b c d e" for i in range(10)]
    manifest = _manifest_for("\n".join(rows))
    rows[6] = "tampered a b c d e"
    with pytest.raises(ValueError, match=r"gold line 7: sha256 mismatch"):
        B.select_gold("\n".join(rows), manifest)


def test_select_gold_missing_line() -> None:
    manifest = _manifest_for("\n".join(f"w{i} a b c d e" for i in range(10)))
    with pytest.raises(ValueError, match=r"gold line 1: sha256 mismatch"):
        B.select_gold("", manifest)


def test_mark_after_a_non_letter_is_ignored() -> None:
    gold = f"1{FATHA}{BA}{FATHA}"
    assert B.score_line(gold, f"1{BA}{FATHA}").with_case.letter_errors == 0


def test_committed_manifest_is_well_formed() -> None:
    manifest = B.load_manifest(Path(__file__).parents[2] / "docs/benchmarks/tashkeel-gold.toml")
    numbers = [n for n, _ in manifest.lines]
    assert len(numbers) == 300
    assert numbers == sorted(set(numbers))
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for _, digest in manifest.lines)
    assert manifest.file_sha256 == (
        "4e851ff836f0a178abb15d9f4a8bcf92748b77cc0d67030d9fae2fbe698baa12"
    )
    assert manifest.commit == "e083579c03e489936200b69d792fd426bb576743"
    assert "MIT" in manifest.licence_notes
    assert "GPLv2" in manifest.licence_notes
