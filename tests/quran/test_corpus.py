from pathlib import Path

import pytest
from conftest import FIXTURES

from narratoflow.errors import VerseCorpusError
from narratoflow.quran import corpus as corpus_module
from narratoflow.quran.corpus import QuranCorpus, default_corpus
from narratoflow.quran.normalize import match_normalize
from narratoflow.settings import Settings

SIMPLE = FIXTURES / "quran/quran-simple-clean.txt"
UTHMANI = FIXTURES / "quran/quran-uthmani.txt"


def test_normalize_spelling_variants() -> None:
    assert match_normalize("إِنَّ") == match_normalize("ان")
    assert match_normalize("يَحْيَىٰ") == match_normalize("يحيي")
    assert match_normalize("\ufd3fالحمد لله\ufd3e") == "الحمد لله"  # noqa: RUF001


def test_normalize_misc() -> None:
    assert match_normalize("ٱلْحَمْدُ") == match_normalize("الحمد")
    assert match_normalize("رحمة") == "رحمه"
    assert match_normalize("مُؤْمِن") == match_normalize("مءمن")
    assert match_normalize("كـتاب،  و\n  قال.") == "كتاب و قال"


def test_corpus_loads_fixture() -> None:
    c = QuranCorpus.from_files(SIMPLE, UTHMANI)
    a = c.ayah(31, 33)
    assert a.surah == 31
    assert "يوم" in a.simple
    assert a.norm_words == tuple(match_normalize(a.simple).split())
    assert len(list(c)) == 16


def test_text_range() -> None:
    c = QuranCorpus.from_files(SIMPLE, UTHMANI)
    joined = c.text_range(19, 1, 3)
    assert joined == " ".join(c.ayah(19, n).uthmani for n in (1, 2, 3))


def test_unknown_ayah() -> None:
    c = QuranCorpus.from_files(SIMPLE, UTHMANI)
    with pytest.raises(VerseCorpusError):
        c.ayah(2, 1)
    with pytest.raises(VerseCorpusError):
        c.text_range(19, 1, 99)


def test_corpus_key_mismatch(tmp_path: Path) -> None:
    lines = UTHMANI.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not ln.startswith("19|7|")]
    bad = tmp_path / "u.txt"
    bad.write_text("\n".join(kept), encoding="utf-8")
    with pytest.raises(VerseCorpusError):
        QuranCorpus.from_files(SIMPLE, bad)


def test_corpus_malformed_line(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("not-a-verse-line\n", encoding="utf-8")
    with pytest.raises(VerseCorpusError):
        QuranCorpus.from_files(bad, bad)
    with pytest.raises(VerseCorpusError):
        QuranCorpus.from_files(tmp_path / "missing.txt", bad)


def _settings(data_dir: Path) -> Settings:
    return Settings(data_dir=data_dir)


def test_default_corpus_from_data_dir(tmp_path: Path) -> None:
    q = tmp_path / "data" / "quran"
    q.mkdir(parents=True)
    (q / "quran-simple-clean.txt").write_bytes(SIMPLE.read_bytes())
    (q / "quran-uthmani.txt").write_bytes(UTHMANI.read_bytes())
    assert default_corpus(_settings(tmp_path / "data")).ayah(31, 33).surah == 31


def test_default_corpus_dev_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # fixture names differ from the fetched names, so stage copies
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "quran-simple-clean.txt").write_bytes(SIMPLE.read_bytes())
    (dev / "quran-uthmani.txt").write_bytes(UTHMANI.read_bytes())
    monkeypatch.setattr(corpus_module, "_dev_quran_dir", lambda: dev)
    assert default_corpus(_settings(tmp_path / "nodata")).ayah(19, 1).ayah == 1


def test_default_corpus_missing_explains_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus_module, "_dev_quran_dir", lambda: tmp_path / "nowhere")
    with pytest.raises(VerseCorpusError, match=r"fetch_quran\.py"):
        default_corpus(_settings(tmp_path / "nodata"))


def test_quran_sources_are_ascii() -> None:
    root = Path(corpus_module.__file__).parent
    for name in ("normalize.py", "corpus.py", "__init__.py"):
        (root / name).read_text(encoding="utf-8").encode("ascii")
