"""Tanzil corpus loader: verified verse text keyed by (surah, ayah)."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from narratoflow.errors import VerseCorpusError
from narratoflow.quran.normalize import match_normalize
from narratoflow.settings import Settings

SIMPLE_NAME = "quran-simple-clean.txt"
UTHMANI_NAME = "quran-uthmani.txt"


def _missing_hint(data_dir: Path) -> str:
    target = data_dir / "quran"
    return (
        f"put {SIMPLE_NAME} and {UTHMANI_NAME} (Tanzil Qur'an text) in {target}/; "
        f"from a source checkout, `uv run python scripts/fetch_quran.py --out {target}` "
        "downloads them"
    )


@dataclass(frozen=True)
class Ayah:
    surah: int
    ayah: int
    simple: str
    uthmani: str
    norm_words: tuple[str, ...]


def _is_number(field: str) -> bool:
    return field.isascii() and field.isdigit()


def _load(path: Path) -> dict[tuple[int, int], str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise VerseCorpusError(f"cannot read corpus file {path}: {e}") from e
    rows: dict[tuple[int, int], str] = {}
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3 or not all(_is_number(p) for p in parts[:2]):
            raise VerseCorpusError(f"{path}:{number}: malformed corpus line")
        key = (int(parts[0]), int(parts[1]))
        if key in rows:
            raise VerseCorpusError(f"{path}:{number}: duplicate ayah {key[0]}:{key[1]}")
        rows[key] = parts[2].strip()
    return rows


class QuranCorpus:
    def __init__(self, ayat: dict[tuple[int, int], Ayah]) -> None:
        self._ayat = ayat

    @classmethod
    def from_files(cls, simple: Path, uthmani: Path) -> "QuranCorpus":
        simple_rows = _load(simple)
        uthmani_rows = _load(uthmani)
        if simple_rows.keys() != uthmani_rows.keys():
            diff = sorted(simple_rows.keys() ^ uthmani_rows.keys())[:5]
            raise VerseCorpusError(f"corpus files disagree on surah/ayah keys, e.g. {diff}")
        ayat = {
            key: Ayah(
                surah=key[0],
                ayah=key[1],
                simple=simple_rows[key],
                uthmani=uthmani_rows[key],
                norm_words=tuple(match_normalize(simple_rows[key]).split()),
            )
            for key in sorted(simple_rows)
        }
        return cls(ayat)

    def ayah(self, surah: int, ayah: int) -> Ayah:
        try:
            return self._ayat[(surah, ayah)]
        except KeyError:
            raise VerseCorpusError(f"no such ayah in corpus: {surah}:{ayah}") from None

    def __iter__(self) -> Iterator[Ayah]:
        return iter(self._ayat.values())

    def text_range(self, surah: int, start: int, end: int) -> str:
        """Joined Uthmani text of ayat start..end (inclusive) of a surah."""
        if start > end:
            raise VerseCorpusError(f"bad ayah range {surah}:{start}-{end}: start > end")
        return " ".join(self.ayah(surah, n).uthmani for n in range(start, end + 1))


def _dev_quran_dir() -> Path:
    return Path(__file__).parents[3] / "assets" / "quran"


def _has_files(directory: Path) -> bool:
    return (directory / SIMPLE_NAME).is_file() and (directory / UTHMANI_NAME).is_file()


def default_corpus(settings: Settings) -> QuranCorpus:
    for directory in (settings.data_dir / "quran", _dev_quran_dir()):
        if _has_files(directory):
            return QuranCorpus.from_files(directory / SIMPLE_NAME, directory / UTHMANI_NAME)
    raise VerseCorpusError(f"Qur'an corpus not found; {_missing_hint(settings.data_dir)}")
