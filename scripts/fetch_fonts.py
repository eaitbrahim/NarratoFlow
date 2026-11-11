"""Fetch the bundled OFL fonts into assets/fonts/ (needs network). Run manually:

    uv run python scripts/fetch_fonts.py

Downloads one zip per upstream release, verifies the pinned zip sha256, extracts only the named
members and writes assets/fonts/index.toml with each file's own sha256 and provenance.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

OUT = Path(__file__).parents[1] / "assets" / "fonts"


@dataclass(frozen=True)
class FontMember:
    id: str
    member: str
    file: str
    family: str


@dataclass(frozen=True)
class ZipSpec:
    url: str
    sha256: str
    fonts: tuple[FontMember, ...]
    license_member: str
    license_file: str


SPECS = (
    ZipSpec(
        url="https://github.com/aliftype/amiri/releases/download/1.003/Amiri-1.003.zip",
        sha256="81af0aff7d2086d8af24cea7202f7546130997982534691373485cd96744d05e",
        fonts=(
            FontMember(
                "amiri-regular", "Amiri-1.003/Amiri-Regular.ttf", "Amiri-Regular.ttf", "Amiri"
            ),
            FontMember("amiri-bold", "Amiri-1.003/Amiri-Bold.ttf", "Amiri-Bold.ttf", "Amiri"),
            FontMember(
                "amiri-quran", "Amiri-1.003/AmiriQuran.ttf", "AmiriQuran.ttf", "Amiri Quran"
            ),
        ),
        license_member="Amiri-1.003/OFL.txt",
        license_file="OFL-Amiri.txt",
    ),
    ZipSpec(
        url=(
            "https://github.com/notofonts/latin-greek-cyrillic/releases/download/"
            "NotoSans-v2.015/NotoSans-v2.015.zip"
        ),
        sha256="0c34df072a3fa7efbb7cbf34950e1f971a4447cffe365d3a359e2d4089b958f5",
        fonts=(
            FontMember(
                "notosans-regular",
                "NotoSans/hinted/ttf/NotoSans-Regular.ttf",
                "NotoSans-Regular.ttf",
                "Noto Sans",
            ),
            FontMember(
                "notosans-bold",
                "NotoSans/hinted/ttf/NotoSans-Bold.ttf",
                "NotoSans-Bold.ttf",
                "Noto Sans",
            ),
        ),
        license_member="OFL.txt",
        license_file="OFL-NotoSans.txt",
    ),
)


def extract_pack(data: bytes, spec: ZipSpec, out_dir: Path) -> dict[str, str]:
    """Verify `data` against the pinned zip sha256 and extract only the named members.

    Returns {font id: sha256 of the extracted file}. Raises ValueError on a bad zip hash or a
    missing member; nothing is written in that case.
    """
    got = hashlib.sha256(data).hexdigest()
    if got != spec.sha256:
        raise ValueError(f"{spec.url}: zip sha256 {got} != pinned {spec.sha256}")
    shas: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        wanted = [m.member for m in spec.fonts] + [spec.license_member]
        missing = [m for m in wanted if m not in names]
        if missing:
            raise ValueError(f"{spec.url}: missing zip members {missing}")
        blobs = {m: archive.read(m) for m in wanted}
    out_dir.mkdir(parents=True, exist_ok=True)
    for font in spec.fonts:
        blob = blobs[font.member]
        (out_dir / font.file).write_bytes(blob)
        shas[font.id] = hashlib.sha256(blob).hexdigest()
    (out_dir / spec.license_file).write_bytes(blobs[spec.license_member])
    return shas


def index_text(shas: dict[str, str]) -> str:
    parts: list[str] = []
    for spec in SPECS:
        for font in spec.fonts:
            parts.append(
                f'[fonts.{font.id}]\nfile = "{font.file}"\nfamily = "{font.family}"\n'
                f'license = "OFL-1.1"\nlicense_file = "{spec.license_file}"\n'
                f'source = "{spec.url}#{font.member}"\nsha256 = "{shas[font.id]}"\n'
            )
    return "\n".join(parts)


def main() -> None:
    shas: dict[str, str] = {}
    for spec in SPECS:
        with urllib.request.urlopen(spec.url, timeout=120) as resp:  # noqa: S310
            data = resp.read()
        shas.update(extract_pack(data, spec, OUT))
    (OUT / "index.toml").write_text(index_text(shas), encoding="utf-8")


if __name__ == "__main__":
    main()
