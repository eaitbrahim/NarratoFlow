# Asset Licenses

Every bundled asset pack must be listed here with source URL, license and attribution text.
Allowed: CC0, MIT, ISC, Apache-2.0, SIL OFL. CC-BY / CC-BY-SA only with attribution recorded.

| Pack | Path | Source | License | Attribution |
|---|---|---|---|---|
| Amiri 1.003 (Regular, Bold, Quran) | assets/fonts/ | https://github.com/aliftype/amiri/releases/download/1.003/Amiri-1.003.zip | OFL-1.1 (OFL-Amiri.txt) | — |
| Noto Sans v2.015 (Regular, Bold; hinted) | assets/fonts/ | https://github.com/notofonts/latin-greek-cyrillic/releases/download/NotoSans-v2.015/NotoSans-v2.015.zip | OFL-1.1 (OFL-NotoSans.txt) | — |
| Hand sprite (right hand with marker) | assets/hands/ | scripts/make_hand.py (drawn polygons, NarratoFlow's own work) | Apache-2.0 | — |
| Tabler Icons 3.49.0 (206 outline icons, rasterized) | assets/icons/ | https://registry.npmjs.org/@tabler/icons/-/icons-3.49.0.tgz | MIT (LICENSE-tabler.txt) | Copyright (c) 2020-2026 Paweł Kuna |

## Reciters

Every reciter in `assets/reciters/index.toml` must be listed here with its licence,
licence URL, recording source, segment-timing source and the date its terms were checked.
The terms must allow use of the recordings in published (possibly monetised) videos.

No reciter is shipped (checked 2026-11-11). Rejected sources:

- quran.com (https://quran.com/terms-and-conditions): personal, non-commercial use only.
- quranicaudio.com (https://quranicaudio.com/about): no commercial use.
- everyayah.com (https://everyayah.com/recitations_ayat.html): no licence stated.
- alquran.cloud (https://alquran.cloud/terms-and-conditions, section IV): a non-commercial
  redistribution licence; the reciters keep copyright and may require removal at their
  discretion.

Verse scenes therefore FAIL until a licensed reciter is added.

| Reciter id | Riwaya | Source | License | License URL | Segments | Checked |
|---|---|---|---|---|---|---|

## SFX

| Pack | Path | Source | License | Attribution |
|---|---|---|---|---|
| Marker squeaks (4) | assets/sfx/ | scripts/make_sfx.py (synthesized noise, NarratoFlow's own work) | Apache-2.0 | — |

Regenerate with `uv run python scripts/make_sfx.py` (needs ffmpeg); output is committed.
Index licences must be one of CC0-1.0, MIT, ISC, Apache-2.0, CC-BY-4.0 (the last with attribution).

Fonts: re-fetch with `uv run python scripts/fetch_fonts.py` (pinned zip sha256, needs network); icons: `uv run python scripts/fetch_icons.py` (pinned npm tarball sha256, needs network); the hand: `uv run python scripts/make_hand.py`. Output is committed; per-file sha256 values live in each `index.toml`.
