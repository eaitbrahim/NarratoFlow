"""Download the diacritizer gold set (Fadel et al. test split) and verify its pinned sha256.

The text derives from the Tashkeela corpus (GPLv2), so the repo commits no Arabic text:
the file goes to a git-ignored cache and docs/benchmarks/tashkeel-gold.toml holds only
per-line hashes. Needs the network once (about 1.7 MB).

    uv run python scripts/fetch_gold_tashkeel.py [--out DIR]
    uv run python scripts/fetch_gold_tashkeel.py --make-manifest docs/benchmarks/tashkeel-gold.toml
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import httpx

from narratoflow.text.tashkeel import bench

COMMIT = "e083579c03e489936200b69d792fd426bb576743"
URL = (
    f"https://raw.githubusercontent.com/AliOsm/arabic-text-diacritization/{COMMIT}/dataset/test.txt"
)
SHA256 = "4e851ff836f0a178abb15d9f4a8bcf92748b77cc0d67030d9fae2fbe698baa12"
NAME = "test.txt"
COUNT = 300
MIN_WORDS = 5
MAX_WORDS = 60
LICENCE_NOTES = (
    "Repository AliOsm/arabic-text-diacritization (Fadel et al. 2019) is MIT licensed. "
    "Its text derives from the Tashkeela corpus (Zerrouki and Balla 2017), listed as GPLv2 "
    "on SourceForge and in the Hugging Face loader (the TFDS catalog says no known licence); "
    "the texts come from the Shamela library. This project is Apache-2.0 and the layering is "
    "not clearly redistributable, so no Arabic text is committed: fetch it with "
    "scripts/fetch_gold_tashkeel.py."
)


def _write_atomic(target: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _mismatch(actual: bytes) -> bool:
    digest = hashlib.sha256(actual).hexdigest()
    if digest == SHA256:
        return False
    print(f"sha256 mismatch: expected {SHA256}, got {digest}", file=sys.stderr)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=Path(".cache/bench/tashkeel"))
    parser.add_argument("--make-manifest", type=Path, metavar="PATH")
    args = parser.parse_args(argv)
    target: Path = args.out / NAME
    if args.make_manifest is not None:
        if not target.is_file():
            print(f"{target} is missing: run this script without --make-manifest", file=sys.stderr)
            return 1
        data = target.read_bytes()
        if _mismatch(data):
            return 1
        manifest = bench.make_manifest(
            data.decode("utf-8"),
            source_url=URL,
            commit=COMMIT,
            licence_notes=LICENCE_NOTES,
            count=COUNT,
            min_words=MIN_WORDS,
            max_words=MAX_WORDS,
        )
        args.make_manifest.write_text(bench.manifest_toml(manifest), encoding="utf-8")
        print(f"wrote {args.make_manifest} ({len(manifest.lines)} lines)")
        return 0
    response = httpx.get(URL, follow_redirects=True, timeout=60)
    response.raise_for_status()
    if _mismatch(response.content):
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    _write_atomic(target, response.content)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
