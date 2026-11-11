"""Download the Tanzil Qur'an text (txt-2 format) and verify pinned sha256 digests.

Smoke-run manually only; there are no automated tests because it needs the network.
Licence header lines in the files are kept intact (see assets/quran/LICENSE-tanzil.txt).

    uv run python scripts/fetch_quran.py [--out DIR]
"""

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import httpx

_BASE = "https://tanzil.net/pub/download/index.php?quranType={kind}&outType=txt-2&agree=true"

# file name -> (Tanzil quranType, pinned sha256). Tanzil text v1.1.
FILES: dict[str, tuple[str, str]] = {
    "quran-simple-clean.txt": (
        "simple-clean",
        "228df2a717671aeb9d2ff573002bd28d6b3f973f4bc7153554e3a81663d67610",
    ),
    "quran-uthmani.txt": (
        "uthmani",
        "bf4f57b968d03f4131c070b1e285da9be0e0a108a21c910e872801ca273312c8",
    ),
}


def _write_atomic(target: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=Path("assets/quran"))
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, (kind, expected) in FILES.items():
        response = httpx.get(_BASE.format(kind=kind), follow_redirects=True, timeout=60)
        response.raise_for_status()
        actual = hashlib.sha256(response.content).hexdigest()
        if actual != expected:
            print(f"sha256 mismatch for {name}: expected {expected}, got {actual}", file=sys.stderr)
            return 1
        _write_atomic(out / name, response.content)
        print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
