"""Score diacritizers on the pinned gold set: DER and WER per engine, offline.

Engines: identity (free floor), mishkal (CPU, needs `uv run --with mishkal==0.4.1`), and
llm / catt, which are listed as not run (paid; no macOS wheel) and never executed here.

    uv run --with mishkal==0.4.1 python scripts/bench_tashkeel.py \
        --engines identity,mishkal,llm,catt --out docs/benchmarks/2026-11-13-tashkeel-results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from narratoflow.text.tashkeel import bench
from narratoflow.text.tashkeel.engines import MishkalDiacritizer

ENGINES = ("identity", "mishkal", "llm", "catt")
NOT_RUN = {
    "llm": "not run (paid): needs a budgeted LLM run (ruling R3); see ADR 0005 for the issue",
    "catt": (
        "not run: catt-tashkeel needs onnxruntime-gpu, which has no macOS arm64 wheel; "
        "its weights are a 72 to 86 MB download and its training data may overlap Tashkeela"
    ),
}


def build_engine(name: str) -> Callable[[str], str]:
    if name == "identity":
        return lambda text: text
    return MishkalDiacritizer().diacritize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--gold", type=Path, default=Path(".cache/bench/tashkeel"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/benchmarks/tashkeel-gold.toml"))
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    names = [n.strip() for n in args.engines.split(",") if n.strip()]
    unknown = [n for n in names if n not in ENGINES]
    if unknown:
        parser.error(f"unknown engine {unknown[0]!r}; choose from {', '.join(ENGINES)}")
    manifest = bench.load_manifest(args.manifest)
    text = (args.gold / "test.txt").read_bytes().decode("utf-8")
    gold = bench.select_gold(text, manifest)
    results = [
        bench.not_run(n, NOT_RUN[n]) if n in NOT_RUN else bench.run_engine(n, build_engine(n), gold)
        for n in names
    ]
    print(bench.results_markdown(results))
    if args.out is not None:
        args.out.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
