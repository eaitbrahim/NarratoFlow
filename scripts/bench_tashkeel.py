"""Score diacritizers on the pinned gold set: DER and WER per engine, offline.

Engines: identity (free floor), mishkal (CPU, needs `uv run --with mishkal==0.4.1`), llm
(`LlmDiacritizer` on a local Ollama model at temperature 0, thinking off, $0), and catt
(catt-tashkeel 1.0.2, which needs onnxruntime-gpu: Linux only, so run it in a linux/arm64
container; see docs/adr/0005-diacritizer-benchmark.md for the exact `docker run`).

    uv run --with mishkal==0.4.1 python scripts/bench_tashkeel.py \
        --engines identity,mishkal,llm --out .cache/bench/tashkeel-results.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

from narratoflow.errors import ProviderError
from narratoflow.llm.base import LLMClient
from narratoflow.llm.ollama import OllamaClient
from narratoflow.text.tashkeel import bench
from narratoflow.text.tashkeel.engines import LlmDiacritizer, MishkalDiacritizer

ENGINES = ("identity", "mishkal", "llm", "catt")
LLM_URL = "http://localhost:11434"
LLM_MODEL = "hf.co/unsloth/GLM-4.7-Flash-GGUF:Q4_K_M"
CATT_HELP = (
    "catt-tashkeel is not installed. It needs onnxruntime-gpu (no macOS wheel), so run it in a "
    'linux/arm64 container: docker run --rm --platform linux/arm64 -v "$PWD":/w -w /w '
    "-e UV_PROJECT_ENVIRONMENT=/tmp/venv ghcr.io/astral-sh/uv:python3.12-bookworm-slim "
    "uv run --with catt-tashkeel==1.0.2 python scripts/bench_tashkeel.py --engines catt"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--gold", type=Path, default=Path(".cache/bench/tashkeel"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/benchmarks/tashkeel-gold.toml"))
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--llm-url", default=LLM_URL)
    parser.add_argument("--llm-model", default=LLM_MODEL)
    return parser.parse_args(argv)


def _llm_engine(args: argparse.Namespace, llm: LLMClient | None) -> Callable[[str], str]:
    client = llm or OllamaClient(args.llm_url, args.llm_model, temperature=0.0, think=False)
    return LlmDiacritizer(client).diacritize


def _catt_engine(import_module: Callable[[str], ModuleType]) -> Callable[[str], str]:
    try:
        catt: Any = import_module("catt_tashkeel")
    except ImportError as e:
        raise SystemExit(CATT_HELP) from e
    tagger = catt.CATTEncoderOnly()

    def diacritize(text: str) -> str:
        try:
            return str(tagger.do_tashkeel_batch([text], batch_size=1, verbose=False)[0])
        except Exception as e:  # any CATT or onnxruntime failure rejects the line
            raise ProviderError(f"catt failed on one line: {e}") from e

    return diacritize


def build_engine(
    name: str,
    args: argparse.Namespace | None = None,
    *,
    llm: LLMClient | None = None,
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> Callable[[str], str]:
    args = args if args is not None else parse_args([])
    if name == "identity":
        return lambda text: text
    if name == "llm":
        return _llm_engine(args, llm)
    if name == "catt":
        return _catt_engine(import_module)
    return MishkalDiacritizer().diacritize


def engine_label(name: str, args: argparse.Namespace) -> str:
    """The row name: the llm row records provider and model."""
    if name == "llm":
        return f"llm (ollama {args.llm_model})"
    return name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = [n.strip() for n in args.engines.split(",") if n.strip()]
    unknown = [n for n in names if n not in ENGINES]
    if unknown:
        raise SystemExit(f"unknown engine {unknown[0]!r}; choose from {', '.join(ENGINES)}")
    manifest = bench.load_manifest(args.manifest)
    text = (args.gold / "test.txt").read_bytes().decode("utf-8")
    gold = bench.select_gold(text, manifest)
    engines = [(engine_label(n, args), build_engine(n, args)) for n in names]  # fail before work
    results = [bench.run_engine(label, engine, gold) for label, engine in engines]
    print(bench.results_markdown(results))
    if args.out is not None:
        args.out.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
