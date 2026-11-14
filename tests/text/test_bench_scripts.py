import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from narratoflow.text.tashkeel import bench as B

ROOT = Path(__file__).parents[2]
BA, FATHA = chr(0x0628), chr(0x064E)
LINES = [f"{BA}{FATHA}{BA}{FATHA} {BA}{FATHA}" + f" {BA}" * i for i in range(3)]
TEXT = "\n".join(LINES)


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gold_dir(tmp_path: Path) -> tuple[Path, Path]:
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "test.txt").write_text(TEXT, encoding="utf-8")
    manifest = B.make_manifest(
        TEXT, source_url="https://example.test/x", commit="c", licence_notes="n", min_words=1
    )
    path = tmp_path / "gold.toml"
    path.write_text(B.manifest_toml(manifest), encoding="utf-8")
    return gold, path


def test_bench_script_llm_and_catt_not_run(tmp_path: Path) -> None:
    bench = _script("bench_tashkeel")
    gold, manifest = _gold_dir(tmp_path)
    out = tmp_path / "results.json"
    code = bench.main(
        [
            "--gold", str(gold),
            "--manifest", str(manifest),
            "--engines", "identity,llm,catt",
            "--out", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    rows = {row["engine"]: row for row in json.loads(out.read_text(encoding="utf-8"))}
    assert list(rows) == ["identity", "llm", "catt"]
    assert rows["identity"]["not_run"] is None
    assert rows["identity"]["lines"] == 3
    assert "paid" in rows["llm"]["not_run"]
    assert "onnxruntime-gpu" in rows["catt"]["not_run"]


def test_bench_script_prints_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bench = _script("bench_tashkeel")
    gold, manifest = _gold_dir(tmp_path)
    bench.main(
        ["--gold", str(gold), "--manifest", str(manifest), "--engines", "identity", "--out",
         str(tmp_path / "r.json")]
    )  # fmt: skip
    assert "| identity | 3 | 0 |" in capsys.readouterr().out


def test_bench_script_reads_gold_bytes_exactly_like_the_manifest_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A CRLF gold file hashes with its carriage returns, as fetch_gold_tashkeel does."""
    bench = _script("bench_tashkeel")
    gold, _ = _gold_dir(tmp_path)
    crlf = TEXT.replace("\n", "\r\n")
    (gold / "test.txt").write_bytes(crlf.encode("utf-8"))
    manifest = B.make_manifest(
        crlf, source_url="https://example.test/x", commit="c", licence_notes="n", min_words=1
    )
    path = tmp_path / "crlf.toml"
    path.write_text(B.manifest_toml(manifest), encoding="utf-8")
    bench.main(["--gold", str(gold), "--manifest", str(path), "--engines", "identity"])
    assert "| identity |" in capsys.readouterr().out


def test_bench_script_rejects_unknown_engine(tmp_path: Path) -> None:
    bench = _script("bench_tashkeel")
    gold, manifest = _gold_dir(tmp_path)
    with pytest.raises(SystemExit):
        bench.main(["--gold", str(gold), "--manifest", str(manifest), "--engines", "nope"])


def test_bench_script_refuses_a_tampered_gold_file(tmp_path: Path) -> None:
    bench = _script("bench_tashkeel")
    gold, manifest = _gold_dir(tmp_path)
    (gold / "test.txt").write_text(TEXT.replace(BA, "x", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="gold line 1: sha256 mismatch"):
        bench.main(["--gold", str(gold), "--manifest", str(manifest), "--engines", "identity"])


def test_bench_script_mishkal_engine_runs_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _script("bench_tashkeel")

    class FakeMishkal:
        def diacritize(self, text: str) -> str:
            return text

    monkeypatch.setattr(bench, "MishkalDiacritizer", FakeMishkal)
    assert bench.build_engine("mishkal")("x") == "x"


def _fake_get(content: bytes) -> Any:
    def get(url: str, **_kw: Any) -> SimpleNamespace:
        return SimpleNamespace(content=content, raise_for_status=lambda: None)

    return get


def test_fetch_script_refuses_bad_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _script("fetch_gold_tashkeel")
    monkeypatch.setattr(fetch.httpx, "get", _fake_get(b"wrong bytes"))
    assert fetch.main(["--out", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out" / "test.txt").exists()
    assert not list((tmp_path / "out").glob("*"))


def test_fetch_script_writes_verified_file_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = _script("fetch_gold_tashkeel")
    data = TEXT.encode("utf-8")
    monkeypatch.setattr(fetch, "SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(fetch.httpx, "get", _fake_get(data))
    out = tmp_path / "out"
    assert fetch.main(["--out", str(out)]) == 0
    assert (out / "test.txt").read_bytes() == data
    manifest_path = tmp_path / "m.toml"
    monkeypatch.setattr(fetch, "MIN_WORDS", 1)
    assert fetch.main(["--out", str(out), "--make-manifest", str(manifest_path)]) == 0
    manifest = B.load_manifest(manifest_path)
    assert manifest.commit == fetch.COMMIT
    assert manifest.source_url == fetch.URL
    assert len(manifest.lines) == 3
    assert manifest_path.read_text(encoding="utf-8").isascii()


def test_fetch_script_make_manifest_refuses_a_modified_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = _script("fetch_gold_tashkeel")
    (tmp_path / "test.txt").write_bytes(b"not the pinned file")
    code = fetch.main(["--out", str(tmp_path), "--make-manifest", str(tmp_path / "m.toml")])
    assert code == 1
    assert not (tmp_path / "m.toml").exists()
