import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_gpl_pyarabic_is_not_a_core_dependency() -> None:
    """Nothing imports pyarabic; mishkal (the `arabic` extra) may still pull it in."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    core = [dep.lower() for dep in project["dependencies"]]
    assert not any(dep.startswith("pyarabic") for dep in core)
