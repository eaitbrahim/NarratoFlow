from decimal import Decimal
from pathlib import Path

import pytest

from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.errors import ChannelConfigError, PlanningError
from narratoflow.settings import Settings


@pytest.fixture
def prices() -> PriceTable:
    return default_prices(Settings(data_dir=Path("/nonexistent-narratoflow")))


def test_packaged_table_loads(prices: PriceTable) -> None:
    assert prices.llm("anthropic", "claude-opus-5-5") == (Decimal("4.00"), Decimal("20.00"))
    assert prices.embed("voyage", "voyage-multilingual-2") == Decimal("0.12")
    assert prices.image("openai", "gpt-image-1") == Decimal("0.04")
    assert prices.vision("anthropic", "claude-opus-5-5") == Decimal("0.01")
    assert prices.voice("elevenlabs", "eleven_multilingual_v2") == Decimal("0.30")


def test_free_provider_costs_zero(prices: PriceTable) -> None:
    assert prices.llm("ollama", "qwen2.5") == (Decimal(0), Decimal(0))
    assert prices.llm("OLLAMA", "x") == (Decimal(0), Decimal(0))
    assert prices.voice("edge", "") == Decimal(0)
    assert prices.image("local-diffusion", "sdxl") == Decimal(0)
    assert prices.embed("ollama", "x") == Decimal(0)
    assert prices.vision("ollama", "x") == Decimal(0)


def test_unknown_paid_model_explains(prices: PriceTable) -> None:
    with pytest.raises(PlanningError, match=r"prices\.toml"):
        prices.llm("anthropic", "made-up")


def test_data_dir_override(tmp_path: Path) -> None:
    (tmp_path / "prices.toml").write_text('[embed."a/b"]\nper_mtok = "9"\n', encoding="utf-8")
    table = default_prices(Settings(data_dir=tmp_path))
    assert table.embed("a", "b") == Decimal(9)
    with pytest.raises(PlanningError):
        table.llm("anthropic", "claude-opus-5-5")


@pytest.mark.parametrize(
    "body",
    [
        "not toml [",
        '[llm."a/b"]\ninput_per_mtok = "-1"\noutput_per_mtok = "1"\n',
        '[llm."a/b"]\ninput_per_mtok = "abc"\noutput_per_mtok = "1"\n',
        '[llm."a/b"]\ninput_per_mtok = "1"\n',
        '[llm."a/b"]\ninput_per_mtok = 1.5\noutput_per_mtok = "1"\n',
        '[llm."nokey"]\ninput_per_mtok = "1"\noutput_per_mtok = "1"\n',
        '[embed."a/b"]\nper_mtok = "NaN"\n',
        "llm = 3\n",
    ],
)
def test_bad_values_rejected(tmp_path: Path, body: str) -> None:
    path = tmp_path / "prices.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ChannelConfigError):
        PriceTable.load(path)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ChannelConfigError):
        PriceTable.load(tmp_path / "nope.toml")
