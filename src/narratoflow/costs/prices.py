"""Creator-editable USD price table.

The packaged default lives in ``narratoflow/data/prices.toml``. A file named
``prices.toml`` in ``settings.data_dir`` overrides it. Free providers cost 0
for any model without needing an entry.
"""

from __future__ import annotations

import tomllib
from decimal import Decimal, InvalidOperation
from importlib import resources
from pathlib import Path
from typing import Any

from narratoflow.channels.keys import FREE_PROVIDERS
from narratoflow.errors import ChannelConfigError, PlanningError
from narratoflow.settings import Settings

_ZERO = Decimal(0)
_FIELDS: dict[str, tuple[str, ...]] = {
    "llm": ("input_per_mtok", "output_per_mtok"),
    "embed": ("per_mtok",),
    "image": ("per_image",),
    "vision": ("per_image_check",),
    "voice": ("per_kchar",),
}


def _decimal(value: object, where: str) -> Decimal:
    if not isinstance(value, str):
        raise ChannelConfigError(f"prices.toml: {where} must be a decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as e:
        raise ChannelConfigError(f"prices.toml: {where} is not a number: {value!r}") from e
    if not number.is_finite() or number < 0:
        raise ChannelConfigError(f"prices.toml: {where} must be a finite, non-negative number")
    return number


def _parse(raw: dict[str, Any]) -> dict[str, dict[str, tuple[Decimal, ...]]]:
    tables: dict[str, dict[str, tuple[Decimal, ...]]] = {}
    for kind, fields in _FIELDS.items():
        section = raw.get(kind, {})
        if not isinstance(section, dict):
            raise ChannelConfigError(f"prices.toml: [{kind}] must be a table")
        parsed: dict[str, tuple[Decimal, ...]] = {}
        for key, entry in section.items():
            if "/" not in key or not isinstance(entry, dict):
                raise ChannelConfigError(f'prices.toml: [{kind}."{key}"] must be provider/model')
            values = []
            for field in fields:
                if field not in entry:
                    raise ChannelConfigError(f'prices.toml: [{kind}."{key}"] missing {field}')
                values.append(_decimal(entry[field], f'{kind}."{key}".{field}'))
            parsed[key] = tuple(values)
        tables[kind] = parsed
    return tables


class PriceTable:
    def __init__(self, tables: dict[str, dict[str, tuple[Decimal, ...]]]) -> None:
        self._tables = tables

    @classmethod
    def _from_text(cls, text: str, origin: str) -> PriceTable:
        try:
            raw = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ChannelConfigError(f"Invalid price table {origin}: {e}") from e
        return cls(_parse(raw))

    @classmethod
    def load(cls, path: Path) -> PriceTable:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise ChannelConfigError(f"Cannot read price table {path}: {e}") from e
        return cls._from_text(text, str(path))

    def _lookup(self, kind: str, provider: str, model: str) -> tuple[Decimal, ...]:
        provider = provider.strip().lower()
        if provider in FREE_PROVIDERS:
            return tuple(_ZERO for _ in _FIELDS[kind])
        found = self._tables.get(kind, {}).get(f"{provider}/{model}")
        if found is None:
            raise PlanningError(f"no price for {provider}/{model}; add it to prices.toml")
        return found

    def llm(self, provider: str, model: str) -> tuple[Decimal, Decimal]:
        """USD per million (input, output) tokens."""
        inp, out = self._lookup("llm", provider, model)
        return inp, out

    def embed(self, provider: str, model: str) -> Decimal:
        """USD per million embedded tokens."""
        return self._lookup("embed", provider, model)[0]

    def image(self, provider: str, model: str) -> Decimal:
        """USD per generated image."""
        return self._lookup("image", provider, model)[0]

    def vision(self, provider: str, model: str) -> Decimal:
        """USD per image check."""
        return self._lookup("vision", provider, model)[0]

    def voice(self, provider: str, model: str) -> Decimal:
        """USD per thousand characters of speech."""
        return self._lookup("voice", provider, model)[0]


def default_prices(settings: Settings) -> PriceTable:
    override = settings.data_dir / "prices.toml"
    if override.is_file():
        return PriceTable.load(override)
    packaged = resources.files("narratoflow").joinpath("data", "prices.toml")
    return PriceTable._from_text(packaged.read_text(encoding="utf-8"), "packaged prices.toml")
