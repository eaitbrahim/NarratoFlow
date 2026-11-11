"""Money formatting shared by errors, planning and the CLI (imports nothing heavy)."""

from decimal import Decimal

_CENT = Decimal("0.01")


def format_usd(amount: Decimal) -> str:
    """Whole cents as `$0.12`; finer amounts to four decimals, e.g. `$0.0123`."""
    if amount == amount.quantize(_CENT):
        return f"${amount:.2f}"
    return f"${amount:.4f}"
