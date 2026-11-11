"""Project error hierarchy."""

from decimal import Decimal

from narratoflow.money import format_usd


class NarratoFlowError(Exception):
    """Base class for all NarratoFlow errors."""


class MissingApiKeyError(NarratoFlowError):
    """A required provider API key is not configured."""


class ChannelConfigError(NarratoFlowError):
    """A channel configuration is missing or invalid."""


class InvalidIdeaKeyError(NarratoFlowError):
    """An idea key is malformed."""


class LibraryError(NarratoFlowError):
    """The sketch library could not complete an operation."""


class CorruptInputError(NarratoFlowError):
    """An input file is corrupt or unreadable."""


class SettingsError(NarratoFlowError):
    """Application settings (environment or .env) are invalid."""


class ScriptBlockedError(NarratoFlowError):
    """A script was blocked before any spend (e.g. in-copyright third-party work)."""


class ProviderError(NarratoFlowError):
    """A paid or local provider call failed."""


class ProviderRefusalError(ProviderError):
    """The model refused the request."""

    def __init__(self, category: str | None = None) -> None:
        self.category = category
        super().__init__(f"provider refused the request (category: {category})")


class BudgetExceededError(NarratoFlowError):
    """An estimate exceeds the configured spending limit."""

    def __init__(self, estimate_usd: Decimal, limit_usd: Decimal) -> None:
        self.estimate_usd = estimate_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"estimated cost {format_usd(estimate_usd)} exceeds limit {format_usd(limit_usd)}"
        )


class SegmentationError(NarratoFlowError):
    """A script could not be segmented into parts or scenes."""


class PlanningError(NarratoFlowError):
    """Script planning failed or a series file is invalid."""


class VerseCorpusError(NarratoFlowError):
    """The verse corpus is missing, corrupt or cannot resolve a reference."""


class SketchRejectedError(NarratoFlowError):
    """A generated sketch is not clean line art (empty, filled or hatched)."""
