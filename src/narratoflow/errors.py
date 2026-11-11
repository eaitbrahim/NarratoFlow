"""Project error hierarchy."""

from decimal import Decimal

from narratoflow.money import format_usd


class NarratoFlowError(Exception):
    """Base class for all NarratoFlow errors."""


class MissingApiKeyError(NarratoFlowError):
    """A required provider API key is not configured."""


class ChannelConfigError(NarratoFlowError):
    """A channel configuration is missing or invalid."""


class ExtraMissingError(ChannelConfigError):
    """An optional extra (e.g. 'tts-edge') is not installed; the message names it."""


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


class BilledProviderError(ProviderError):
    """The provider accepted (and bills) the call, but its response was unusable."""


class ProviderRefusalError(ProviderError):
    """The model refused the request."""

    def __init__(self, category: str | None = None) -> None:
        self.category = category
        super().__init__(f"provider refused the request (category: {category})")


class BudgetExceededError(NarratoFlowError):
    """An estimate exceeds the configured spending limit."""

    def __init__(
        self,
        estimate_usd: Decimal,
        limit_usd: Decimal,
        *,
        spent_usd: Decimal | None = None,
        what: str | None = None,
    ) -> None:
        """With `spent_usd`, `estimate_usd` is the call's own estimate and the message
        names the remaining room; without it, `estimate_usd` is the projected total.
        """
        self.estimate_usd = estimate_usd
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
        self.what = what
        if spent_usd is None:
            message = (
                f"estimated cost {format_usd(estimate_usd)} exceeds limit {format_usd(limit_usd)}"
            )
        else:
            remaining = max(Decimal(0), limit_usd - spent_usd)
            prefix = f"{what}: " if what else ""
            message = (
                f"{prefix}estimated cost {format_usd(estimate_usd)} exceeds the remaining "
                f"{format_usd(remaining)} (spent {format_usd(spent_usd)} of the "
                f"{format_usd(limit_usd)} per-part limit; "
                "raise spending_limit_usd in the channel file)"
            )
        super().__init__(message)


class SegmentationError(NarratoFlowError):
    """A script could not be segmented into parts or scenes."""


class PlanningError(NarratoFlowError):
    """Script planning failed or a series file is invalid."""


class VerseCorpusError(NarratoFlowError):
    """The verse corpus is missing, corrupt or cannot resolve a reference."""


class SketchRejectedError(NarratoFlowError):
    """A generated sketch is not clean line art (empty, filled or hatched)."""


class StrokeDataError(NarratoFlowError):
    """Stroke data is malformed (bad JSON, out-of-range points) or the ink mask is invalid."""


class AudioError(NarratoFlowError):
    """Audio decoding, encoding, loudness or assembly failed."""


class ReciterError(AudioError):
    """A verse recitation could not be resolved (no TTS fallback exists)."""


class AlignerUnavailableError(AudioError):
    """The forced aligner is not installed or cannot run."""
