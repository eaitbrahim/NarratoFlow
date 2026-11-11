"""Project error hierarchy."""


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
