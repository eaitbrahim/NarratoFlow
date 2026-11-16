"""NarratoFlow: open-source whiteboard animation video engine."""

from importlib.metadata import version

# One source of truth: `[project].version` in pyproject.toml, bumped by the release workflow.
__version__ = version("narratoflow")
