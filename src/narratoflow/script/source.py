"""Load a script with its declared source, and gate on rights before any spend."""

import tomllib
from pathlib import Path

from pydantic import ValidationError

from narratoflow.errors import ScriptBlockedError
from narratoflow.models import Language, RightsBasis, ScriptSource
from narratoflow.text.quality import require_clean

_FENCE = "+++"
_EXAMPLE = (
    "+++\n"
    'author = "Your Name"\n'
    'work = "Title of the work"\n'
    'edition = ""\n'
    'rights = "own"  # own | public_domain | permission\n'
    'permission_note = ""\n'
    "ai_transcribed = false\n"
    "+++"
)


def check_rights(source: ScriptSource) -> None:
    """Block in-copyright third-party work, and permission claims without a note."""
    if source.rights is RightsBasis.THIRD_PARTY_IN_COPYRIGHT:
        raise ScriptBlockedError(
            f"'{source.work}' by {source.author} is third-party work in copyright; "
            "it cannot be used."
        )
    if source.rights is RightsBasis.PERMISSION and not source.permission_note.strip():
        raise ScriptBlockedError(
            "rights = 'permission' requires a non-empty permission_note describing the permission."
        )


def _split(raw: str) -> tuple[str, str]:
    lines = raw.removeprefix("\ufeff").replace("\r\n", "\n").split("\n")
    if lines[0].rstrip() != _FENCE:
        raise ScriptBlockedError(
            f"declare the script source: the file must start with a front-matter block, "
            f"for example:\n{_EXAMPLE}"
        )
    for index in range(1, len(lines)):
        if lines[index].rstrip() == _FENCE:
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    raise ScriptBlockedError(f"front matter has no closing '{_FENCE}' line.")


def load_script(path: Path, language: Language) -> tuple[ScriptSource, str]:
    """Read a script file, validate its source declaration and rights, return source and body."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ScriptBlockedError(f"cannot read script {path}: {exc}") from exc
    front, body = _split(raw)
    try:
        data = tomllib.loads(front)
    except tomllib.TOMLDecodeError as exc:
        raise ScriptBlockedError(f"front matter is not valid TOML: {exc}") from exc
    try:
        source = ScriptSource.model_validate(data)
    except ValidationError as exc:
        raise ScriptBlockedError(f"front matter is invalid:\n{exc}") from exc
    check_rights(source)
    return source, require_clean(body, language)
