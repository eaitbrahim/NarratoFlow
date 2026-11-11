from pathlib import Path

import pytest
from conftest import FIXTURES

from narratoflow.errors import CorruptInputError, ScriptBlockedError
from narratoflow.models import Language, RightsBasis, ScriptSource
from narratoflow.script import source as source_module
from narratoflow.script.source import check_rights, load_script


def read(name: str) -> str:
    return (FIXTURES / "input" / name).read_text(encoding="utf-8")


def write_script(tmp_path: Path, rights: str, body: str, extra: str = "") -> Path:
    path = tmp_path / "script.txt"
    path.write_text(
        f'+++\nauthor = "A"\nwork = "W"\nrights = "{rights}"\n{extra}+++\n{body}',
        encoding="utf-8",
    )
    return path


def test_own_script_loads(tmp_path: Path) -> None:
    body = read("clean_ar.txt")
    source, text = load_script(write_script(tmp_path, "own", body), Language.ARABIC)
    assert source.rights is RightsBasis.OWN
    assert source.author == "A"
    assert text.strip() == body.strip()


def test_in_copyright_blocked(tmp_path: Path) -> None:
    p = write_script(tmp_path, "third_party_in_copyright", read("clean_en.txt"))
    with pytest.raises(ScriptBlockedError, match="copyright"):
        load_script(p, Language.ENGLISH)


def test_permission_requires_note(tmp_path: Path) -> None:
    p = write_script(tmp_path, "permission", read("clean_en.txt"))
    with pytest.raises(ScriptBlockedError, match="permission"):
        load_script(p, Language.ENGLISH)


def test_permission_with_note_loads(tmp_path: Path) -> None:
    p = write_script(
        tmp_path, "permission", read("clean_en.txt"), 'permission_note = "email 2026"\n'
    )
    source, _ = load_script(p, Language.ENGLISH)
    assert source.permission_note == "email 2026"


def test_missing_front_matter(tmp_path: Path) -> None:
    p = tmp_path / "s.txt"
    p.write_text(read("clean_en.txt"), encoding="utf-8")
    with pytest.raises(ScriptBlockedError, match=r"\+\+\+"):
        load_script(p, Language.ENGLISH)


def test_unterminated_front_matter(tmp_path: Path) -> None:
    p = tmp_path / "s.txt"
    p.write_text('+++\nauthor = "A"\n', encoding="utf-8")
    with pytest.raises(ScriptBlockedError, match="closing"):
        load_script(p, Language.ENGLISH)


def test_invalid_toml(tmp_path: Path) -> None:
    p = tmp_path / "s.txt"
    p.write_text("+++\nauthor = \n+++\nbody", encoding="utf-8")
    with pytest.raises(ScriptBlockedError, match="TOML"):
        load_script(p, Language.ENGLISH)


def test_invalid_fields(tmp_path: Path) -> None:
    p = write_script(tmp_path, "bogus", "body")
    with pytest.raises(ScriptBlockedError, match="invalid"):
        load_script(p, Language.ENGLISH)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ScriptBlockedError, match="cannot read"):
        load_script(tmp_path / "nope.txt", Language.ENGLISH)


def test_not_utf8(tmp_path: Path) -> None:
    p = tmp_path / "s.txt"
    p.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ScriptBlockedError, match="cannot read"):
        load_script(p, Language.ENGLISH)


def test_corrupt_body_rejected(tmp_path: Path) -> None:
    p = write_script(tmp_path, "own", read("corrupt_ar.txt"))
    with pytest.raises(CorruptInputError):
        load_script(p, Language.ARABIC)


def test_ai_transcribed_flag_kept(tmp_path: Path) -> None:
    p = write_script(tmp_path, "own", read("clean_en.txt"), "ai_transcribed = true\n")
    source, _ = load_script(p, Language.ENGLISH)
    assert source.ai_transcribed is True


def test_check_rights_direct() -> None:
    check_rights(ScriptSource(author="a", work="w", rights=RightsBasis.PUBLIC_DOMAIN))


def test_unknown_key_rejected(tmp_path: Path) -> None:
    p = write_script(tmp_path, "own", "body", 'permission_notes = "x"\n')
    with pytest.raises(ScriptBlockedError, match="permission_notes"):
        load_script(p, Language.ENGLISH)


def test_indented_fence_not_accepted(tmp_path: Path) -> None:
    p = tmp_path / "s.txt"
    p.write_text(' +++\nauthor = "A"\n+++\nbody', encoding="utf-8")
    with pytest.raises(ScriptBlockedError, match=r"\+\+\+"):
        load_script(p, Language.ENGLISH)


def test_bom_and_crlf_load(tmp_path: Path) -> None:
    text = '+++\nauthor = "A"\nwork = "W"\nrights = "own"\n+++\n' + read("clean_en.txt")
    p = tmp_path / "s.txt"
    p.write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))
    source, body = load_script(p, Language.ENGLISH)
    assert source.author == "A"
    assert "\r" not in body


def test_source_module_is_ascii() -> None:
    path = Path(source_module.__file__ or "")
    assert path.read_bytes().isascii()
