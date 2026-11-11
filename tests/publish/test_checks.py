from pathlib import Path

import pytest
from pydantic import ValidationError

from narratoflow.publish.checks import (
    CheckId,
    CheckResult,
    Severity,
    bare_paths,
    summarize,
    worst,
)


def _r(severity: Severity) -> CheckResult:
    return CheckResult(check=CheckId.TECHNICAL, severity=severity, reason="x")


def test_worst_and_summarize() -> None:
    assert worst([]) is Severity.PASS
    assert worst([_r(Severity.WARN), _r(Severity.FAIL)]) is Severity.FAIL
    assert worst([_r(Severity.PASS), _r(Severity.WARN)]) is Severity.WARN
    only = summarize(CheckId.VERSE_TEXT, [], "all good")
    assert [(r.check, r.severity, r.reason) for r in only] == [
        (CheckId.VERSE_TEXT, Severity.PASS, "all good")
    ]
    finding = _r(Severity.FAIL)
    assert summarize(CheckId.TECHNICAL, [finding], "ok") == [finding]


def test_check_result_rejects_empty_reason_and_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        CheckResult(check=CheckId.TECHNICAL, severity=Severity.PASS, reason="")
    with pytest.raises(ValidationError):
        CheckResult.model_validate(
            {"check": "technical", "severity": "PASS", "reason": "x", "extra": 1}
        )


def test_publish_sources_are_ascii() -> None:
    root = Path(__file__).parents[2] / "src" / "narratoflow" / "publish"
    paths = sorted(root.glob("*.py"))
    assert len(paths) >= 10
    for path in paths:
        path.read_bytes().decode("ascii")  # raises UnicodeDecodeError on any non-ASCII byte


def test_bare_paths_cuts_directories() -> None:
    assert bare_paths("cannot read /Users/x/data/quran/q.txt: bad") == "cannot read q.txt: bad"
    assert bare_paths("[Errno 2] No such file: '/tmp/a/b.ass'") == "[Errno 2] No such file: 'b.ass'"
    assert bare_paths("ratio 9/16 and a/b stay") == "ratio 9/16 and a/b stay"
