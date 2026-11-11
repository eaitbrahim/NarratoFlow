import pytest
from pydantic import ValidationError

from narratoflow.publish.checks import CheckId, CheckResult, Severity, summarize, worst


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
