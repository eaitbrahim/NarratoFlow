import importlib
import sys
from collections.abc import Sequence
from types import ModuleType, SimpleNamespace

import pytest

from narratoflow.errors import DialectUnavailableError, ProviderError
from narratoflow.text.dialect import (
    DialectReport,
    SentenceDialect,
    check_dialect,
    load_camel_model,
    split_sentences,
)

AR = "\u0643\u062a\u0628"


class FakeModel:
    def __init__(self, labels: Sequence[tuple[str, float]]) -> None:
        self.labels = list(labels)
        self.calls: list[list[str]] = []

    def predict(self, sentences: Sequence[str]) -> list[tuple[str, float]]:
        self.calls.append(list(sentences))
        return self.labels[: len(sentences)]


def test_split_sentences_lines_and_marks() -> None:
    text = "\u0623 \u0628. \u062c\u061f\n\nd"
    assert split_sentences(text) == [
        (1, "\u0623 \u0628."),
        (1, "\u062c\u061f"),
        (3, "d"),
    ]


def test_flags_non_msa_above_threshold() -> None:
    model = FakeModel([("CAI", 0.8), ("MSA", 0.9), ("BEI", 0.3)])
    report = check_dialect(f"{AR}.\n{AR}.\n{AR}.", model, threshold=0.5, model_name="m")
    assert [s.flagged for s in report.sentences] == [True, False, False]
    assert [(s.label, s.confidence) for s in report.sentences] == [
        ("CAI", 0.8),
        ("MSA", 0.9),
        ("BEI", 0.3),
    ]
    assert report.flagged == (report.sentences[0],)
    assert model.calls == [[f"{AR}.", f"{AR}.", f"{AR}."]]


def test_threshold_boundary_inclusive() -> None:
    report = check_dialect(AR, FakeModel([("CAI", 0.5)]), threshold=0.5, model_name="m")
    assert report.sentences[0].flagged


def test_non_arabic_sentences_not_checked() -> None:
    model = FakeModel([("MSA", 0.9)])
    report = check_dialect(f"Hello.\n1453\n{AR}", model, threshold=0.5, model_name="m")
    assert model.calls == [[AR]]
    assert [(s.line, s.text) for s in report.sentences] == [(3, AR)]


@pytest.mark.parametrize("text", ["", "Hello"])
def test_empty_text_reports_nothing(text: str) -> None:
    model = FakeModel([])
    report = check_dialect(text, model, threshold=0.5, model_name="m")
    assert model.calls == []
    assert report.sentences == ()


@pytest.mark.parametrize("threshold", [0, -1, 1.5])
def test_bad_threshold_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        check_dialect(AR, FakeModel([]), threshold=threshold, model_name="m")


@pytest.mark.parametrize("confidence", [-0.1, 1.5, float("nan")])
def test_out_of_range_confidence_is_provider_error(confidence: float) -> None:
    with pytest.raises(ProviderError):
        check_dialect(AR, FakeModel([("CAI", confidence)]), threshold=0.5, model_name="m")


def test_confidence_field_is_bounded() -> None:
    with pytest.raises(ValueError):
        SentenceDialect(line=1, text=AR, label="CAI", confidence=1.5, flagged=True)


def test_wrong_prediction_count_is_provider_error() -> None:
    with pytest.raises(ProviderError):
        check_dialect(f"{AR}.\n{AR}.", FakeModel([("MSA", 0.9)]), threshold=0.5, model_name="m")


def test_report_round_trips() -> None:
    r = check_dialect(AR, FakeModel([("CAI", 0.7)]), threshold=0.5, model_name="m")
    assert DialectReport.model_validate_json(r.model_dump_json()) == r


def _install_fake_camel(monkeypatch: pytest.MonkeyPatch, identifier: object) -> None:
    module = ModuleType("camel_tools.dialectid")
    module.__dict__["DialectIdentifier"] = identifier
    monkeypatch.setitem(sys.modules, "camel_tools", ModuleType("camel_tools"))
    monkeypatch.setitem(sys.modules, "camel_tools.dialectid", module)


def test_camel_adapter_maps_didpred(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[list[str], str]] = []

    class Model:
        def predict(self, sentences: list[str], output: str) -> list[SimpleNamespace]:
            seen.append((sentences, output))
            return [SimpleNamespace(top="MSA", scores={"MSA": 0.9, "CAI": 0.1})]

    class Identifier:
        @staticmethod
        def pretrained() -> Model:
            return Model()

    _install_fake_camel(monkeypatch, Identifier)
    assert load_camel_model().predict((AR,)) == [("MSA", 0.9)]
    assert seen == [([AR], "label")]


def test_camel_import_error_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(name: str) -> object:
        raise ImportError("SECRET-detail")

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(DialectUnavailableError) as info:
        load_camel_model()
    assert "camel_data -i dialectid-model26" in str(info.value)
    assert "SECRET-detail" not in str(info.value)


def test_camel_load_error_hides_message(monkeypatch: pytest.MonkeyPatch) -> None:
    class Identifier:
        @staticmethod
        def pretrained() -> object:
            raise RuntimeError("SECRET-detail")

    _install_fake_camel(monkeypatch, Identifier)
    with pytest.raises(DialectUnavailableError) as info:
        load_camel_model()
    assert "camel_data -i dialectid-model26" in str(info.value)
    assert "RuntimeError" in str(info.value)
    assert "SECRET-detail" not in str(info.value)


def test_camel_predict_error_is_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model:
        def predict(self, sentences: list[str], output: str) -> list[object]:
            raise RuntimeError("SECRET-detail")

    class Identifier:
        @staticmethod
        def pretrained() -> Model:
            return Model()

    _install_fake_camel(monkeypatch, Identifier)
    model = load_camel_model()
    with pytest.raises(ProviderError) as info:
        model.predict((AR,))
    assert "RuntimeError" in str(info.value)
    assert "SECRET-detail" not in str(info.value)


def test_split_keeps_decimals_and_abbreviations_whole() -> None:
    text = f"{AR} 3.5 {AR}. U.S.A {AR}!"
    assert split_sentences(text) == [(1, f"{AR} 3.5 {AR}."), (1, f"U.S.A {AR}!")]


def test_split_breaks_at_end_of_text_and_before_whitespace() -> None:
    assert split_sentences("a.b c. d") == [(1, "a.b c."), (1, "d")]


def test_quranic_verses_are_not_sent_to_the_model() -> None:
    # Arabic text opens a verse with U+FD3F and closes it with U+FD3E (as quran/detect.py does).
    verse = "\ufd3f\u0628\u0633\u0645 \u0627\u0644\u0644\u0647.\n\u0643\u0644\u0645\u0629\ufd3e"
    text = f"{AR} \u0661.\n{verse} {AR}.\n{AR}."
    model = FakeModel([("CAI", 0.9), ("MSA", 0.9), ("MSA", 0.9)])
    report = check_dialect(text, model, threshold=0.5, model_name="m")
    sent = [s for call in model.calls for s in call]
    assert not any(
        c in s
        for s in sent
        for c in ("\u0628\u0633\u0645", "\u0643\u0644\u0645\u0629", "\ufd3e", "\ufd3f")
    )
    assert [s.line for s in report.sentences] == [1, 3, 4]


def test_narration_between_two_verses_is_checked() -> None:
    text = f"\ufd3f\u0628\u0633\u0645\ufd3e {AR} \ufd3f\u0643\u0644\u0645\u0629\ufd3e"
    model = FakeModel([("CAI", 0.9)])
    check_dialect(text, model, threshold=0.5, model_name="m")
    assert model.calls == [[AR]]
