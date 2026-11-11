"""AI-disclosure text (spec 6.3) and the disclosure-set check."""

from __future__ import annotations

from narratoflow.models import Language
from narratoflow.publish.checks import CheckId, CheckResult, PartInputs, Severity
from narratoflow.publish.provenance import is_generated
from narratoflow.publish.rules import Rules

_EN = {
    "header": "AI disclosure:",
    "voice": "- The narration voice is AI-generated ({provider}).",
    "sketches": "- The sketches were generated with AI image models ({models}).",
    "none": "- No AI-generated voice or images are used in this part.",
    "recitation": "- Qur'an recitation: a recording by {name}, used under {license}.",
    "script": "- Script: {text}.",
    "script_none": "- Script: not recorded.",
    "sep": ", ",
}
_AR = {
    "header": (
        "\u0625\u0641\u0635\u0627\u062d \u0639\u0646 \u0627\u0633"
        "\u062a\u062e\u062f\u0627\u0645 \u0627\u0644\u0630\u0643"
        "\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627"
        "\u0639\u064a:"
    ),
    "voice": (
        "- \u0635\u0648\u062a \u0627\u0644\u0633\u0631\u062f \u0645"
        "\u0648\u0644\u062f \u0628\u0627\u0644\u0630\u0643\u0627"
        "\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639"
        "\u064a ({provider})."
    ),
    "sketches": (
        "- \u0627\u0644\u0631\u0633\u0648\u0645\u0627\u062a \u0645"
        "\u0648\u0644\u062f\u0629 \u0628\u0646\u0645\u0627\u0630"
        "\u062c \u0630\u0643\u0627\u0621 \u0627\u0635\u0637\u0646"
        "\u0627\u0639\u064a ({models})."
    ),
    "none": (
        "- \u0644\u0627 \u064a\u0633\u062a\u062e\u062f\u0645 \u0647"
        "\u0630\u0627 \u0627\u0644\u062c\u0632\u0621 \u0635\u0648"
        "\u062a\u0627 \u0623\u0648 \u0635\u0648\u0631\u0627 \u0645"
        "\u0648\u0644\u062f\u0629 \u0628\u0627\u0644\u0630\u0643"
        "\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627"
        "\u0639\u064a."
    ),
    "recitation": (
        "- \u0627\u0644\u062a\u0644\u0627\u0648\u0629: \u062a\u0633"
        "\u062c\u064a\u0644 \u0628\u0635\u0648\u062a \u0627\u0644"
        "\u0642\u0627\u0631\u0626 {name}\u060c \u0628\u0645\u0648"
        "\u062c\u0628 {license}."
    ),
    "script": "- \u0627\u0644\u0646\u0635: {text}.",
    "script_none": ("- \u0627\u0644\u0646\u0635: \u063a\u064a\u0631 \u0645\u0633\u062c\u0644."),
    "sep": "\u060c ",
}


def _generated_models(inputs: PartInputs) -> list[str] | None:
    """Sorted unique `provider/model` of the drawn generated sketches; None when none."""
    models: set[str] = set()
    found = False
    for scene in inputs.render.scenes:
        record = inputs.sketches.get(scene.sketch_id) if scene.sketch_id is not None else None
        if record is None or not is_generated(record):
            continue
        found = True
        p = record.provenance
        if p.provider and p.model:
            models.add(f"{p.provider}/{p.model}")
        else:
            models.add(p.provider or p.model or "unrecorded")
    return sorted(models) if found else None


def disclosure_text(inputs: PartInputs) -> str:
    t = _AR if inputs.series.language is Language.ARABIC else _EN
    sep = t["sep"]
    lines = [t["header"]]
    voice = inputs.timeline.voice
    models = _generated_models(inputs)
    if voice is not None:
        lines.append(t["voice"].format(provider=voice.provider))
    if models is not None:
        lines.append(t["sketches"].format(models=sep.join(models)))
    if voice is None and models is None:
        lines.append(t["none"])
    seen: set[str] = set()
    for used in inputs.timeline.recitations:
        if used.reciter_id in seen:
            continue
        seen.add(used.reciter_id)
        lines.append(t["recitation"].format(name=used.name, license=used.license))
    source = inputs.series.source
    named = [x for x in (source.work.strip(), source.author.strip()) if x]
    if named:
        lines.append(t["script"].format(text=sep.join(named)))
    else:
        lines.append(t["script_none"])
    return "\n".join(lines) + "\n"


def platform_labels(rules: Rules) -> dict[str, str]:
    return {name: p.ai_label for name, p in rules.file.platforms.items()}


def disclosure_checks(inputs: PartInputs, text: str) -> list[CheckResult]:
    if inputs.series.ai_disclosure is False:
        return [
            CheckResult(
                check=CheckId.AI_DISCLOSURE,
                severity=Severity.FAIL,
                reason=(
                    "AI disclosure is off for this series; set "
                    '"ai_disclosure": true in the series file '
                    "(NarratoFlow videos use an AI voice and must be disclosed)"
                ),
            )
        ]
    if not text.strip():
        return [
            CheckResult(
                check=CheckId.AI_DISCLOSURE,
                severity=Severity.FAIL,
                reason="the disclosure text is blank",
            )
        ]
    return [
        CheckResult(
            check=CheckId.AI_DISCLOSURE,
            severity=Severity.PASS,
            reason="disclosure text ready; turn on each platform's AI label (see the report)",
        )
    ]
