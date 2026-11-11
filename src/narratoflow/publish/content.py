"""Content-rules check: the reviewed text, the review prompt, and the WARN/FAIL mapping.

The paid review itself lives in `publish.review`; this module spends nothing.
A FAIL needs a clear violation in a category whose `max_severity` is "fail".
`religious_discussion`, unknown categories (mapped to `other`) and findings on verse
scenes are clamped to WARN here, whatever the rules file says.
"""

from __future__ import annotations

import hashlib
import re

from narratoflow.models import ContentFinding, ContentReview, Part, SceneKind
from narratoflow.planning.approval import verse_label
from narratoflow.planning.prompts import CONTENT_REVIEW_SYSTEM
from narratoflow.publish.checks import CheckId, CheckResult, Severity
from narratoflow.publish.rules import Rules

WARN_ONLY_CATEGORIES = frozenset({"religious_discussion", "other"})
NOT_CHECKED = "content rules not checked; run narratoflow review"
_SCRIPT_TAG = re.compile(r"<(/?script)", re.IGNORECASE)


def review_text(part: Part) -> str:
    """The text the review judges, one bracketed line each; verses are not sent."""
    lines = [f"[hook] {part.hook_text}"]
    if part.recap_text:
        lines.append(f"[recap] {part.recap_text}")
    if part.teaser_text:
        lines.append(f"[teaser] {part.teaser_text}")
    for scene in part.scenes:
        if scene.kind is SceneKind.VERSE:
            label = "" if scene.verse is None else f" {verse_label(scene.verse)}"
            lines.append(f"[{scene.id}] (Qur'an{label}, recited; not reviewed)")
        else:
            lines.append(f"[{scene.id}] {scene.narration.source_text}")
        if scene.card_text:
            lines.append(f"[{scene.id} card] {scene.card_text}")
    return "\n".join(lines)


def review_sha256(part: Part) -> str:
    return hashlib.sha256(review_text(part).encode("utf-8")).hexdigest()


def review_prompt(part: Part, rules: Rules) -> tuple[str, str]:
    """`(system, user)` of the review call. The script is untrusted data: it is
    wrapped in <script> tags, and any script tag inside it is defused."""
    categories = "\n".join(
        f"- {name}: {category.description}" for name, category in rules.file.content.items()
    )
    system = CONTENT_REVIEW_SYSTEM.format(categories=categories)
    body = _SCRIPT_TAG.sub(r"&lt;\1", review_text(part))
    return system, f"<script>\n{body}\n</script>"


def _stale(part: Part, review: ContentReview, rules: Rules) -> bool:
    return review.text_sha256 != review_sha256(part) or review.rules_version != rules.file.version


def _severity(finding: ContentFinding, rules: Rules, verse_ids: set[str]) -> Severity:
    content = rules.file.content
    category = content.get(finding.category, content["other"])
    if (
        finding.clear_violation
        and category.max_severity == "fail"
        and finding.category in content
        and finding.category not in WARN_ONLY_CATEGORIES
        and finding.scene_id not in verse_ids
    ):
        return Severity.FAIL
    return Severity.WARN


def content_checks(part: Part, rules: Rules) -> list[CheckResult]:
    """`CONTENT_RULES` results for `part` from its stored review."""
    review = part.content_review
    if review is None:
        return [
            CheckResult(check=CheckId.CONTENT_RULES, severity=Severity.WARN, reason=NOT_CHECKED)
        ]
    if _stale(part, review, rules):
        return [
            CheckResult(
                check=CheckId.CONTENT_RULES,
                severity=Severity.WARN,
                reason=(
                    "content review is stale (the part or the platform rules changed since "
                    f"{review.reviewed_at.date().isoformat()}); run narratoflow review"
                ),
            )
        ]
    scene_ids = {s.id for s in part.scenes}
    verse_ids = {s.id for s in part.scenes if s.kind is SceneKind.VERSE}
    results = [
        CheckResult(
            check=CheckId.CONTENT_RULES,
            severity=_severity(f, rules, verse_ids),
            reason=f"{f.category}: {f.reason}",
            scene_id=f.scene_id if f.scene_id in scene_ids else None,
        )
        for f in review.findings
    ]
    if results:
        return results
    return [
        CheckResult(
            check=CheckId.CONTENT_RULES,
            severity=Severity.PASS,
            reason=(
                f"no content-rule issues found by {review.provider}/{review.model} "
                f"(rules {review.rules_version})"
            ),
        )
    ]
