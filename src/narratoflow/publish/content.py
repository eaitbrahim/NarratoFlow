"""Content-rules check: the reviewed text, the review prompt, and the WARN/FAIL mapping.

The paid review itself lives in `publish.review`; this module spends nothing.
A FAIL needs a clear violation in a category whose `max_severity` is "fail".
`religious_discussion`, unknown categories (mapped to `other`) and findings on verse
scenes are clamped to WARN here, whatever the rules file says.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from narratoflow.models import ContentFinding, ContentReview, Part, SceneKind
from narratoflow.planning.approval import verse_label
from narratoflow.planning.prompts import CONTENT_REVIEW_SYSTEM
from narratoflow.publish.checks import CheckId, CheckResult, Severity
from narratoflow.publish.rules import Rules

WARN_ONLY_CATEGORIES = frozenset({"religious_discussion", "other"})
NOT_CHECKED = "content rules not checked; run narratoflow review"
_SCRIPT_TAG = re.compile(r"<\s*(/?)\s*script", re.IGNORECASE)
MAX_CATEGORY_CHARS = 64
MAX_SCENE_ID_CHARS = 64
MAX_REASON_CHARS = 300
_CATEGORY_SEPARATORS = re.compile(r"[\s-]+")


def review_text(part: Part) -> str:
    """Everything a viewer sees or hears, one labelled line each; verses are not sent.

    Lines: hook, recap, teaser, then per scene its title, narration (or a verse
    marker), the spoken and caption texts where they differ from the narration,
    and the card text. Editing any of them makes a stored review stale.
    """
    lines = [f"[hook] {part.hook_text}"]
    if part.recap_text:
        lines.append(f"[recap] {part.recap_text}")
    if part.teaser_text:
        lines.append(f"[teaser] {part.teaser_text}")
    for scene in part.scenes:
        if scene.title:
            lines.append(f"[{scene.id} title] {scene.title}")
        if scene.kind is SceneKind.VERSE:
            label = "" if scene.verse is None else f" {verse_label(scene.verse)}"
            lines.append(f"[{scene.id}] (Qur'an{label}, recited; not reviewed)")
        else:
            narration = scene.narration
            lines.append(f"[{scene.id}] {narration.source_text}")
            if narration.tts_text is not None and narration.tts_text != narration.source_text:
                lines.append(f"[{scene.id} spoken] {narration.tts_text}")
            if (
                narration.caption_text is not None
                and narration.caption_text != narration.source_text
            ):
                lines.append(f"[{scene.id} caption] {narration.caption_text}")
        if scene.card_text:
            lines.append(f"[{scene.id} card] {scene.card_text}")
    return "\n".join(lines)


def review_sha256(part: Part) -> str:
    return hashlib.sha256(review_text(part).encode("utf-8")).hexdigest()


def _strip_format(text: str) -> str:
    """`text` without format characters (Unicode Cf, e.g. zero-width joiners), which
    could otherwise hide a script tag from the tag check."""
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def _clean(text: str, limit: int) -> str:
    """Model text made safe for a report line: whitespace collapsed, truncated."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "\u2026"


def normalize_category(category: str) -> str:
    """`" Graphic-Violence "` -> `"graphic_violence"`; empty becomes `"other"`."""
    name = _CATEGORY_SEPARATORS.sub("_", category.strip().lower()).strip("_")
    return name or "other"


def review_prompt(part: Part, rules: Rules) -> tuple[str, str]:
    """`(system, user)` of the review call. The script is untrusted data: it is
    wrapped in <script> tags; format characters are removed from it and any script
    tag inside it (any case, any spacing) is defused."""
    categories = "\n".join(
        f"- {name}: {category.description}" for name, category in rules.file.content.items()
    )
    system = CONTENT_REVIEW_SYSTEM.format(categories=categories)
    body = _SCRIPT_TAG.sub(r"&lt;\1script", _strip_format(review_text(part)))
    return system, f"<script>\n{body}\n</script>"


def _stale(part: Part, review: ContentReview, rules: Rules) -> bool:
    return review.text_sha256 != review_sha256(part) or review.rules_version != rules.file.version


def _severity(
    finding: ContentFinding, name: str, scene_id: str | None, rules: Rules, verse_ids: set[str]
) -> Severity:
    content = rules.file.content
    category = content.get(name, content["other"])
    if (
        finding.clear_violation
        and category.max_severity == "fail"
        and name in content
        and name not in WARN_ONLY_CATEGORIES
        and scene_id not in verse_ids
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
    results: list[CheckResult] = []
    for f in review.findings:
        name = normalize_category(f.category)
        raw_id = None if f.scene_id is None else _clean(f.scene_id, MAX_SCENE_ID_CHARS)
        scene_id = raw_id if raw_id in scene_ids else None
        results.append(
            CheckResult(
                check=CheckId.CONTENT_RULES,
                severity=_severity(f, name, scene_id, rules, verse_ids),
                reason=(
                    f"{_clean(name, MAX_CATEGORY_CHARS)}: {_clean(f.reason, MAX_REASON_CHARS)}"
                ),
                scene_id=scene_id,
            )
        )
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
