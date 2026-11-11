"""Rule-checked idea assignment.

Each NARRATION scene gets one drawable idea key from the LLM. The owner's drawing
rules are in `IDEAS_SYSTEM` and are enforced again here by `validate_idea`: no
person or logo, no forbidden object, and a key that parses. A rejected answer is
retried once with the violations listed; a second rejection raises `PlanningError`.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from narratoflow.errors import InvalidIdeaKeyError, PlanningError
from narratoflow.library.keys import IdeaKey, parse_idea_key
from narratoflow.llm.base import LLMClient
from narratoflow.models import Scene, SceneKind
from narratoflow.planning.prompts import IDEAS_SYSTEM

FORBIDDEN_OBJECTS = frozenset(
    {
        "man",
        "woman",
        "person",
        "people",
        "child",
        "boy",
        "girl",
        "prophet",
        "figure",
        "face",
        "silhouette",
        "angel",
        "god",
        "allah",
        "logo",
    }
)
# Words that name a person or a human figure under another name (spec 2.2a-2.2c).
FIGURE_WORDS = frozenset(
    {
        "human",
        "humans",
        "stickman",
        "stick",
        "body",
        "crowd",
        "skeleton",
        "portrait",
        "mannequin",
        "soldier",
        "soldiers",
        "king",
        "queen",
        "mother",
        "father",
        "baby",
        "warrior",
        "worshipper",
        "worshippers",
        "scholar",
        "saint",
        "statue",
        "idol",
        "profile",
        "bust",
    }
)
# Spec 2.2c also never allows heads or torsos.
_FORBIDDEN_BODY = frozenset({"head", "torso"})
# Harmless objects whose parts would otherwise match a forbidden word.
ALLOWED_COMPOUNDS = frozenset({"clock-face", "arrow-head", "figure-eight"})
# Isolated symbolic body parts that spec 2.2c allows.
_ALLOWED_BODY_WORDS = frozenset({"hand", "hands", "eye", "eyes", "footprint", "footprints"})
_LETTERS = re.compile(r"[^\W\d_]+")
_IRREGULAR_PLURALS = {"men": "man", "women": "woman", "children": "child"}
NAME_CARD = "name-card"
# Brand vocabulary: never acceptable in an object or a description, whatever the flags say.
BRAND_WORDS = frozenset({"logo", "logos", "trademark", "brand", "brands", "mascot"})
MAX_DESCRIPTION_WORDS = 15
_DASH = "\u2014"


class IdeaOut(BaseModel):
    idea_key: str
    description: str
    depicts_person: bool
    depicts_logo: bool


def _forbidden_token(token: str) -> str | None:
    if token in _ALLOWED_BODY_WORDS:
        return None
    for word in (token, _IRREGULAR_PLURALS.get(token, token), token.removesuffix("s")):
        if word in FORBIDDEN_OBJECTS or word in FIGURE_WORDS or word in _FORBIDDEN_BODY:
            return word
    return None


def _without_compounds(text: str) -> str:
    """`text` with every allowed compound (dash- or space-joined) blanked out."""
    for compound in ALLOWED_COMPOUNDS:
        text = re.sub(r"[-\s]+".join(compound.split("-")), " ", text)
    return text


def _forbidden_in(text: str) -> str | None:
    for token in _LETTERS.findall(_without_compounds(text.lower())):
        word = _forbidden_token(token)
        if word is not None:
            return word
    return None


def _forbidden_word(key: IdeaKey) -> str | None:
    return _forbidden_in(key.object)


def validate_idea(out: IdeaOut) -> list[str]:
    """Rule violations of one planner answer; empty when the idea is acceptable."""
    try:
        key = parse_idea_key(out.idea_key)
    except InvalidIdeaKeyError as exc:
        return [f"{exc}; use object|context in lowercase ASCII kebab-case"]
    if key.object == NAME_CARD:
        return []
    violations: list[str] = []
    if out.depicts_person:
        violations.append(
            f"{key}: depicts_person is true; no human figure may be drawn, use an object, "
            "a symbol or a name-card"
        )
    if out.depicts_logo:
        violations.append(f"{key}: depicts_logo is true; draw the generic object instead")
    word = _forbidden_word(key)
    if word is not None:
        violations.append(
            f"{key}: the object {word!r} is forbidden; people, prophets, angels and God are "
            "never drawn"
        )
    described = _forbidden_in(out.description)
    if described is not None:
        violations.append(
            f"{key}: the description mentions {described!r}; describe only objects and "
            "symbols, never a person or a figure"
        )
    for field, text in (("object", key.object), ("description", out.description)):
        # "logo" is already reported by the forbidden-word checks above
        words = _LETTERS.findall(text.lower())
        brand = next((w for w in words if w in BRAND_WORDS and w not in FORBIDDEN_OBJECTS), None)
        if brand is not None:
            violations.append(
                f"{key}: the {field} mentions {brand!r}; no logos or brands, "
                "draw the generic object"
            )
    return violations


def render_candidates(candidates: list[tuple[str, str]]) -> str:
    """The candidate block of an idea prompt, one `key - description` line each."""
    if not candidates:
        return "(none)"
    return "\n".join(f"{key} {_DASH} {description}" for key, description in candidates)


def _user_prompt(scene: Scene, candidates: list[tuple[str, str]]) -> str:
    return (
        f"Scene text:\n{scene.narration.source_text}\n\n"
        f"Candidate keys (key {_DASH} description):\n{render_candidates(candidates)}\n\n"
        "Reuse a candidate key if it fits (same object, same culture or period); "
        "propose a new key only when none fits."
    )


def _retry_note(violations: list[str]) -> str:
    listed = "\n".join(f"- {v}" for v in violations)
    return f"\n\nYour previous answer was rejected:\n{listed}\nFix every violation."


def _ask(llm: LLMClient, user: str) -> tuple[IdeaOut, list[str]]:
    out = llm.complete_structured(system=IDEAS_SYSTEM, user=user, output=IdeaOut).value
    return out, validate_idea(out)


def _assign(scene: Scene, candidates: list[tuple[str, str]], llm: LLMClient) -> Scene:
    user = _user_prompt(scene, candidates)
    out, violations = _ask(llm, user)
    if violations:
        out, violations = _ask(llm, user + _retry_note(violations))
        if violations:
            raise PlanningError(f"scene {scene.id}: idea rejected: " + "; ".join(violations))
    words = out.description.split()
    return scene.model_copy(
        update={
            "idea_key": str(parse_idea_key(out.idea_key)),
            "idea_description": " ".join(words[:MAX_DESCRIPTION_WORDS]),
        }
    )


def assign_ideas(
    scenes: list[Scene], candidates: list[list[tuple[str, str]]], llm: LLMClient
) -> list[Scene]:
    """Give every NARRATION scene an idea key; `candidates[i]` belongs to `scenes[i]`.

    VERSE scenes are returned unchanged: they have no idea and no sketch.
    """
    if len(candidates) != len(scenes):
        raise PlanningError(f"got {len(candidates)} candidate lists for {len(scenes)} scenes")
    return [
        scene if scene.kind is SceneKind.VERSE else _assign(scene, cands, llm)
        for scene, cands in zip(scenes, candidates, strict=True)
    ]
