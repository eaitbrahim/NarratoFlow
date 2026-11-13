# ADR 0004: Series IR and artifact versioning

**Status:** Accepted, 2026-11-13

## Context

The first intermediate representation was `storyboard.json`: one `Project` (schema v1) holding a
flat list of scenes for one video. Planning later needed several videos per script, so the root
became a series of parts. Stages also write their own artifacts (timelines, render reports,
stroke data) that outlive a single run and are read by later stages and by the publish check.

## Decision

- `series.json` is the IR: `Series` -> `Part` -> `Scene`, with `SCHEMA_VERSION = 2`
  (`src/narratoflow/models.py`). It supersedes `storyboard.json`. `Project`
  (`PROJECT_SCHEMA_VERSION = 1`, same file) remains only for `narratoflow validate`
  (`src/narratoflow/cli.py` loads it with `Project.load`). `Series.save` writes atomically
  (temp file, then `os.replace`); `Series.load` validates.
- Each stage artifact is versioned on its own, not by the series version:
  - `PartTimeline`: `TIMELINE_SCHEMA_VERSION = 1` (`src/narratoflow/audio/timeline.py`);
  - `RenderReport`: `REPORT_SCHEMA_VERSION = 1` (`src/narratoflow/render/report.py`);
  - publish reports: `PUBLISH_SCHEMA_VERSION = 1` (`src/narratoflow/publish/report.py`);
  - `StrokeSet`: `version = 1` (`src/narratoflow/visuals/strokes.py`).
- **Versioning rule as built:**
  - The `Series`, `Project`, timeline, render report, publish report and stroke loaders refuse any other
    `schema_version`. `Series.load` raises `PlanningError`
    "unsupported series schema_version {found}; expected {SCHEMA_VERSION}"; the timeline, render
    report and publish loaders raise the same shape ("unsupported ... schema_version ...;
    expected ..."), each naming the version found and the one expected. `StrokeSet.version` is
    constrained to exactly 1 (`Field(ge=1, le=1)`), so another value fails validation as
    `StrokeDataError` ("invalid stroke data"), without that message.
    `Project.load` (legacy `storyboard.json`, used only by `narratoflow validate`) refuses
    the same way: it raises `PlanningError` "unsupported project schema_version {found}; expected
    1", and a missing `schema_version` is refused too (found `None`). Every loader now refuses
    other versions (#148).
  - There are no migrations. Re-planning or re-building the stage is the upgrade path.
  - Adding an optional field with a default does not bump the version. Example:
    `RenderReport.inputs_key: str | None = None` was added with `REPORT_SCHEMA_VERSION` left at 1;
    older reports load with `None` and the part is simply re-rendered. Likewise
    `Part.end_card_text: str = ""` (the closing call to action of the final part) was added to
    `Series` with `SCHEMA_VERSION` left at 2; an older `series.json` loads with `""`, which
    shows no end card. The same rule holds for stroke data: `docs/strokes-schema.md` says an
    added optional field does not bump `StrokeSet.version`, a removed or required one does.
  - Removing, renaming or re-typing a field bumps the version.

## Consequences

- An old `series.json` or artifact is not read silently wrongly; the user gets a clear error and
  regenerates it. The cost is that a bump invalidates every existing artifact of that kind.
- Each stage can change its artifact format without touching the planning IR.
- `Project` and `storyboard.json` stay in the code only for `validate`; nothing else produces them.
- Anything that edits the plan must go through `Series.load` and `Series.save`. The review UI
  package (`src/narratoflow/ui/`) is a docstring-only stub on this branch, so no editor is built
  yet; when it lands it edits `series.json`.

## Alternatives considered

- **Migrations on load:** more code and more states to test for artifacts that are cheap to
  regenerate; rejected.
- **One version number for all artifacts:** every stage change would invalidate every artifact.
- **Unversioned, accept-anything loaders:** silently misread stale files.
