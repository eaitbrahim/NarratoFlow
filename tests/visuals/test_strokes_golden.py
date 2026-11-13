"""Golden stroke JSON for a shipped icon.

Regenerate only as a reviewed diff, never in CI:
    uv run python tests/visuals/test_strokes_golden.py --update
"""

import sys
from pathlib import Path

from narratoflow.library.keys import parse_idea_key
from narratoflow.visuals.bundled import default_bundled_icons
from narratoflow.visuals.lineart import binarize
from narratoflow.visuals.strokes import StrokeSet, extract_strokes

KEY = "lightbulb|generic"
GOLDEN = Path(__file__).parent / "golden" / "lightbulb-generic.strokes.json"
TOLERANCE = 1e-6


def extract_icon(key: str) -> StrokeSet:
    icon = default_bundled_icons().get(parse_idea_key(key))
    if icon is None:
        raise LookupError(f"no bundled icon for {key!r}")
    return extract_strokes(binarize(icon.file.read_bytes()), rtl=False)


def _assert_same(got: StrokeSet, want: StrokeSet) -> None:
    assert (got.width, got.height) == (want.width, want.height), "image size differs"
    assert len(got.strokes) == len(want.strokes), (
        f"stroke count {len(got.strokes)} != {len(want.strokes)}"
    )
    for i, (g, w) in enumerate(zip(got.strokes, want.strokes, strict=True)):
        assert len(g.points) == len(w.points), (
            f"stroke {i}: {len(g.points)} points != {len(w.points)}"
        )
        assert (g.width is None) == (w.width is None), f"stroke {i}: dot width presence differs"
        if g.width is not None and w.width is not None:
            assert abs(g.width - w.width) <= TOLERANCE, f"stroke {i}: dot width differs"
        for j, (gp, wp) in enumerate(zip(g.points, w.points, strict=True)):
            assert abs(gp[0] - wp[0]) <= TOLERANCE and abs(gp[1] - wp[1]) <= TOLERANCE, (
                f"stroke {i} point {j}: {gp} != {wp}"
            )


def test_lightbulb_strokes_match_golden() -> None:
    want = StrokeSet.from_json(GOLDEN.read_bytes())
    _assert_same(extract_icon(KEY), want)


def test_golden_is_valid_schema() -> None:
    assert StrokeSet.from_json(GOLDEN.read_bytes()).version == 1


if __name__ == "__main__":
    if sys.argv[1:] != ["--update"]:
        raise SystemExit("usage: test_strokes_golden.py --update")
    GOLDEN.write_bytes(extract_icon(KEY).to_json())
    print(f"wrote {GOLDEN}")
