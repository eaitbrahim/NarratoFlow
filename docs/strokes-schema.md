# Stroke data schema (version 1)

`narratoflow.visuals.strokes.StrokeSet` is the vector form of a sketch. The
renderer replays it as a draw-on animation. It is produced by
`extract_strokes` and stored next to each sketch as JSON.

## Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `version` | int | Always `1`. Any other value is refused. |
| `width`, `height` | int > 0 | Pixels of the source raster the strokes were extracted from. |
| `strokes` | list | Strokes in drawing order. May be empty. |

Each stroke has:

| Field | Type | Meaning |
| --- | --- | --- |
| `points` | list of `[x, y]` | At least 2 points, in drawing order. Each coordinate is normalized to [0, 1]: x is the column divided by `width`, y is the row divided by `height`. |
| `width` | float in (0, 1] | Present only for dot "taps" (2-point strokes at one spot). The dot diameter divided by the image `width`. Absent for ordinary lines. |

Unknown fields are rejected.

## Serialization

`StrokeSet.to_json()` writes compact JSON (no spaces), keys sorted, UTF-8, and
omits absent optional fields. `StrokeSet.from_json()` raises `StrokeDataError`
on bad JSON, a bad value or an unknown field.

## Ordering

The order is part of the data, since it is the order the pen draws.

1. The pen starts at the top-left corner (top-right for right-to-left
   sketches). Each step takes the stroke with an end nearest the pen, reversed
   when its far end was the nearer one. So the first stroke is drawn from the
   end nearest the corner. Ties go to the lowest (y, x).
2. A stroke that starts within 2 px (at 512 px, scaled with the image) of the
   pen's end, across a paper gap no wider than the local line width, is joined
   to the current stroke so the pen does not lift. Dots are never chained.

## Example

A line and a dot, on a 200 x 120 raster:

```json
{"height":120,"strokes":[{"points":[[0.1,0.2],[0.5,0.2],[0.9,0.5]]},{"points":[[0.5,0.5],[0.5,0.5]],"width":0.05}],"version":1,"width":200}
```

## Versioning

The loader accepts exactly `version` 1. Adding an optional field that older
readers can ignore does not bump it. Changing the meaning of a field, removing
one, or making an optional field required does: bump `version`, update
`StrokeSet.version` bounds, and regenerate the golden
(`uv run python tests/visuals/test_strokes_golden.py --update`) as a reviewed
diff. There are no migrations; a sketch with another version is re-extracted.
