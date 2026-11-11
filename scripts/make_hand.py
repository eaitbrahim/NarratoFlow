"""Draw the hand sprite into assets/hands/ (PIL only). Run manually:

    uv run python scripts/make_hand.py

A right hand from the wrist down gripping a marker, as fixed polygons: ink #111111 outline
(open at the wrist, where the hand leaves the image), white fill, grey marker. 320x440 RGBA,
marker tip at the top-left (tip pixel printed and written to the index). No randomness.
NarratoFlow's own work, Apache-2.0.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parents[1] / "assets" / "hands"
SIZE = (320, 440)
TIP = (24, 24)
SS = 4  # supersampling factor
INK = (17, 17, 17, 255)
WHITE = (255, 255, 255, 255)
GREY = (136, 136, 136, 255)
LINE = 3

Poly = list[tuple[int, int]]
NIB: Poly = [(24, 24), (42, 27), (27, 42)]
BODY: Poly = [(42, 27), (27, 42), (196, 214), (214, 196)]
CAP: Poly = [(190, 208), (208, 190), (232, 214), (214, 232)]
PALM: Poly = [(150, 236), (232, 176), (300, 204), (320, 268), (320, 440), (176, 440), (160, 340)]
FINGERS: Poly = [(150, 236), (190, 204), (250, 232), (240, 262), (190, 270), (160, 300)]
THUMB: Poly = [(176, 270), (212, 238), (232, 262), (200, 304), (170, 310)]


def _scaled(poly: Poly) -> Poly:
    return [(x * SS, y * SS) for x, y in poly]


def draw() -> Image.Image:
    big = Image.new("RGBA", (SIZE[0] * SS, SIZE[1] * SS), (0, 0, 0, 0))
    pen = ImageDraw.Draw(big)
    for poly, fill in ((BODY, GREY), (NIB, INK), (CAP, GREY), (PALM, WHITE), (FINGERS, WHITE)):
        pts = _scaled(poly)
        pen.polygon(pts, fill=fill)
        # The palm is open at the wrist: the hand runs out of view, so no edge along the
        # image's right and bottom borders (PALM's 4th to 6th points lie on them).
        outline = [*pts[5:], *pts[:4]] if poly is PALM else [*pts, pts[0]]
        pen.line(outline, fill=INK, width=LINE * SS, joint="curve")
    pts = _scaled(THUMB)
    pen.polygon(pts, fill=WHITE)
    pen.line([*pts, pts[0]], fill=INK, width=LINE * SS, joint="curve")
    return big.resize(SIZE, Image.Resampling.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "hand-right.png"
    draw().save(path, optimize=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "index.toml").write_text(
        f'[hands.right]\nfile = "hand-right.png"\nlicense = "Apache-2.0"\n'
        f'source = "scripts/make_hand.py"\ntip_x = {TIP[0]}\ntip_y = {TIP[1]}\n'
        f'sha256 = "{sha}"\n',
        encoding="utf-8",
    )
    print(f"tip={TIP} sha256={sha}")


if __name__ == "__main__":
    main()
