"""Deterministic PIL drawings used as image fixtures (no binary blobs in git)."""

import io
import math
import random

from PIL import Image, ImageDraw

SIZE = 512


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_scroll(
    draw: ImageDraw.ImageDraw, ink: tuple[int, ...], *, scale: float = 1.0, width: int = 3
) -> None:
    def box(*coords: int) -> tuple[int, ...]:
        return tuple(int(c * scale) for c in coords)

    draw.rectangle(box(120, 150, 392, 362), outline=ink, width=width)
    draw.ellipse(box(90, 150, 150, 210), outline=ink, width=width)
    draw.ellipse(box(362, 302, 422, 362), outline=ink, width=width)
    for y in (200, 250, 300):
        draw.line(box(170, y, 340, y), fill=ink, width=width)


def line_drawing_scroll() -> bytes:
    image = Image.new("RGB", (SIZE, SIZE), "white")
    _draw_scroll(ImageDraw.Draw(image), (0, 0, 0))
    return _png(image)


def line_drawing_scroll_rgba() -> bytes:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    _draw_scroll(ImageDraw.Draw(image), (0, 0, 0, 255))
    return _png(image)


def filled_silhouette() -> bytes:
    image = Image.new("RGB", (SIZE, SIZE), "white")
    # semi-axes 178 x 142: pi * a * b is about 30% of 512 * 512
    ImageDraw.Draw(image).ellipse((78, 114, 434, 398), fill="black")
    return _png(image)


def crosshatched_box() -> bytes:
    image = Image.new("RGB", (SIZE, SIZE), "white")
    # 2 px lines every 6 px (horizontal offset), the original pinned fixture
    size = 300
    box = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(box)
    for offset in range(-size, size + 1, 6):
        draw.line((offset, 0, offset + size, size), fill="black", width=2)
        draw.line((offset + size, 0, offset, size), fill="black", width=2)
    image.paste(box, ((SIZE - size) // 2, (SIZE - size) // 2))
    return _png(image)


def noisy_line_drawing() -> bytes:
    image = Image.new("RGB", (SIZE, SIZE), "white")
    _draw_scroll(ImageDraw.Draw(image), (0, 0, 0))
    rng = random.Random(0)  # noqa: S311 - deterministic fixture noise
    for _ in range(400):
        image.putpixel((rng.randrange(SIZE), rng.randrange(SIZE)), (0, 0, 0))
    return _png(image)


def scroll(*, size: int = SIZE, width: int = 3) -> bytes:
    """The scroll outline at any resolution and stroke width."""
    image = Image.new("RGB", (size, size), "white")
    _draw_scroll(ImageDraw.Draw(image), (0, 0, 0), scale=size / SIZE, width=width)
    return _png(image)


def hatch(
    *, spacing: int, width: int, angles: tuple[int, ...], size: int = SIZE, box: int = 300
) -> bytes:
    """A centred box of parallel line families; spacing is perpendicular, in px.

    angles=(45, 135) is a 45 degree crosshatch, (0, 90) a grid, (45,) parallel hatching.
    """
    image = Image.new("RGB", (size, size), "white")
    patch = Image.new("RGB", (box, box), "white")
    draw = ImageDraw.Draw(patch)
    centre = box / 2
    for angle in angles:
        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        for k in range(-box // spacing - 1, box // spacing + 2):
            px, py = centre - dy * k * spacing, centre + dx * k * spacing
            draw.line(
                (px - dx * box, py - dy * box, px + dx * box, py + dy * box),
                fill="black",
                width=width,
            )
    image.paste(patch, ((size - box) // 2, (size - box) // 2))
    return _png(image)


def silhouette(coverage: float) -> bytes:
    """A solid black disc covering `coverage` of the image."""
    image = Image.new("RGB", (SIZE, SIZE), "white")
    r = math.sqrt(coverage * SIZE * SIZE / math.pi)
    c = SIZE / 2
    ImageDraw.Draw(image).ellipse((c - r, c - r, c + r, c + r), fill="black")
    return _png(image)


def road_map(*, roads: int = 40, width: int = 3, size: int = SIZE, seed: int = 1) -> bytes:
    """A busy but legitimate street map: near-axis streets plus curved polylines."""
    rng = random.Random(seed)  # noqa: S311 - deterministic fixture
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for _ in range(roads // 4):
        y = rng.randrange(size)
        draw.line((0, y, size, y + rng.randrange(-40, 40)), fill="black", width=width)
        x = rng.randrange(size)
        draw.line((x, 0, x + rng.randrange(-40, 40), size), fill="black", width=width)
    for _ in range(roads // 2):
        points = [(rng.randrange(size), rng.randrange(size)) for _ in range(4)]
        draw.line(points, fill="black", width=width, joint="curve")
    return _png(image)


def label_map(*, seed: int = 7, size: int = SIZE) -> bytes:
    """A map with a dozen roads and forty small label-like boxes."""
    rng = random.Random(seed)  # noqa: S311 - deterministic fixture
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    k = size / SIZE
    for _ in range(12):
        points = [(rng.randrange(SIZE) * k, rng.randrange(SIZE) * k) for _ in range(4)]
        draw.line(points, fill="black", width=max(1, round(3 * k)), joint="curve")
    for _ in range(40):
        x, y = rng.randrange(SIZE - 40), rng.randrange(SIZE - 16)
        w, h = rng.randrange(20, 40), rng.randrange(8, 16)
        box = (x * k, y * k, (x + w) * k, (y + h) * k)
        draw.rectangle(box, outline="black", width=max(1, round(2 * k)))
    return _png(image)


def blank() -> bytes:
    return _png(Image.new("RGB", (SIZE, SIZE), "white"))


# Legitimate single-direction rulings and pen dots (re-review round 2). Coordinates
# are on a 512 grid and scaled to `size`; the stroke width scales too.


def _canvas(size: int) -> tuple[Image.Image, ImageDraw.ImageDraw, float, int]:
    image = Image.new("RGB", (size, size), "white")
    k = size / SIZE
    return image, ImageDraw.Draw(image), k, max(1, round(3 * k))


def notebook(*, spacing: int, size: int = SIZE) -> bytes:
    """A ruled notebook page with a margin line."""
    image, draw, k, w = _canvas(size)
    x0, y0, x1, y1 = int(76 * k), int(40 * k), int(436 * k), int(472 * k)
    draw.rectangle((x0, y0, x1, y1), outline="black", width=w)
    y = y0 + 3 * spacing * k
    while y < y1 - spacing * k / 2:
        draw.line((x0, int(y), x1, int(y)), fill="black", width=w)
        y += spacing * k
    draw.line((x0 + int(40 * k), y0, x0 + int(40 * k), y1), fill="black", width=w)
    return _png(image)


def scroll_text(*, spacing: int, size: int = SIZE) -> bytes:
    """An unrolled scroll covered in wavy handwriting-like text lines."""
    image, draw, k, w = _canvas(size)
    x0, y0, x1, y1 = int(110 * k), int(70 * k), int(402 * k), int(442 * k)
    draw.rectangle((x0, y0, x1, y1), outline="black", width=w)
    for yc in (y0, y1):
        box = (x0 - int(30 * k), yc - int(25 * k), x1 + int(30 * k), yc + int(25 * k))
        draw.ellipse(box, outline="black", width=w)
    rng = random.Random(3)  # noqa: S311 - deterministic fixture
    y = y0 + 40 * k
    while y < y1 - 30 * k:
        x = x0 + 20 * k
        while x < x1 - 30 * k:
            length = rng.uniform(20, 60) * k
            points = [(x + t, y + 2 * k * math.sin(t / (4 * k))) for t in range(0, int(length), 2)]
            draw.line(points, fill="black", width=w)
            x += length + 10 * k
        y += spacing * k
    return _png(image)


def brick(*, brick_w: int, brick_h: int, size: int = SIZE) -> bytes:
    """A running-bond brick wall with a window above it."""
    image, draw, k, w = _canvas(size)
    x0, y0, x1, y1 = int(56 * k), int(96 * k), int(456 * k), int(456 * k)
    bw, bh = int(brick_w * k), int(brick_h * k)
    draw.rectangle((x0, y0, x1, y1), outline="black", width=w)
    for row, y in enumerate(range(y0, y1, bh)):
        draw.line((x0, y, x1, y), fill="black", width=w)
        for x in range(x0 + (bw // 2 if row % 2 else 0), x1, bw):
            draw.line((x, y, x, min(y + bh, y1)), fill="black", width=w)
    draw.rectangle((int(200 * k), int(40 * k), int(312 * k), int(96 * k)), outline="black", width=w)
    return _png(image)


def fence(*, gap: int, size: int = SIZE) -> bytes:
    """A picket fence: pointed vertical pickets, two rails and the ground."""
    image, draw, k, w = _canvas(size)
    x = 40 * k
    while x < 472 * k:
        xi = int(x)
        draw.line((xi, int(200 * k), xi, int(440 * k)), fill="black", width=w)
        draw.line((xi, int(200 * k), xi + int(gap * k / 2), int(190 * k)), fill="black", width=w)
        x += gap * k
    for y in (250, 400):
        draw.line((int(40 * k), int(y * k), int(472 * k), int(y * k)), fill="black", width=w)
    draw.line((0, int(440 * k), size, int(440 * k)), fill="black", width=w)
    return _png(image)


def blinds(*, slat: int, size: int = SIZE) -> bytes:
    """Window blinds: a frame of horizontal slats and a pull cord."""
    image, draw, k, w = _canvas(size)
    x0, y0, x1, y1 = int(110 * k), int(60 * k), int(402 * k), int(452 * k)
    draw.rectangle((x0, y0, x1, y1), outline="black", width=w)
    y = y0 + slat * k
    while y < y1:
        draw.line((x0, int(y), x1, int(y)), fill="black", width=w)
        y += slat * k
    draw.line((int(256 * k), y0, int(256 * k), y1), fill="black", width=1)
    return _png(image)


def ladder(*, rung: int, size: int = SIZE) -> bytes:
    image, draw, k, w = _canvas(size)
    for x in (200, 312):
        draw.line((int(x * k), int(30 * k), int(x * k), int(490 * k)), fill="black", width=w)
    y = 40 * k
    while y < 490 * k:
        draw.line((int(200 * k), int(y), int(312 * k), int(y)), fill="black", width=w)
        y += rung * k
    return _png(image)


def stripes(*, period: int, size: int = SIZE) -> bytes:
    """Barcode-like vertical stripes."""
    image, draw, k, w = _canvas(size)
    x = 100 * k
    while x < 412 * k:
        draw.line((int(x), int(150 * k), int(x), int(360 * k)), fill="black", width=w)
        x += period * k
    return _png(image)


def scene_with_shadow() -> bytes:
    """Outlined objects plus a 160 px crosshatched shadow (2 px lines every 7 px)."""
    image, draw, _, w = _canvas(SIZE)
    draw.rectangle((60, 60, 300, 300), outline="black", width=w)
    draw.ellipse((320, 60, 470, 210), outline="black", width=w)
    patch = Image.new("RGB", (160, 160), "white")
    pd = ImageDraw.Draw(patch)
    for o in range(-160, 321, 7):
        pd.line((o, 0, o + 160, 160), fill="black", width=2)
        pd.line((o + 160, 0, o, 160), fill="black", width=2)
    image.paste(patch, (300, 300))
    return _png(image)


PEN_DOT_FACTORS = (1.0, 1.5, 2.0, 3.0)


def pen_dots(*, size: int = SIZE) -> bytes:
    """A frame plus solid dots of 1, 1.5, 2 and 3 line weights (i-dots, nuqta)."""
    image, draw, _, w = _canvas(size)
    draw.rectangle((size // 4, size // 4, 3 * size // 4, 3 * size // 4), outline="black", width=w)
    for n, factor in enumerate(PEN_DOT_FACTORS):
        diameter = int(w * factor)
        x, y = size // 3 + n * size // 10, size // 2
        draw.ellipse((x, y, x + diameter, y + diameter), fill="black")
    return _png(image)


def calligraphy(*, lines: int = 7, size: int = SIZE, seed: int = 4) -> bytes:
    """Right-to-left cursive-like rows: wavy runs, small loops and ascenders."""
    rng = random.Random(seed)  # noqa: S311 - deterministic fixture
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    k = size / SIZE
    w = max(1, int(3 * k))
    for row in range(lines):
        baseline = (90 + row * 55) * k
        x = 460 * k
        while x > 50 * k:
            run = rng.uniform(25, 70) * k
            points = []
            for t in range(int(run)):
                sign = 1 if t % 40 < 30 else -1
                wave = 6 * k * math.sin(t / (3 * k) + rng.random() * 0.2) * sign
                points.append((x - t, baseline + wave))
            draw.line(points, fill="black", width=w)
            for _ in range(rng.randint(1, 3)):
                cx, cy = x - rng.uniform(0, run), baseline - rng.uniform(8, 25) * k
                r = rng.uniform(3, 8) * k
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="black", width=w)
            for _ in range(rng.randint(0, 2)):
                cx = x - rng.uniform(0, run)
                draw.line((cx, baseline, cx + 3 * k, baseline - 30 * k), fill="black", width=w)
            x -= run + rng.uniform(6, 16) * k
    return _png(image)
