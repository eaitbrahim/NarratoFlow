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


def label_map(*, seed: int = 7) -> bytes:
    """A map with a dozen roads and forty small label-like boxes."""
    rng = random.Random(seed)  # noqa: S311 - deterministic fixture
    image = Image.new("RGB", (SIZE, SIZE), "white")
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        points = [(rng.randrange(SIZE), rng.randrange(SIZE)) for _ in range(4)]
        draw.line(points, fill="black", width=3, joint="curve")
    for _ in range(40):
        x, y = rng.randrange(SIZE - 40), rng.randrange(SIZE - 16)
        w, h = rng.randrange(20, 40), rng.randrange(8, 16)
        draw.rectangle((x, y, x + w, y + h), outline="black", width=2)
    return _png(image)


def blank() -> bytes:
    return _png(Image.new("RGB", (SIZE, SIZE), "white"))
