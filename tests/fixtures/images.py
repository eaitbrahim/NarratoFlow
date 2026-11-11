"""Deterministic PIL drawings used as image fixtures (no binary blobs in git)."""

import io
import random

from PIL import Image, ImageDraw

SIZE = 512


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_scroll(draw: ImageDraw.ImageDraw, ink: tuple[int, ...]) -> None:
    draw.rectangle((120, 150, 392, 362), outline=ink, width=3)
    draw.ellipse((90, 150, 150, 210), outline=ink, width=3)
    draw.ellipse((362, 302, 422, 362), outline=ink, width=3)
    for y in (200, 250, 300):
        draw.line((170, y, 340, y), fill=ink, width=3)


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
    # 2 px lines every 6 px: 1 px lines would not survive the 3x3 open
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


def blank() -> bytes:
    return _png(Image.new("RGB", (SIZE, SIZE), "white"))
