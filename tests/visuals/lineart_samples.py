"""The XDoG sample set: images bundled in the installed scikit-image wheel.

No image binary is committed. Each licence and source below is the one stated in
the docstrings of `skimage/data/_fetchers.py` (scikit-image 0.26).
"""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sample:
    name: str
    licence: str
    source: str


SAMPLES: tuple[Sample, ...] = (
    Sample("camera", "CC0-1.0", "Lav Varshney, via scikit-image data.camera"),
    Sample("horse", "CC0-1.0", "openclipart, Andreas Preuss, via scikit-image data.horse"),
    Sample("coffee", "CC0-1.0", "Rachel Michetti, via scikit-image data.coffee"),
    Sample("chelsea", "CC0-1.0", "Stefan van der Walt, via scikit-image data.chelsea"),
    Sample("astronaut", "public-domain", "NASA, via scikit-image data.astronaut"),
    Sample("rocket", "public-domain", "SpaceX release, via scikit-image data.rocket"),
    Sample("text", "public-domain", "Wikipedia Corner.png, via scikit-image data.text"),
)


def sample_png(name: str) -> bytes:
    """The named sample as PNG bytes, encoded in memory."""
    from skimage import data

    array = np.asarray(getattr(data, name)())
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()
