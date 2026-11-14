"""Raster -> binary ink mask and the line-art density check.

`xdog_lineart` turns an uploaded photo or diagram into the same ink mask that
`binarize` returns, so the density check and stroke extraction take it unchanged.
Pipeline: decode (alpha over white), resize the long side to XDOG_LONG_SIDE_PX,
stretch the 1st to 99th percentile to [0, 1], then the XDoG filter
D = G(sigma) - tau * G(k * sigma); T = 1 where D >= eps, else
1 + tanh(phi * (D - eps)); ink is T < 0.5. Measured at XdogParams(1.5, 1.6, 1.0,
-0.03, 200.0) on scikit-image's bundled samples (strokes / ink ratio / mean
stroke width); all pass `check_density`:
#   camera    127  0.040  3.3     horse       9  0.050  4.4
#   coffee     91  0.025  3.0     chelsea   244  0.038  3.4
#   astronaut 278  0.068  3.1     rocket    123  0.018  2.7
#   text       84  0.106  3.6
The classic (1 + p) G1 - p G2 tone form at eps 0.6 filled dark regions (ink ratio
0.43 to 0.94) and failed density on every sample, so it is not used.
"""

from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.morphology import skeletonize

from narratoflow.errors import SketchRejectedError

XDOG_LONG_SIDE_PX = 512
XDOG_PERCENTILES = (1.0, 99.0)
MIN_UPLOAD_SIDE_PX = 32
_EXIF_ORIENTATION = 0x0112
MAX_UPLOAD_SIDE_PX = 8192
_FLAT_GREY_LEVELS = 1.0
_PAPER = 255

# Density gate. A FAIL has no override, so the gate is biased against false
# rejects: a missed hatch only costs one paid vision call, whose outlier check
# is the second line of defence.
#   1. stroke_width: mean_stroke_width = ink_px / skeleton_px. Primary fill test;
#      a filled shape has a tiny skeleton for its ink. Limit
#      max(MIN_FILL_WIDTH_PX, FILL_WIDTH_FACTOR * line_weight_px).
#   2. ink_ratio: loose global backstop (busiest legit fixture 0.240).
#   3. dense_tiles: crosshatch. 8x8 tile grid; the largest 4-connected block of
#      tiles with ink > DENSE_TILE_INK must have >= MIN_DENSE_TILES tiles and its
#      spectrum must show all three of:
#      - periodicity (energy share of the SPECTRAL_PEAK_BINS strongest bins)
#        >= MIN_BLOCK_PERIODICITY: hatch >= 0.37, busy maps <= 0.10;
#      - fundamental period <= MAX_HATCH_PERIOD_PX (scaled to resolution), so
#        sparse lattices and chart grids (spacing >= 12 at 512) stay legal;
#      - second_orientation_ratio >= MIN_SECOND_ORIENTATION: a second line family
#        at least 30 degrees from the first, at hatch frequencies. Crosshatch and
#        grids score >= 0.75; ruled pages, fences, blinds, ladders, stripes and
#        brick courses score <= 0.21 and stay legal at any spacing. Single-
#        direction parallel hatching is therefore left to the vision check.
#      Tile ink alone cannot work (a 40-road map has 43-49 dense tiles at up to
#      0.56 ink); Sobel orientation coherence overlapped (aliasing on 45 degrees).
# junction_clusters_per_kpx is informational only.
#
# Calibration (tests/fixtures/images.py; 512 px unless noted; hatch spacing is
# perpendicular, box 300 px; expect "none" = no assertion either way). Columns:
# ink_ratio, stroke_w, max tile ink, dense block, periodicity, period px,
# second orientation, junction clusters per kpx, verdict.
#   fixture                expect ink   stroke_w tile blk per  period 2nd  jc/kpx verdict
#   cross45 w2 s6          reject 0.200     2.2 0.58  32 0.41   6.0 1.00 104.5  dense_tiles
#   cross45 w2 s7          reject 0.176     2.3 0.52  32 0.39   7.0 1.00  92.0  dense_tiles
#   cross45 w2 s8          reject 0.158     2.4 0.47  32 0.37   8.0 1.00  80.5  dense_tiles
#   cross45 w2 s9          reject 0.143     2.4 0.42  16 0.40   9.1 1.00  71.8  dense_tiles
#   cross45 w2 s10         reject 0.130     2.5 0.38  16 0.38  10.1 1.00  66.4  dense_tiles
#   cross45 w3 s6          reject 0.285     3.1 0.83  32 0.42   6.0 1.00 103.1  dense_tiles
#   cross45 w3 s7          reject 0.259     3.4 0.76  32 0.44   7.0 1.00  91.6  dense_tiles
#   cross45 w3 s8          reject 0.237     3.5 0.70  32 0.42   8.0 1.00  79.9  dense_tiles
#   cross45 w3 s9          reject 0.217     3.7 0.63  32 0.44   9.1 1.00  71.1  dense_tiles
#   cross45 w3 s10         reject 0.199     3.8 0.58  32 0.42  10.1 1.00  66.1  dense_tiles
#   grid w2 s6             reject 0.191     1.8 0.57  32 0.43   6.0 0.75  91.0  dense_tiles
#   grid w2 s7             reject 0.169     1.9 0.51  32 0.42   7.0 0.80  77.9  dense_tiles
#   grid w2 s8             reject 0.149     1.9 0.44  24 0.41   8.0 0.88  66.4  dense_tiles
#   grid w2 s9             reject 0.134     1.9 0.41  16 0.44   9.1 0.89  58.4  dense_tiles
#   grid w2 s10            reject 0.124     1.9 0.39  16 0.41  10.2 0.90  52.7  dense_tiles
#   grid w3 s6             reject 0.257     2.4 0.77  32 0.46   6.0 0.75  93.4  dense_tiles
#   grid w3 s7             reject 0.232     2.6 0.68  32 0.46   7.0 0.80  77.9  dense_tiles
#   grid w3 s8             reject 0.207     2.6 0.61  32 0.43   8.0 0.85  66.5  dense_tiles
#   grid w3 s9             reject 0.189     2.7 0.57  32 0.44   9.1 0.85  58.5  dense_tiles
#   grid w3 s10            reject 0.175     2.6 0.53  32 0.43  10.1 0.93  54.6  dense_tiles
#   parallel0 w2 s6        none   0.114     2.0 0.34  16 0.87   6.1 0.00   0.0  ok
#   parallel0 w2 s7        none   0.098     2.0 0.30  16 0.81   7.1 0.00   0.0  ok
#   parallel0 w2 s8        none   0.085     2.0 0.25  16 0.73   8.0 0.00   0.0  ok
#   parallel0 w3 s6        none   0.172     3.0 0.52  32 0.83   6.0 0.00   0.0  ok
#   parallel0 w3 s7        none   0.148     3.0 0.44  22 0.86   7.1 0.00   0.0  ok
#   parallel0 w3 s8        none   0.127     3.0 0.38  16 0.81   8.0 0.00   0.0  ok
#   parallel45 w2 s6       none   0.122     3.0 0.36  16 0.71   6.0 0.04   0.0  ok
#   parallel45 w2 s7       none   0.104     3.0 0.31  16 0.71   7.0 0.02   0.0  ok
#   parallel45 w2 s8       none   0.091     3.0 0.27  16 0.73   8.2 0.02   0.0  ok
#   parallel45 w3 s6       none   0.203     4.9 0.60  32 0.69   6.0 0.03   0.0  ok
#   parallel45 w3 s7       none   0.174     4.9 0.51  32 0.73   7.0 0.02   0.0  ok
#   parallel45 w3 s8       none   0.152     4.9 0.45  32 0.73   8.0 0.01   0.0  ok
#   parallel30 w2 s6       none   0.093     1.9 0.31  16 0.61   6.0 0.02   0.0  ok
#   parallel30 w2 s7       none   0.080     1.9 0.27  16 0.66   7.0 0.03   0.0  ok
#   parallel30 w2 s8       none   0.070     1.9 0.23  16 0.49   7.9 0.01   0.0  ok
#   parallel30 w3 s6       none   0.157     3.1 0.54  24 0.68   6.0 0.02   0.0  ok
#   parallel30 w3 s7       none   0.135     3.1 0.47  23 0.71   7.1 0.02   0.0  ok
#   parallel30 w3 s8       none   0.118     3.1 0.41  16 0.65   7.9 0.01   0.0  ok
#   scene + shadow xhatch s7 reject 0.082     2.2 0.67   8 0.41   5.0 0.97  54.9  dense_tiles
#   notebook 8             pass   0.228     3.0 0.38  42 0.80   8.0 0.00   7.7  ok
#   notebook 10            pass   0.184     3.0 0.33  36 0.77  10.1 0.00   7.5  ok
#   scroll text 10         pass   0.100     2.8 0.24  24 0.22  10.1 0.01   1.6  ok
#   brick 20x10            pass   0.236     2.8 0.44  36 0.49  10.1 0.17  67.3  ok
#   brick 24x12            pass   0.204     2.9 0.36  36 0.44  12.0 0.18  55.2  ok
#   fence 8                pass   0.166     2.9 0.40  29 0.78   8.0 0.00  11.0  ok
#   fence 10               pass   0.137     2.9 0.36  26 0.73  10.0 0.00  10.7  ok
#   blinds 8               pass   0.174     2.9 0.38  24 0.79   8.0 0.00   9.4  ok
#   blinds 10              pass   0.143     2.9 0.34  24 0.75  10.1 0.00   9.1  ok
#   ladder 10              pass   0.067     3.0 0.33  14 0.76  10.0 0.00  15.2  ok
#   stripes 8              pass   0.094     3.0 0.38  18 0.79   8.0 0.00   0.0  ok
#   notebook 8 @1024       pass   0.228     5.9 0.38  42 0.76  16.0 0.00   3.8  ok
#   notebook 10 @1024      pass   0.184     5.9 0.33  36 0.74  20.2 0.00   3.8  ok
#   scroll text 10 @1024   pass   0.109     6.1 0.27  24 0.21  20.2 0.01   3.7  ok
#   brick 20x10 @1024      pass   0.236     5.6 0.44  36 0.47  20.2 0.17  33.2  ok
#   brick 24x12 @1024      pass   0.204     5.7 0.36  36 0.43  24.0 0.18  27.3  ok
#   fence 8 @1024          pass   0.167     5.9 0.41  29 0.74  16.0 0.00   7.2  ok
#   fence 10 @1024         pass   0.138     5.8 0.36  28 0.70  20.1 0.00   7.1  ok
#   blinds 8 @1024         pass   0.173     5.8 0.38  24 0.75  16.0 0.00   4.7  ok
#   blinds 10 @1024        pass   0.141     5.8 0.33  24 0.73  20.2 0.00   4.6  ok
#   ladder 10 @1024        pass   0.067     5.9 0.33  14 0.73  19.9 0.00   7.6  ok
#   stripes 8 @1024        pass   0.094     6.1 0.38  18 0.74  16.0 0.00   0.0  ok
#   silhouette 0.10        reject 0.101 13274.0 1.00  12 0.91  64.0 1.00   0.0  stroke_width
#   silhouette 0.15        reject 0.150 19690.0 1.00  12 0.98  42.7 1.00   0.0  stroke_width
#   silhouette 0.20        reject 0.202 26524.0 1.00  16 1.00  64.0 1.00   0.0  stroke_width
#   silhouette 0.30        reject 0.303 39680.0 1.00  24 0.94  63.1 1.00   0.0  stroke_width
#   crosshatched_box (pinned) reject 0.248     1.9 0.73  32 0.48   4.2 1.00   2.8  dense_tiles
#   cross45 w2 s12         none   0.111     2.6 0.33  16 0.37  12.1 1.00  54.2  ok
#   cross45 w2 s13         none   0.103     2.6 0.30  16 0.33  12.9 1.00  52.3  ok
#   cross45 w2 s14         none   0.096     2.6 0.29  16 0.34  13.9 1.00  49.2  ok
#   cross45 w2 s15         none   0.090     2.7 0.26  16 0.31  15.1 1.00  46.1  ok
#   cross45 w2 s16         none   0.085     2.7 0.25  16 0.28  16.5 1.00  43.5  ok
#   cross45 w3 s12         none   0.173     4.0 0.51  32 0.43  12.3 1.00  53.5  ok
#   cross45 w3 s13         none   0.161     4.0 0.47  32 0.38  12.9 0.99  51.9  ok
#   cross45 w3 s14         none   0.151     4.1 0.45  20 0.38  14.1 0.89  48.8  ok
#   cross45 w3 s15         none   0.142     4.2 0.42  16 0.41  15.1 1.00  45.7  ok
#   cross45 w3 s16         none   0.135     4.2 0.40  16 0.39  16.5 1.00  43.2  ok
#   scroll 512 w3          pass   0.021     3.0 0.09   0 0.00   0.0 0.00   3.4  ok
#   scroll 1024 w3         pass   0.010     3.1 0.05   0 0.00   0.0 0.00   1.7  ok
#   scroll 1024 w6         pass   0.021     6.1 0.09   0 0.00   0.0 0.00   1.7  ok
#   scroll 512 w6          pass   0.040     6.1 0.19   2 0.87  64.0 0.00   3.5  ok
#   scroll 512 w1          pass   0.007     1.0 0.03   0 0.00   0.0 0.00   3.3  ok
#   scroll 512 w2          pass   0.014     2.0 0.06   0 0.00   0.0 0.00   3.4  ok
#   scroll 1024 w1         pass   0.003     1.0 0.02   0 0.00   0.0 0.00   1.7  ok
#   road map 40 seed1      pass   0.221     2.8 0.43  49 0.03  51.2 0.79  33.9  ok
#   road map 40 seed2      pass   0.231     3.1 0.40  48 0.02  55.5 0.88  37.4  ok
#   road map 40 seed3      pass   0.225     3.0 0.51  47 0.06  60.3 0.76  35.4  ok
#   road map 40 seed4      pass   0.222     3.0 0.56  43 0.07  64.0 0.61  35.7  ok
#   road map 40 seed5      pass   0.240     3.0 0.53  47 0.06  36.6 0.81  41.5  ok
#   road map 40 1024 w6    pass   0.239     5.9 0.47  54 0.04 114.5 0.71  22.1  ok
#   label map              pass   0.129     3.0 0.30  25 0.03  53.3 0.65  26.8  ok
#   cross45 w2 s20         pass   0.069     2.7 0.20  16 0.28  20.1 1.00  33.2  ok
#   cross45 w2 s24         pass   0.058     2.8 0.17  16 0.21  25.9 1.00  26.3  ok
#   cross45 w2 s32         pass   0.044     2.8 0.13   0 0.00   0.0 0.00  20.7  ok
#   cross45 w2 s40         pass   0.036     2.8 0.11   0 0.00   0.0 0.00  18.5  ok
#   cross45 w3 s20         pass   0.111     4.3 0.33  16 0.36  20.1 1.00  32.9  ok
#   cross45 w3 s24         pass   0.095     4.5 0.28  16 0.31  25.9 1.00  26.0  ok
#   cross45 w3 s32         pass   0.072     4.6 0.22  16 0.26  32.8 1.00  20.4  ok
#   cross45 w3 s40         pass   0.058     4.6 0.17  16 0.21  45.3 1.00  18.3  ok
#   grid w2 s20            pass   0.065     2.0 0.23  16 0.26  19.7 0.90  25.7  ok
#   grid w2 s24            pass   0.057     2.0 0.18  14 0.19  25.6 0.71  22.2  ok
#   grid w2 s32            pass   0.040     2.0 0.12   0 0.00   0.0 0.00  15.3  ok
#   grid w2 s40            pass   0.031     2.0 0.12   0 0.00   0.0 0.00  11.8  ok
#   grid w3 s20            pass   0.095     2.9 0.31  16 0.33  19.7 0.90  25.8  ok
#   grid w3 s24            pass   0.083     2.9 0.26  16 0.31  25.6 0.85  22.3  ok
#   grid w3 s32            pass   0.059     2.9 0.18  16 0.28  32.0 0.92  15.3  ok
#   grid w3 s40            pass   0.046     2.9 0.18  13 0.15  42.7 0.84  11.9  ok
#   parallel0 w2 s20       pass   0.034     2.0 0.12   0 0.00   0.0 0.00   0.0  ok
#   parallel0 w2 s24       pass   0.030     2.0 0.09   0 0.00   0.0 0.00   0.0  ok
#   parallel0 w2 s32       pass   0.021     2.0 0.06   0 0.00   0.0 0.00   0.0  ok
#   parallel0 w2 s40       pass   0.016     2.0 0.06   0 0.00   0.0 0.00   0.0  ok
#   parallel0 w3 s20       pass   0.051     3.0 0.17   8 0.55  21.3 0.00   0.0  ok
#   parallel0 w3 s24       pass   0.045     3.0 0.14   0 0.00   0.0 0.00   0.0  ok
#   parallel0 w3 s32       pass   0.031     3.0 0.09   0 0.00   0.0 0.00   0.0  ok
#   parallel0 w3 s40       pass   0.024     3.0 0.09   0 0.00   0.0 0.00   0.0  ok
#   cross45 w4 s20 @1024   reject 0.111     4.4 0.32  16 0.35  20.1 1.00  34.7  dense_tiles
#   road map 80 (scribble) reject 0.402     3.1 0.70  63 0.10  56.5 0.49  53.1  ink_ratio
MIN_FILL_WIDTH_PX = 12.0
FILL_WIDTH_FACTOR = 3.0
MAX_INK_RATIO = 0.35
TILE_GRID = 8
DENSE_TILE_INK = 0.15
MIN_DENSE_TILES = 8
MIN_BLOCK_PERIODICITY = 0.20
SPECTRAL_PEAK_BINS = 16
MAX_HATCH_PERIOD_PX = 11.0  # at 512 px; scaled by min(h, w) / 512
REFERENCE_SIZE_PX = 512
MIN_SECOND_ORIENTATION = 0.4
SECOND_ORIENTATION_SEPARATION_DEG = 30.0
MIN_SPECK_PX = 5
SPECK_AREA_FRACTION = 2e-5


@dataclass(frozen=True)
class XdogParams:
    sigma: float
    k: float
    tau: float
    eps: float
    phi: float


XDOG_DEFAULT = XdogParams(1.5, 1.6, 1.0, -0.03, 200.0)


@dataclass(frozen=True)
class DensityReport:
    ink_ratio: float
    mean_stroke_width: float
    max_tile_ink: float
    dense_tiles: int
    block_periodicity: float
    hatch_period_px: float
    second_orientation_ratio: float
    junction_clusters_per_kpx: float
    ok: bool
    reason: str | None


def _upright(raster: bytes) -> bytes:
    """The raster with its EXIF Orientation applied; bytes without a tag pass through."""
    try:
        with Image.open(BytesIO(raster)) as image:
            if image.getexif().get(_EXIF_ORIENTATION, 1) == 1:
                return raster
            upright = ImageOps.exif_transpose(image)
            buffer = BytesIO()
            upright.save(buffer, format="PNG")
            return buffer.getvalue()
    except (UnidentifiedImageError, OSError, ValueError):
        return raster  # the decode below reports it


def _to_gray(png: bytes) -> np.ndarray:
    data = cv2.imdecode(np.frombuffer(_upright(png), np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise SketchRejectedError("undecodable image")
    if data.dtype != np.uint8:
        data = cv2.convertScaleAbs(data, alpha=255.0 / 65535.0)  # 16-bit PNG
    if data.ndim == 2:
        return data
    if data.shape[2] == 4:
        alpha = data[:, :, 3:4].astype(np.float32) / 255.0
        color = data[:, :, :3].astype(np.float32)
        flat = color * alpha + 255.0 * (1.0 - alpha)
        gray: np.ndarray = cv2.cvtColor(np.rint(flat).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        return gray
    return np.asarray(cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2GRAY))


def binarize(png: bytes, *, method: Literal["otsu", "adaptive"] = "otsu") -> np.ndarray:
    gray = _to_gray(png)
    if method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
    ink = _drop_specks(mask > 0)
    if not ink.any():
        raise SketchRejectedError("empty")
    return ink


def _drop_specks(ink: np.ndarray) -> np.ndarray:
    """Remove 8-connected components smaller than max(5 px, 2e-5 of the image).

    A pen dot (i-dot, nuqta) as wide as the line weight must survive: at 512 px a
    3 px dot is about 7 px, at 2048 a 12 px dot about 113 px (limit 84).
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    min_area = max(MIN_SPECK_PX, round(ink.shape[0] * ink.shape[1] * SPECK_AREA_FRACTION))
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # background
    return np.asarray(keep[labels])


def _tile_edges(length: int) -> np.ndarray:
    return np.linspace(0, length, TILE_GRID + 1).astype(int)


def _tile_ink(ink: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    tiles = np.zeros((TILE_GRID, TILE_GRID))
    for i in range(TILE_GRID):
        for j in range(TILE_GRID):
            tile = ink[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            tiles[i, j] = tile.sum() / max(tile.size, 1)
    return tiles


def _largest_block(dense: np.ndarray) -> np.ndarray:
    """The largest 4-connected block of dense tiles, as a tile mask."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dense.astype(np.uint8), connectivity=4
    )
    if count <= 1:
        return np.zeros_like(dense)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.asarray(labels == best)


def _spectrum(
    ink: np.ndarray, block: np.ndarray, ys: np.ndarray, xs: np.ndarray, period_limit: float
) -> tuple[float, float, float]:
    """(periodicity, period_px, second_orientation_ratio) over the dense block.

    periodicity is the share of spectral energy in the strongest bins: hatching
    puts it in a few sharp peaks, a busy map spreads it. period_px is the
    fundamental: the lowest-frequency peak with at least half the top power.
    second_orientation_ratio looks only at hatch frequencies (period <= limit):
    the strongest bin at least 30 degrees from the top bin's orientation, over
    the top bin. Crosshatch and grids have two line families (ratio near 1);
    ruled pages, fences, blinds and brick courses have one (ratio near 0).
    Frequencies under one cycle per tile (the block envelope) are ignored; tiles
    are h/8 x w/8, so the cutoff is measured in cycles per tile on each axis.
    """
    rows, cols = np.nonzero(block.any(axis=1))[0], np.nonzero(block.any(axis=0))[0]
    y0, y1, x0, x1 = ys[rows[0]], ys[rows[-1] + 1], xs[cols[0]], xs[cols[-1] + 1]
    inside = np.zeros(ink.shape, bool)
    for i, j in zip(*np.nonzero(block), strict=True):
        inside[ys[i] : ys[i + 1], xs[j] : xs[j + 1]] = True
    inside = inside[y0:y1, x0:x1]
    region = ink[y0:y1, x0:x1].astype(np.float64)
    region = np.where(inside, region - region[inside].mean(), 0.0)
    region *= np.outer(np.hanning(y1 - y0), np.hanning(x1 - x0))
    power = np.abs(np.fft.fft2(region)) ** 2
    power[0, 0] = 0.0
    fy = np.fft.fftfreq(y1 - y0)[:, None]
    fx = np.fft.fftfreq(x1 - x0)[None, :]
    freq = np.hypot(fy, fx)
    per_tile = np.hypot(fy * ink.shape[0] / TILE_GRID, fx * ink.shape[1] / TILE_GRID)
    fine = np.where(per_tile >= 1.0, power, 0.0)
    total = float(power.sum())
    if total <= 1e-9 or float(fine.max()) <= 1e-9:
        return 0.0, 0.0, 0.0
    periodicity = float(np.sort(power.ravel())[-SPECTRAL_PEAK_BINS:].sum()) / total
    strong = fine >= 0.5 * fine.max()
    period = 1.0 / float(freq[strong].min())
    hatch = np.where(freq >= 1.0 / period_limit, fine, 0.0)
    top = float(hatch.max())
    if top <= 1e-9:
        return periodicity, period, 0.0
    angle = np.degrees(np.arctan2(fy, fx) + np.zeros_like(freq)) % 180.0
    peak = np.unravel_index(int(np.argmax(hatch)), hatch.shape)
    apart = np.abs(angle - angle[peak])
    apart = np.minimum(apart, 180.0 - apart)
    second = float(np.where(apart >= SECOND_ORIENTATION_SEPARATION_DEG, hatch, 0.0).max())
    return periodicity, period, second / top


def _junction_clusters(skeleton: np.ndarray) -> int:
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0.0
    neighbours = cv2.filter2D(skeleton, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
    junctions = ((neighbours >= 3) & (skeleton > 0)).astype(np.uint8)
    return int(cv2.connectedComponents(junctions, connectivity=8)[0]) - 1


def check_density(ink: np.ndarray, *, line_weight_px: float = 3.0) -> DensityReport:
    ink_px = int(ink.sum())
    ink_ratio = ink_px / max(ink.size, 1)
    skeleton = skeletonize(ink).astype(np.uint8)  # type: ignore[no-untyped-call]
    skeleton_px = int(skeleton.sum())
    if skeleton_px == 0:
        stroke_width = 0.0
        clusters_per_kpx = 0.0
    else:
        stroke_width = ink_px / skeleton_px
        clusters_per_kpx = _junction_clusters(skeleton) * 1000.0 / skeleton_px
    ys, xs = _tile_edges(ink.shape[0]), _tile_edges(ink.shape[1])
    tiles = _tile_ink(ink, ys, xs)
    block = _largest_block(tiles > DENSE_TILE_INK)
    dense_tiles = int(block.sum())
    period_limit = MAX_HATCH_PERIOD_PX * min(ink.shape) / REFERENCE_SIZE_PX
    periodicity, period, second = (
        _spectrum(ink, block, ys, xs, period_limit) if dense_tiles else (0.0, 0.0, 0.0)
    )
    width_limit = max(MIN_FILL_WIDTH_PX, FILL_WIDTH_FACTOR * line_weight_px)
    reason: str | None = None
    if stroke_width > width_limit:
        reason = f"stroke_width {stroke_width:.1f} > {width_limit:g}"
    elif ink_ratio > MAX_INK_RATIO:
        reason = f"ink_ratio {ink_ratio:.2f} > {MAX_INK_RATIO}"
    elif (
        dense_tiles >= MIN_DENSE_TILES
        and periodicity >= MIN_BLOCK_PERIODICITY
        and period <= period_limit
        and second >= MIN_SECOND_ORIENTATION
    ):
        reason = (
            f"dense_tiles {dense_tiles} >= {MIN_DENSE_TILES}"
            f" (periodicity {periodicity:.2f}, period {period:.1f} px,"
            f" second orientation {second:.2f})"
        )
    return DensityReport(
        ink_ratio=ink_ratio,
        mean_stroke_width=stroke_width,
        max_tile_ink=float(tiles.max()),
        dense_tiles=dense_tiles,
        block_periodicity=periodicity,
        hatch_period_px=period,
        second_orientation_ratio=second,
        junction_clusters_per_kpx=clusters_per_kpx,
        ok=reason is None,
        reason=reason,
    )


def _check_upload_size(raster: bytes) -> None:
    """Refuse an undecodable, too large or too small image from its header alone."""
    try:
        with Image.open(BytesIO(raster)) as image:
            width, height = image.size
    except Image.DecompressionBombError as e:
        raise SketchRejectedError("too large") from e
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise SketchRejectedError("undecodable image") from e
    if max(width, height) > MAX_UPLOAD_SIDE_PX:
        raise SketchRejectedError("too large")
    if min(width, height) < MIN_UPLOAD_SIDE_PX:
        raise SketchRejectedError("too small")


def _resized(gray: np.ndarray) -> np.ndarray:
    scale = XDOG_LONG_SIDE_PX / max(gray.shape)
    if scale == 1.0:
        return gray
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    size = (round(gray.shape[1] * scale), round(gray.shape[0] * scale))
    return np.asarray(cv2.resize(gray, size, interpolation=interpolation))


def xdog_lineart(raster: bytes, *, params: XdogParams = XDOG_DEFAULT) -> np.ndarray:
    """An uploaded raster as a 2-D bool ink mask (True = ink), like `binarize`.

    Raises `SketchRejectedError` for an undecodable, too large (long side over
    MAX_UPLOAD_SIDE_PX) or too small (short side under MIN_UPLOAD_SIDE_PX) image,
    and "empty" when no ink is left (a flat image, or only specks).
    """
    _check_upload_size(raster)
    gray = _resized(_to_gray(raster)).astype(np.float64)
    low, high = np.percentile(gray, XDOG_PERCENTILES)
    if high - low < _FLAT_GREY_LEVELS:
        raise SketchRejectedError("empty")
    stretched = np.clip((gray - low) / (high - low), 0.0, 1.0)
    narrow = cv2.GaussianBlur(stretched, (0, 0), params.sigma)
    wide = cv2.GaussianBlur(stretched, (0, 0), params.k * params.sigma)
    diff = narrow - params.tau * wide
    tone = np.where(diff >= params.eps, 1.0, 1.0 + np.tanh(params.phi * (diff - params.eps)))
    ink = _drop_specks(tone < 0.5)
    if not ink.any():
        raise SketchRejectedError("empty")
    return ink


def ink_to_png(ink: np.ndarray) -> bytes:
    """An ink mask as an 8-bit grayscale PNG: ink 0, paper 255 (#FFFFFF)."""
    if ink.ndim != 2:
        raise ValueError(f"ink mask must be 2-D, got {ink.ndim}-D")
    pixels = np.where(ink, 0, _PAPER).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()
