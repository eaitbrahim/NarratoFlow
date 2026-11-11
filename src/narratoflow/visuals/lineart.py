"""Raster -> binary ink mask and the line-art density check."""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.morphology import skeletonize

from narratoflow.errors import SketchRejectedError

# Calibrated on the code-drawn fixtures (tests/fixtures/images.py), 512x512:
#   metric               line drawing   silhouette   crosshatch
#   ink_ratio            0.019          0.305        0.168
#   branch pts per kpx   15.4           0.0          842.1
# The silhouette skeleton is a thin line, so only the ink ratio catches it; the
# hatch stays under the ink limit, so only branch density catches it.
MAX_INK_RATIO = 0.22
MAX_BRANCH_DENSITY = 100.0


@dataclass(frozen=True)
class DensityReport:
    ink_ratio: float
    branch_points_per_kpx: float
    endpoint_ratio: float
    ok: bool
    reason: str | None


def _to_gray(png: bytes) -> np.ndarray:
    data = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ink = mask > 0
    if not ink.any():
        raise SketchRejectedError("empty")
    return ink


def check_density(ink: np.ndarray) -> DensityReport:
    ink_ratio = float(ink.mean())
    skeleton = skeletonize(ink).astype(np.uint8)  # type: ignore[no-untyped-call]
    skeleton_px = int(skeleton.sum())
    if skeleton_px == 0:
        branch_per_kpx = 0.0
        endpoint_ratio = 0.0
    else:
        kernel = np.ones((3, 3), np.float32)
        kernel[1, 1] = 0.0
        neighbours = cv2.filter2D(skeleton, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
        on = skeleton > 0
        branch_per_kpx = float(((neighbours >= 3) & on).sum()) * 1000.0 / skeleton_px
        endpoint_ratio = float(((neighbours == 1) & on).sum()) / skeleton_px
    reason: str | None = None
    if ink_ratio > MAX_INK_RATIO:
        reason = f"ink_ratio {ink_ratio:.2f} > {MAX_INK_RATIO}"
    elif branch_per_kpx > MAX_BRANCH_DENSITY:
        reason = f"branch_density {branch_per_kpx:.1f} > {MAX_BRANCH_DENSITY}"
    return DensityReport(ink_ratio, branch_per_kpx, endpoint_ratio, reason is None, reason)
