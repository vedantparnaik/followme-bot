"""Detector boxes to range and bearing.

target_fix: the followed person, from the box size (pinhole model). If the
head or feet are cut off, the height reads too small, so the width is used.

ground_fix: anything standing on the floor, from the image row of the box
bottom and the camera height (assumes flat ground). Used for camera-only
obstacles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from followme.config import Camera, Follow

Box = Tuple[float, float, float, float]   # x, y, w, h in pixels


@dataclass
class Detection:
    box: Box
    track_id: Optional[int] = None
    label: str = "person"
    conf: float = 1.0
    feature: Optional[np.ndarray] = None   # appearance embedding, persons only


@dataclass
class TargetFix:
    range_m: float
    bearing: float        # rad, + left
    err_x: float          # box centre vs image centre, -1..1, + right
    clipped: str = ""     # "", "head", "feet", "side", "both"
    close_fill: bool = False   # box fills the frame vertically: definitely close


@dataclass
class GroundFix:
    range_m: float
    bearing: float
    half_angle: float     # angular half-width of the box
    clipped_bottom: bool  # bottom below the frame: at least this close


def clip_flags(box: Box, cam: Camera, margin: int):
    x, y, w, h = box
    return (
        y <= margin,
        y + h >= cam.height - margin,
        x <= margin,
        x + w >= cam.width - margin,
    )


def bearing_of_u(u: float, cam: Camera) -> float:
    return math.atan2(-(u - cam.cx), cam.fx)


def target_fix(box: Box, cam: Camera, cfg: Follow,
               last_range: Optional[float] = None) -> TargetFix:
    x, y, w, h = box
    top, bot, left, right = clip_flags(box, cam, cfg.edge_margin)
    vert, side = top or bot, left or right
    cx = x + w / 2.0
    err_x = max(-1.0, min(1.0, (cx - cam.cx) / (cam.width / 2.0)))
    d_h = cfg.person_h * cam.fy / h if h > 8 else None
    d_w = cfg.person_w * cam.fx / w if w > 8 else None

    clipped = ""
    if vert and not side and d_w:
        d, clipped = d_w, ("feet" if bot else "head")
    elif side and not vert and d_h:
        d, clipped = d_h, "side"
    elif not (vert or side) and d_h:
        d = d_h
    elif d_w:
        d, clipped = d_w, "both" if (vert and side) else ""
    else:
        d = last_range if last_range is not None else cfg.target_dist
    return TargetFix(
        range_m=d + cam.forward_m,
        bearing=bearing_of_u(cx, cam),
        err_x=err_x,
        clipped=clipped,
        close_fill=vert and h > 0.90 * cam.height,
    )


def ground_fix(box: Box, cam: Camera, margin: int = 4) -> Optional[GroundFix]:
    x, y, w, h = box
    bottom = y + h
    clipped = bottom >= cam.height - margin
    v = min(bottom, cam.height - 1)
    if v <= cam.cy + 2:
        return None   # at or above the horizon: far away or not on the floor
    z = cam.height_m * cam.fy / (v - cam.cy)
    cxu = x + w / 2.0
    b = bearing_of_u(cxu, cam)
    half = abs(bearing_of_u(x, cam) - bearing_of_u(x + w, cam)) / 2.0
    return GroundFix(range_m=z + cam.forward_m, bearing=b, half_angle=half,
                     clipped_bottom=clipped)


def iou(a: Box, b: Box) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def max_overlap(box: Box, others: Sequence[Box]) -> float:
    return max((iou(box, o) for o in others), default=0.0)
