"""Appearance features for re-identification, from the camera image.

Colour histograms of the torso and the legs, in HSV so lighting changes
move the V channel more than hue. Square-rooted and L2-normalised, so the
cosine similarity used by ``followme.reid`` is the Bhattacharyya coefficient.

Cheap (well under a millisecond per person) and good enough to tell a red
jacket from a grey one. It will not separate two people in the same uniform;
a learned re-ID embedding (OSNet etc.) drops in here with the same interface.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

H_BINS, S_BINS = 8, 3
TORSO = (0.18, 0.52)     # fraction of box height
LEGS = (0.55, 0.90)
MIN_PX = 12


def _region_hist(hsv: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros(H_BINS * S_BINS + 1)
    # Near-grey pixels have meaningless hue: count them in their own bin.
    sat = patch[..., 1] >= 40
    val = patch[..., 2] >= 30
    colour = sat & val
    h = cv2.calcHist([patch], [0, 1], colour.astype(np.uint8), [H_BINS, S_BINS],
                     [0, 180, 40, 256]).ravel()
    grey = float(np.count_nonzero(~colour))
    return np.append(h, grey)


def to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """Convert once per frame, then call ``person_feature`` per box."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


def person_feature(hsv: np.ndarray, box) -> Optional[np.ndarray]:
    H, W = hsv.shape[:2]
    x, y, w, h = box
    # Inner 60% horizontally: skip background at the sides of the box.
    x0, x1 = int(max(0, x + 0.2 * w)), int(min(W, x + 0.8 * w))
    if x1 - x0 < MIN_PX or h < 3 * MIN_PX:
        return None
    parts = []
    for a, b in (TORSO, LEGS):
        y0, y1 = int(max(0, y + a * h)), int(min(H, y + b * h))
        parts.append(_region_hist(hsv, x0, x1, y0, y1) if y1 - y0 >= MIN_PX
                     else np.zeros(H_BINS * S_BINS + 1))
    f = np.concatenate(parts)
    s = f.sum()
    if s <= 0:
        return None
    return np.sqrt(f / s)
