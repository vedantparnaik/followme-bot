"""Appearance gallery for the followed person.

ByteTrack IDs survive short occlusions but not a few seconds behind a pillar,
and an ID can jump to someone who crosses in front. The gallery is used to
re-identify the target on a new track and to notice when the tracked ID no
longer looks like them.

Features can be any fixed-length vector. On the robot they're square-rooted
colour histograms of torso and legs (so cosine similarity is the
Bhattacharyya coefficient); in the sim they're synthetic.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


class Gallery:
    def __init__(self, size: int = 24):
        self.items: deque = deque(maxlen=size)

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def add(self, feat: Optional[np.ndarray]) -> None:
        if feat is not None:
            self.items.append(normalize(feat))

    def score(self, feat: Optional[np.ndarray]) -> float:
        """Similarity to the gallery: mean of the three best matches."""
        if feat is None or not self.items:
            return 0.0
        f = normalize(feat)
        sims = sorted((float(np.dot(f, g)) for g in self.items), reverse=True)
        top = sims[:3]
        return sum(top) / len(top)
