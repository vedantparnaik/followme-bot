"""Appearance memory for the followed person.

A tracker ID (ByteTrack) survives brief occlusion, but not a person walking
behind a pillar for three seconds, and it can jump to whoever crossed in front.
The gallery remembers what the target looks like so the brain can say "that
new track is the same person" or "the tracked ID is no longer them".

Features are any fixed-length vectors. On the robot they are colour
histograms of torso and legs (square-rooted, so cosine similarity is the
Bhattacharyya coefficient). In the sim they are synthetic histograms.
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
