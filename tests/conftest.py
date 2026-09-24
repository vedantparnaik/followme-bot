import math

import numpy as np
import pytest

from followme.config import Camera, hardware
from followme.perception import Detection


def person_box(range_m: float, bearing: float = 0.0, cam: Camera = Camera(),
               height: float = 1.7, width: float = 0.48):
    """Pixel box of an upright person at ``range_m`` from the chassis centre."""
    z = range_m - cam.forward_m
    h = height * cam.fy / z
    w = width * cam.fx / z
    u = cam.cx - cam.fx * math.tan(bearing)
    bottom = cam.cy + cam.height_m * cam.fy / z
    return (u - w / 2, bottom - h, w, h)


def feature(seed: int, n: int = 24) -> np.ndarray:
    return np.sqrt(np.random.default_rng(seed).dirichlet(np.full(n, 0.35)))


def person(range_m, bearing=0.0, tid=1, look=1, noise=0.0, rng=None):
    f = feature(look)
    if noise and rng is not None:
        f = np.abs(f + rng.normal(0, noise * 0.05, f.shape))
    return Detection(person_box(range_m, bearing), track_id=tid, feature=f)


@pytest.fixture
def cfg():
    return hardware()
