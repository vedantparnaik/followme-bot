"""A flat 2-D world: people walking scripted paths, static obstacles.

People and obstacles are circles or axis-aligned boxes on the ground plane,
with a height. Poses are a pure function of time, so two runs with the same
scenario see exactly the same people at the same moments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Circle:
    x: float
    y: float
    r: float

    def distance(self, px: float, py: float) -> float:
        return math.hypot(px - self.x, py - self.y) - self.r

    def ray(self, ox, oy, dx, dy) -> Optional[float]:
        fx, fy = ox - self.x, oy - self.y
        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - self.r * self.r
        disc = b * b - c
        if disc < 0:
            return None
        s = math.sqrt(disc)
        t = -b - s
        if t < 0:
            t = -b + s
        return t if t >= 0 else None

    def outline(self, n: int = 12):
        return [(self.x + self.r * math.cos(2 * math.pi * i / n),
                 self.y + self.r * math.sin(2 * math.pi * i / n)) for i in range(n)]


@dataclass
class Rect:
    """Axis-aligned box: centre and full size."""
    x: float
    y: float
    sx: float
    sy: float

    def distance(self, px: float, py: float) -> float:
        dx = abs(px - self.x) - self.sx / 2
        dy = abs(py - self.y) - self.sy / 2
        if dx <= 0 and dy <= 0:
            return max(dx, dy)
        return math.hypot(max(dx, 0.0), max(dy, 0.0))

    def ray(self, ox, oy, dx, dy) -> Optional[float]:
        lo_x, hi_x = self.x - self.sx / 2, self.x + self.sx / 2
        lo_y, hi_y = self.y - self.sy / 2, self.y + self.sy / 2
        tmin, tmax = -math.inf, math.inf
        for o, d, lo, hi in ((ox, dx, lo_x, hi_x), (oy, dy, lo_y, hi_y)):
            if abs(d) < 1e-12:
                if o < lo or o > hi:
                    return None
            else:
                t1, t2 = (lo - o) / d, (hi - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin, tmax = max(tmin, t1), min(tmax, t2)
                if tmin > tmax:
                    return None
        if tmax < 0:
            return None
        return tmin if tmin >= 0 else 0.0

    def outline(self, n: int = 0):
        hx, hy = self.sx / 2, self.sy / 2
        pts = []
        for i in range(4):
            ax, ay = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)][i]
            bx, by = [(hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy)][i]
            for k in range(4):
                f = k / 4
                pts.append((self.x + ax + (bx - ax) * f, self.y + ay + (by - ay) * f))
        return pts


@dataclass
class Obstacle:
    shape: object                    # Circle or Rect
    height: float = 1.0
    name: str = "obstacle"
    camera_label: Optional[str] = None   # what YOLO would call it; None = invisible to it
    color: Tuple[float, float, float] = (0.45, 0.47, 0.52)


@dataclass
class Person:
    name: str
    path: List[Tuple[float, float]]
    speed: float = 0.7
    stops: Dict[int, float] = field(default_factory=dict)   # waypoint index -> seconds
    loop: bool = False
    start_s: float = 0.0
    appearance: Optional[np.ndarray] = None
    is_target: bool = False
    radius: float = 0.24
    height: float = 1.70
    color: Tuple[float, float, float] = (0.95, 0.45, 0.15)

    def __post_init__(self):
        self._segs, self._period = self._build()

    def _build(self):
        segs, t = [], 0.0
        pts = list(self.path)
        if self.loop:
            pts = pts + [pts[0]]
        for i in range(len(pts) - 1):
            (ax, ay), (bx, by) = pts[i], pts[i + 1]
            if i in self.stops:
                segs.append((t, t + self.stops[i], ax, ay, ax, ay, False))
                t += self.stops[i]
            dur = math.hypot(bx - ax, by - ay) / self.speed
            segs.append((t, t + dur, ax, ay, bx, by, True))
            t += dur
        last = len(pts) - 1
        if not self.loop and last in self.stops:
            segs.append((t, t + self.stops[last], *pts[last], *pts[last], False))
            t += self.stops[last]
        return segs, t

    def pose(self, t: float):
        """(x, y, heading, walking)"""
        u = t - self.start_s
        segs = self._segs
        if u <= 0 or not segs:
            x, y = self.path[0]
            nx, ny = self.path[1] if len(self.path) > 1 else (x + 1, y)
            return x, y, math.atan2(ny - y, nx - x), False
        if self.loop:
            u = u % self._period
        for t0, t1, ax, ay, bx, by, walk in segs:
            if u < t1:
                f = 0.0 if t1 <= t0 else (u - t0) / (t1 - t0)
                hd = math.atan2(by - ay, bx - ax) if walk else None
                if hd is None:
                    hd = self._heading_near(ax, ay)
                return ax + (bx - ax) * f, ay + (by - ay) * f, hd, walk
        _, _, ax, ay, bx, by, _ = segs[-1]
        return bx, by, self._heading_near(bx, by), False

    def _heading_near(self, x, y):
        for (ax, ay), (bx, by) in zip(self.path, self.path[1:]):
            if (ax, ay) == (x, y) or (bx, by) == (x, y):
                return math.atan2(by - ay, bx - ax)
        return 0.0

    @property
    def duration(self) -> float:
        return self.start_s + self._period


@dataclass
class World:
    people: List[Person]
    obstacles: List[Obstacle]
    t: float = 0.0

    def person_circles(self, t: Optional[float] = None) -> List[Tuple[Person, Circle]]:
        t = self.t if t is None else t
        out = []
        for p in self.people:
            x, y, _, _ = p.pose(t)
            out.append((p, Circle(x, y, p.radius)))
        return out

    def raycast(self, ox, oy, angle, max_r, include_people=True,
                min_height: float = 0.0) -> float:
        dx, dy = math.cos(angle), math.sin(angle)
        best = max_r
        for ob in self.obstacles:
            if ob.height < min_height:
                continue
            t = ob.shape.ray(ox, oy, dx, dy)
            if t is not None and t < best:
                best = t
        if include_people:
            for _, c in self.person_circles():
                t = c.ray(ox, oy, dx, dy)
                if t is not None and t < best:
                    best = t
        return best
