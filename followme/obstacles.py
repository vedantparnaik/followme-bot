"""Reactive obstacle avoidance. No global map, a few seconds of memory.

Every source (range sensors, the camera's ground-plane fixes, other people)
becomes a set of points in the rover frame. The followed person is masked out,
otherwise the rover would "avoid" the person it is following.

Planning is a fan of straight corridors, one per candidate heading. A corridor
is as wide as the rover plus a margin. For each heading we measure how far we
can drive before the corridor hits a point. Headings that are blocked inside
``stop_dist`` are rejected; the rest are scored by how far they point away
from the person and how cramped they are. The chosen heading replaces the
person's bearing in the steering law; the free distance scales speed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from followme.config import Config


@dataclass
class ObstaclePoint:
    x: float
    y: float
    source: str = "range"

    @staticmethod
    def polar(r: float, bearing: float, source: str = "range") -> "ObstaclePoint":
        return ObstaclePoint(r * math.cos(bearing), r * math.sin(bearing), source)


@dataclass
class RangeCone:
    """One range reading and the space it swept: everything inside the cone
    nearer than ``range_m`` is known to be free right now."""
    x: float              # sensor origin in the rover frame
    y: float
    angle: float          # centre of the cone, rad, + left
    half: float           # half-angle
    range_m: float        # reading (max range if nothing was hit)


@dataclass
class Plan:
    heading: float                 # rad, + left
    speed_scale: float             # 0..1
    clearance: float               # free distance along the chosen corridor
    blocked: bool = False
    avoiding: bool = False
    leaves_view: bool = False     # the only way round takes the person out of frame
    fan: List[Tuple[float, float]] = field(default_factory=list)  # (heading, clearance)


def spread(r: float, bearing: float, half_angle: float, source: str,
           step: float = math.radians(4.0)) -> List[ObstaclePoint]:
    """An extended return (sonar cone, camera box) as a short arc of points."""
    n = max(1, int(math.ceil(2 * half_angle / step)))
    if n == 1:
        return [ObstaclePoint.polar(r, bearing, source)]
    return [ObstaclePoint.polar(r, bearing - half_angle + 2 * half_angle * i / (n - 1), source)
            for i in range(n)]


def mask_target(points: Iterable[ObstaclePoint], target_xy: Optional[Tuple[float, float]],
                radius: float, beam_half: float = 0.0) -> List[ObstaclePoint]:
    """Drop points that are the followed person. A wide beam (sonar) smears one
    echo across its whole cone, so range points at the person's distance
    within ``beam_half`` of their bearing go too."""
    pts = list(points)
    if target_xy is None:
        return pts
    tx, ty = target_xy
    tr, tb = math.hypot(tx, ty), math.atan2(ty, tx)

    def is_target(p):
        if math.hypot(p.x - tx, p.y - ty) <= radius:
            return True
        if beam_half <= 0 or p.source.startswith("camera"):
            return False
        db = (math.atan2(p.y, p.x) - tb + math.pi) % (2 * math.pi) - math.pi
        return abs(math.hypot(p.x, p.y) - tr) <= 0.3 and abs(db) <= beam_half

    return [p for p in pts if not is_target(p)]


class Memory:
    """Short-term obstacle memory in the odometry frame.

    Range sensors only see a cone ahead; while the rover curves around a
    corner, the corner leaves the cone but is still there. Points are kept for
    ``keep_s`` on a 5 cm grid and re-expressed in the current rover frame.
    Other people move, so camera person fixes are never remembered. The
    followed person is masked before anything is stored.
    """

    CELL = 0.05

    def __init__(self, keep_s: float):
        self.keep_s = keep_s
        self.cells = {}          # (i, j) -> (t, x, y, source)

    def clear(self) -> None:
        self.cells.clear()

    def update(self, now: float, odom, pts: Sequence[ObstaclePoint],
               free: Sequence[RangeCone] = ()) -> List[ObstaclePoint]:
        ox, oy, oyaw = odom
        c, s = math.cos(oyaw), math.sin(oyaw)
        if free and self.cells:
            self._clear(ox, oy, c, s, free)
        for p in pts:
            if p.source == "camera:person":
                continue
            wx, wy = ox + c * p.x - s * p.y, oy + s * p.x + c * p.y
            self.cells[(int(math.floor(wx / self.CELL)), int(math.floor(wy / self.CELL)))] = \
                (now, wx, wy, p.source)
        old = [k for k, v in self.cells.items() if now - v[0] > self.keep_s]
        for k in old:
            del self.cells[k]
        out = [p for p in pts if p.source == "camera:person"]
        for t, wx, wy, src in self.cells.values():
            dx, dy = wx - ox, wy - oy
            out.append(ObstaclePoint(c * dx + s * dy, -s * dx + c * dy, src))
        return out

    def _clear(self, ox, oy, c, s, free: Sequence[RangeCone]) -> None:
        """Forget remembered points a sensor can now see through (people walk away)."""
        gone = []
        for k, (t, wx, wy, src) in self.cells.items():
            dx, dy = wx - ox, wy - oy
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            for cone in free:
                r = math.hypot(lx - cone.x, ly - cone.y)
                a = math.atan2(ly - cone.y, lx - cone.x) - cone.angle
                a = (a + math.pi) % (2 * math.pi) - math.pi
                if abs(a) <= cone.half and r < cone.range_m - 0.15:
                    gone.append(k)
                    break
        for k in gone:
            del self.cells[k]


def within(points: Sequence[ObstaclePoint], radius: float) -> bool:
    return any(math.hypot(p.x, p.y) < radius for p in points)


def corridor_clearance(pts: np.ndarray, heading: float, half_width: float,
                       front: float, lookahead: float) -> float:
    """Free distance from the front bumper along ``heading``."""
    if pts.size == 0:
        return lookahead
    c, s = math.cos(heading), math.sin(heading)
    along = pts[:, 0] * c + pts[:, 1] * s
    lateral = -pts[:, 0] * s + pts[:, 1] * c
    hit = (along > 0.0) & (np.abs(lateral) < half_width)
    if not np.any(hit):
        return lookahead
    return float(max(0.0, min(lookahead, np.min(along[hit]) - front)))


class Planner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._side = 0.0          # committed detour side: +1 left, -1 right
        self._side_until = -1e9

    def reset(self) -> None:
        self._side, self._side_until = 0.0, -1e9

    def plan(self, goal_bearing: float, points: Sequence[ObstaclePoint], now: float,
             goal_range: Optional[float] = None, seen_half: Optional[float] = None,
             keep_view: bool = True) -> Plan:
        """Detours prefer to stay within ``view_keep`` of the person's bearing,
        so the person stays in the camera while the rover goes around something.
        Only if no such heading is free does it take a wider one. ``seen_half``
        is how far to the side the range sensors actually look; unseen
        headings are not free, just unknown."""
        a, body = self.cfg.avoid, self.cfg.body
        if not a.enabled or not points:
            return Plan(goal_bearing, 1.0, a.lookahead)
        keep = self.cfg.camera.hfov / 2.0 - math.radians(a.view_margin_deg)
        if not keep_view:
            keep = math.pi
        pts = np.array([[p.x, p.y] for p in points], dtype=np.float64)
        half = body.width / 2.0 + a.side_margin
        front = body.length / 2.0
        look = a.lookahead
        if goal_range is not None:
            look = max(a.stop_dist + 0.3, min(look, goal_range - front))

        span, step = math.radians(a.span_deg), math.radians(a.step_deg)
        if seen_half is not None:
            span = max(step, min(span, seen_half))
        n = int(round(2 * span / step))
        g = max(-span, min(span, goal_bearing))
        heads = [-span + i * step for i in range(n + 1)] + [g]

        fan, best, wide = [], None, None
        for h in heads:
            clr = corridor_clearance(pts, h, half, front, look)
            fan.append((h, clr))
            if clr < a.stop_dist:
                continue
            cramp = max(0.0, (a.slow_dist - clr) / a.slow_dist)
            cost = a.w_goal * abs(h - g) + a.w_clear * cramp
            if now < self._side_until and self._side and (h - g) * self._side < -1e-6 \
                    and clr < look - 1e-6:
                cost += 0.5   # don't flip to the other side mid-detour
            if abs(h - g) <= keep + 1e-9:
                if best is None or cost < best[2]:
                    best = (h, clr, cost)
            elif wide is None or cost < wide[2]:
                wide = (h, clr, cost)

        straight = corridor_clearance(pts, g, half, front, look)
        leaves = best is None and wide is not None
        best = best or wide
        if best is None:
            return Plan(g, 0.0, straight, blocked=True, fan=fan)
        h, clr, _ = best
        avoiding = abs(h - g) > math.radians(a.step_deg) * 0.99
        if avoiding:
            self._side = 1.0 if h > g else -1.0
            self._side_until = now + a.commit_s
        scale = 1.0 if clr >= a.slow_dist else max(0.0, (clr - a.stop_dist) / (a.slow_dist - a.stop_dist))
        return Plan(h, scale, clr, blocked=False, avoiding=avoiding, leaves_view=leaves, fan=fan)
