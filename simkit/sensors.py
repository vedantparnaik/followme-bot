"""Virtual sensors: what the rover would perceive in a ``World``.

``VirtualCamera`` stands in for YOLO + ByteTrack. It projects people and
camera-visible obstacles through the same pinhole the brain assumes, and adds
the failure modes that matter: box jitter, dropped frames, occlusion by
pillars and by other people, a new track ID after a long occlusion, and
(optionally) the tracker handing the target's ID to whoever walked in front.

``Lidar`` and ``Sonar3`` are range sensors: a planar scan, or three
ultrasonic cones like the HC-SR04 layout in ``docs/hardware.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from followme.config import Camera
from followme.obstacles import ObstaclePoint, RangeCone, spread
from followme.perception import Detection
from simkit.world import Person, World

BINS = 24


def make_appearance(rng: np.random.Generator, like: Optional[np.ndarray] = None,
                    mix: float = 0.0) -> np.ndarray:
    """A colour histogram. ``mix`` > 0 makes it resemble ``like`` (similar clothes)."""
    h = rng.dirichlet(np.full(BINS, 0.35))
    if like is not None and mix > 0:
        h = (1 - mix) * h + mix * like
    return h / h.sum()


def observe_feature(rng, hist: np.ndarray, noise: float = 0.5) -> np.ndarray:
    obs = np.clip(hist * (1.0 + noise * rng.standard_normal(hist.shape)), 0.0, None) + 1e-4
    obs = np.sqrt(obs / obs.sum())
    return obs / np.linalg.norm(obs)


def lens_pose(pose, cam: Camera):
    x, y, yaw = pose
    return x + cam.forward_m * math.cos(yaw), y + cam.forward_m * math.sin(yaw), yaw


def to_local(lx, ly, yaw, px, py):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * (px - lx) + s * (py - ly), -s * (px - lx) + c * (py - ly)


@dataclass
class _Track:
    tid: Optional[int] = None
    last_seen: float = -1e9


class VirtualCamera:
    def __init__(self, cam: Camera, rng: np.random.Generator, jitter: float = 0.02,
                 dropout: float = 0.02, track_buffer_s: float = 1.0, id_swaps: bool = False,
                 min_visible: float = 0.35):
        self.cam, self.rng = cam, rng
        self.jitter, self.dropout = jitter, dropout
        self.track_buffer_s, self.id_swaps = track_buffer_s, id_swaps
        self.min_visible = min_visible
        self._tracks: Dict[str, _Track] = {}
        self._next_id = 1
        self._occluded_by: Dict[str, Optional[str]] = {}
        self.truth: Dict[int, str] = {}    # track id -> person name (for scoring)

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _project_person(self, p: Person, lx, ly, yaw, px, py):
        cam = self.cam
        dx, dy = to_local(lx, ly, yaw, px, py)
        if dx < 0.3:
            return None
        u = cam.cx + cam.fx * (-dy) / dx
        half_w = cam.fx * p.radius / dx
        v0 = cam.cy + cam.fy * (cam.height_m - p.height) / dx
        v1 = cam.cy + cam.fy * cam.height_m / dx
        x0, x1 = u - half_w, u + half_w
        cx0, cx1 = max(0.0, x0), min(float(cam.width), x1)
        cy0, cy1 = max(0.0, v0), min(float(cam.height), v1)
        if cx1 - cx0 < self.min_visible * (x1 - x0) or cy1 <= cy0:
            return None
        return (cx0, cy0, cx1 - cx0, cy1 - cy0), dx

    def _occlusion(self, world: World, p: Person, lx, ly, px, py):
        """Fraction of 5 sight lines across the person that are blocked, and by whom."""
        ang = math.atan2(py - ly, px - lx)
        nx, ny = -math.sin(ang), math.cos(ang)
        blocked, by = 0, None
        for k in (-0.8, -0.4, 0.0, 0.4, 0.8):
            tx, ty = px + nx * p.radius * k, py + ny * p.radius * k
            d = math.hypot(tx - lx, ty - ly) - p.radius * 0.5
            a = math.atan2(ty - ly, tx - lx)
            dx, dy = math.cos(a), math.sin(a)
            hit = False
            for ob in world.obstacles:
                if ob.height < 1.0:
                    continue
                t = ob.shape.ray(lx, ly, dx, dy)
                if t is not None and t < d:
                    hit = True
                    break
            if not hit:
                for q, c in world.person_circles():
                    if q is p:
                        continue
                    t = c.ray(lx, ly, dx, dy)
                    if t is not None and t < d:
                        hit, by = True, q.name
                        break
            blocked += hit
        return blocked / 5.0, by

    def observe(self, world: World, pose, t: float) -> List[Detection]:
        cam, rng = self.cam, self.rng
        lx, ly, yaw = lens_pose(pose, cam)
        dets: List[Detection] = []
        people = {p.name: p for p in world.people}
        seen: Dict[str, Tuple] = {}
        occluded_now: Dict[str, Optional[str]] = {}

        for p, circ in world.person_circles(t):
            proj = self._project_person(p, lx, ly, yaw, circ.x, circ.y)
            if proj is None:
                continue
            frac, by = self._occlusion(world, p, lx, ly, circ.x, circ.y)
            if frac >= 0.6:
                occluded_now[p.name] = by
                continue
            seen[p.name] = (proj, frac, by)

        if self.id_swaps:
            for name, by in occluded_now.items():
                tr = self._tracks.get(name)
                if (by and by in seen and tr and tr.tid is not None
                        and self._occluded_by.get(name) != by and t - tr.last_seen < 0.2):
                    occ = self._tracks.setdefault(by, _Track())
                    occ.tid, occ.last_seen = tr.tid, t
                    self.truth[tr.tid] = by
                    tr.tid = None
            self._occluded_by = occluded_now

        for name, ((box, _dx), frac, by) in seen.items():
            if rng.random() < self.dropout:
                continue
            p = people[name]
            tr = self._tracks.setdefault(name, _Track())
            if tr.tid is None or t - tr.last_seen > self.track_buffer_s:
                tr.tid = self._new_id()
                self.truth[tr.tid] = name
            tr.last_seen = t
            x, y, w, h = box
            j = self.jitter
            w2, h2 = w * (1 + j * rng.standard_normal()), h * (1 + j * rng.standard_normal())
            x2 = x + (w - w2) / 2 + j * w * rng.standard_normal()
            y2 = y + (h - h2) / 2
            hist = p.appearance
            if by is not None and frac > 0:
                hist = (1 - frac) * hist + frac * people[by].appearance
            dets.append(Detection(box=(x2, y2, max(2.0, w2), max(2.0, h2)), track_id=tr.tid,
                                  label="person", conf=0.9,
                                  feature=observe_feature(rng, hist)))

        for ob in world.obstacles:
            if ob.camera_label is None:
                continue
            box = self._project_obstacle(ob, lx, ly, yaw)
            if box is not None and rng.random() >= self.dropout:
                dets.append(Detection(box=box, label=ob.camera_label, conf=0.7))
        return dets

    def _project_obstacle(self, ob, lx, ly, yaw):
        cam = self.cam
        us, vs = [], []
        for gx, gy in ob.shape.outline(16):
            dx, dy = to_local(lx, ly, yaw, gx, gy)
            if dx < 0.3:
                continue
            u = cam.cx + cam.fx * (-dy) / dx
            us.append(u)
            vs.append(cam.cy + cam.fy * cam.height_m / dx)
            vs.append(cam.cy + cam.fy * (cam.height_m - ob.height) / dx)
        if not us:
            return None
        x0, x1 = max(0.0, min(us)), min(float(cam.width), max(us))
        y0, y1 = max(0.0, min(vs)), min(float(cam.height), max(vs))
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None
        return (x0, y0, x1 - x0, y1 - y0)


class Lidar:
    """Planar scan, front half-plane."""

    def __init__(self, rng, fov_deg=180.0, step_deg=2.0, max_r=6.0, forward=0.28, noise=0.01):
        self.rng, self.max_r, self.forward, self.noise = rng, max_r, forward, noise
        n = int(round(fov_deg / step_deg))
        self.angles = [math.radians(-fov_deg / 2 + i * step_deg) for i in range(n + 1)]
        self.half = math.radians(step_deg / 2)
        self.last: List[Tuple[float, float]] = []

    def cones(self) -> List[RangeCone]:
        return [RangeCone(self.forward, 0.0, a, self.half, r) for a, r in self.last]

    def sense(self, world: World, pose) -> List[ObstaclePoint]:
        x, y, yaw = pose
        ox, oy = x + self.forward * math.cos(yaw), y + self.forward * math.sin(yaw)
        pts, self.last = [], []
        for a in self.angles:
            r = world.raycast(ox, oy, yaw + a, self.max_r, min_height=0.15)
            self.last.append((a, r))
            if r < self.max_r:
                r += self.noise * self.rng.standard_normal()
                pts.append(ObstaclePoint(self.forward + r * math.cos(a), r * math.sin(a), "lidar"))
        return pts


class Sonar3:
    """Three ultrasonic cones: front-left, front, front-right."""

    def __init__(self, rng, angles_deg=(30.0, 0.0, -30.0), half_deg=12.0, max_r=4.0,
                 forward=0.31, noise=0.02, rays=7):
        self.rng, self.max_r, self.forward, self.noise = rng, max_r, forward, noise
        self.angles = [math.radians(a) for a in angles_deg]
        self.half = math.radians(half_deg)
        self.rays = rays
        self.last: List[Tuple[float, float]] = []

    def cones(self) -> List[RangeCone]:
        return [RangeCone(self.forward, 0.0, a, self.half, r) for a, r in self.last]

    def sense(self, world: World, pose) -> List[ObstaclePoint]:
        x, y, yaw = pose
        ox, oy = x + self.forward * math.cos(yaw), y + self.forward * math.sin(yaw)
        pts, self.last = [], []
        for a in self.angles:
            r = min(world.raycast(ox, oy, yaw + a - self.half + 2 * self.half * i / (self.rays - 1),
                                  self.max_r, min_height=0.15) for i in range(self.rays))
            self.last.append((a, r))
            if r < self.max_r:
                r = max(0.02, r + self.noise * self.rng.standard_normal())
                for p in spread(r, a, self.half, "sonar"):
                    p.x += self.forward
                    pts.append(p)
        return pts


SENSORS = {"none": None, "lidar": Lidar, "sonar3": Sonar3}
