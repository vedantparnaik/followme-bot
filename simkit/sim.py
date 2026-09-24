"""Closed loop: world -> virtual sensors -> Brain -> plant -> world.

Used by the headless runner, the tests and the ROS 2 sim node.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from followme.brain import Brain, Output, State
from followme.config import PROFILES, Config
from simkit.plants import PLANTS
from simkit.scenarios import Scenario
from simkit.sensors import SENSORS, VirtualCamera

FOLLOWING = (State.FOLLOW, State.AVOID, State.BLOCKED)


def footprint(pose, length: float, width: float, n: int = 3):
    """Points on the rover's outline, world frame."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2, width / 2
    local = []
    for i in range(n + 1):
        f = -1 + 2 * i / n
        local += [(hl, hw * f), (-hl, hw * f), (hl * f, hw), (hl * f, -hw)]
    return [(x + c * lx - s * ly, y + s * lx + c * ly) for lx, ly in local]


@dataclass
class Metrics:
    collisions: int = 0
    person_contacts: int = 0
    min_obstacle_clearance: float = math.inf
    min_person_clearance: float = math.inf
    first_lock_t: Optional[float] = None
    wrong_follow_s: float = 0.0
    follow_target_s: float = 0.0
    reidentified: int = 0
    range_err_sum: float = 0.0
    range_err_n: int = 0
    max_target_dist: float = 0.0
    final_target_dist: float = 0.0
    final_state: str = ""
    final_on_target: bool = False
    state_s: Dict[str, float] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)

    @property
    def mean_range_err(self) -> float:
        return self.range_err_sum / self.range_err_n if self.range_err_n else float("nan")

    def passed(self) -> bool:
        return (self.collisions == 0 and self.person_contacts == 0
                and self.wrong_follow_s < 0.5 and self.final_on_target)

    def row(self) -> dict:
        return {
            "pass": self.passed(),
            "collisions": self.collisions,
            "person_contacts": self.person_contacts,
            "min_clear_m": round(self.min_obstacle_clearance, 2),
            "min_person_m": round(self.min_person_clearance, 2),
            "follow_s": round(self.follow_target_s, 1),
            "wrong_s": round(self.wrong_follow_s, 1),
            "reid": self.reidentified,
            "range_err_m": round(self.mean_range_err, 2),
            "max_dist_m": round(self.max_target_dist, 2),
            "final": f"{self.final_state}@{self.final_target_dist:.1f}m",
        }


@dataclass
class Frame:
    t: float
    pose: tuple
    people: list          # (name, x, y, is_target, color)
    state: str
    locked: Optional[str]
    heading: Optional[float]
    obstacles: list       # world-frame points
    note: str


class Sim:
    def __init__(self, scenario: Scenario, profile: str = "hardware", plant: str = "hardware",
                 sensor: str = "sonar3", seed: int = 0, brain_hz: float = 30.0,
                 auto_arm: bool = True, cfg: Optional[Config] = None, record: bool = False):
        self.sc = scenario
        self.cfg = cfg or PROFILES[profile]()
        self.brain = Brain(self.cfg)
        rng = np.random.default_rng(seed + 1000)
        self.camera = VirtualCamera(self.cfg.camera, rng, id_swaps=scenario.id_swaps)
        cls = SENSORS[sensor]
        self.ranger = cls(rng) if cls else None
        self.plant = PLANTS[plant](self.cfg.body, *scenario.start)
        self.brain_dt = 1.0 / brain_hz
        self.t = 0.0
        self.cmd = (0.0, 0.0)
        self._next_brain = 0.0
        self.out: Optional[Output] = None
        self.dets = []
        self.m = Metrics()
        self.record = record
        self.frames: List[Frame] = []
        self._in_collision = False
        self._in_contact = False
        if auto_arm:
            self.brain.arm()

    def step(self, dt: float = 0.01) -> None:
        w = self.sc.world
        w.t = self.t
        if self.t + 1e-9 >= self._next_brain:
            self._next_brain += self.brain_dt
            self.dets = self.camera.observe(w, self.plant.pose, self.t)
            pts = self.ranger.sense(w, self.plant.pose) if self.ranger else []
            free = self.ranger.cones() if self.ranger else []
            # Dead reckoning from applied duty, as the Pi reports it. The sim has no
            # wheel slip, so this equals the true pose.
            self.out = self.brain.step(self.t, self.dets, pts, odom=self.plant.pose, free=free)
            self.cmd = (self.out.l, self.out.r)
            self._score(self.out)
        prev = (self.plant.x, self.plant.y, self.plant.yaw)
        self.plant.step(self.cmd[0], self.cmd[1], dt, self.t)
        self._contacts(prev)
        self.t += dt

    def run(self, until: Optional[float] = None, dt: float = 0.01) -> Metrics:
        end = self.sc.duration if until is None else until
        while self.t < end:
            self.step(dt)
        self._finish()
        return self.m

    def _contacts(self, prev) -> None:
        body = self.cfg.body
        pts = footprint(self.plant.pose, body.length, body.width)
        d_obs = min((ob.shape.distance(px, py) for ob in self.sc.world.obstacles
                     for px, py in pts), default=math.inf)
        d_ppl = min((c.distance(px, py) for _, c in self.sc.world.person_circles()
                     for px, py in pts), default=math.inf)
        self.m.min_obstacle_clearance = min(self.m.min_obstacle_clearance, max(d_obs, 0.0))
        self.m.min_person_clearance = min(self.m.min_person_clearance, max(d_ppl, 0.0))
        hit_obs, hit_person = d_obs < 0, d_ppl < 0
        if hit_obs or hit_person:
            self.plant.x, self.plant.y, self.plant.yaw = prev
            self.plant.halt()
        if hit_obs and not self._in_collision:
            self.m.collisions += 1
            self.m.events.append(f"{self.t:6.2f}s  COLLISION")
        if hit_person and not self._in_contact:
            self.m.person_contacts += 1
            self.m.events.append(f"{self.t:6.2f}s  PERSON CONTACT")
        self._in_collision, self._in_contact = hit_obs, hit_person

    def _target_dist(self) -> float:
        tx, ty, _, _ = self.sc.target.pose(self.t)
        return math.hypot(tx - self.plant.x, ty - self.plant.y)

    def _score(self, out: Output) -> None:
        m, dt = self.m, self.brain_dt
        name = out.state.value
        m.state_s[name] = m.state_s.get(name, 0.0) + dt
        locked = self.camera.truth.get(out.target_id) if out.target_id is not None else None
        if out.event:
            m.events.append(f"{self.t:6.2f}s  {name:8s} {out.event}")
            if "re-identified" in out.event:
                m.reidentified += 1
        if m.first_lock_t is None and out.target_id is not None:
            m.first_lock_t = self.t
        on_target = out.fix is not None and locked == self.sc.target_name
        if out.state in FOLLOWING and out.fix is not None:
            if on_target:
                m.follow_target_s += dt
            else:
                m.wrong_follow_s += dt
        d = self._target_dist()
        if m.first_lock_t is not None:
            m.max_target_dist = max(m.max_target_dist, d)
            if on_target and out.state in FOLLOWING:
                m.range_err_sum += abs(d - self.cfg.follow.target_dist)
                m.range_err_n += 1
        if self.record:
            people = []
            for p, c in self.sc.world.person_circles():
                people.append((p.name, c.x, c.y, p.is_target, p.color))
            x, y, yaw = self.plant.pose
            cs, sn = math.cos(yaw), math.sin(yaw)
            obs = [(x + cs * q.x - sn * q.y, y + sn * q.x + cs * q.y) for q in out.obstacles]
            heading = yaw + out.plan.heading if out.plan is not None else None
            self.frames.append(Frame(self.t, (x, y, yaw), people, name, locked, heading,
                                     obs, out.note))

    def _finish(self) -> None:
        out = self.out
        self.m.final_state = out.state.value if out else ""
        self.m.final_target_dist = self._target_dist()
        locked = self.camera.truth.get(out.target_id) if out and out.target_id is not None else None
        self.m.final_on_target = bool(out and locked == self.sc.target_name
                                      and self.m.final_target_dist < 4.5)
