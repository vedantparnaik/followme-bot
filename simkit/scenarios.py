"""Test scenarios. Same seed -> same world, same people, same timing.

loop       the original: an ellipse with four stops, no obstacles.
obstacles  the person walks around furniture; chasing them directly cuts
           the corner through it. One obstacle is invisible to the camera
           (only a range sensor sees it).
crossing   someone in similar clothes walks between the rover and the target,
           stops right in the line of sight, and the tracker hands them the
           target's ID. Re-ID has to notice and recover.
pillar     the target walks behind a pillar for a few seconds and comes back
           with a new track ID. A stranger stands nearby; he must not be
           adopted as the target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from simkit.sensors import make_appearance
from simkit.world import Circle, Obstacle, Person, Rect, World

TARGET_COLOR = (0.95, 0.45, 0.15)
OTHER_COLOR = (0.25, 0.55, 0.95)


@dataclass
class Scenario:
    name: str
    about: str
    world: World
    duration: float
    id_swaps: bool = False
    start: tuple = (0.0, 0.0, 0.0)
    target_name: str = "target"

    @property
    def target(self) -> Person:
        return next(p for p in self.world.people if p.name == self.target_name)


def _ellipse(cx=6.0, cy=0.0, rx=3.4, ry=2.8, n=96):
    return [(cx + rx * math.cos(math.pi + 2 * math.pi * i / n),
             cy + ry * math.sin(math.pi + 2 * math.pi * i / n)) for i in range(n)]


def loop(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    target = Person("target", _ellipse(), speed=0.7, loop=True, start_s=2.0,
                    stops={20: 3.2, 48: 3.2, 68: 3.2, 88: 3.2},
                    appearance=make_appearance(rng), is_target=True, color=TARGET_COLOR)
    return Scenario("loop", "ellipse with four stops", World([target], []), duration=46.0)


def obstacles(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    path = [(2.6, 0.0), (4.6, -1.1), (8.2, -1.1), (8.4, 2.6), (11.5, 3.4), (13.5, 1.0)]
    target = Person("target", path, speed=0.65, start_s=2.0, stops={2: 2.0, 4: 2.5},
                    appearance=make_appearance(rng), is_target=True, color=TARGET_COLOR)
    obs = [
        Obstacle(Rect(6.6, 1.0, 1.6, 1.6), height=0.75, name="pallet",
                 camera_label=None, color=(0.55, 0.42, 0.25)),
        Obstacle(Rect(9.6, 1.1, 0.7, 0.7), height=0.9, name="chair",
                 camera_label="chair", color=(0.35, 0.55, 0.35)),
        Obstacle(Circle(3.9, 0.25, 0.22), height=1.0, name="bollard",
                 camera_label=None, color=(0.6, 0.6, 0.62)),
    ]
    return Scenario("obstacles", "walks around furniture; the direct line cuts through it",
                    World([target], obs), duration=34.0)


def crossing(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    t_app = make_appearance(rng)
    target = Person("target", [(2.6, 0.0), (6.0, 0.0), (12.0, 0.0)], speed=0.6,
                    stops={0: 3.0, 1: 4.0}, appearance=t_app, is_target=True,
                    color=TARGET_COLOR)
    other = Person("stranger", [(4.9, 3.5), (4.9, 0.0), (4.9, -3.5)], speed=0.8,
                   start_s=7.5, stops={1: 2.5},
                   appearance=make_appearance(rng, like=t_app, mix=0.25), color=OTHER_COLOR)
    return Scenario("crossing", "similar-looking stranger stops in the line of sight",
                    World([target, other], []), duration=30.0, id_swaps=True)


def pillar(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    target = Person("target", [(2.6, 0.0), (5.2, 0.0), (7.4, 2.6), (11.0, 2.6)], speed=0.65,
                    start_s=2.0, stops={3: 3.0}, appearance=make_appearance(rng),
                    is_target=True, color=TARGET_COLOR)
    bystander = Person("bystander", [(7.0, -2.2), (7.0, -2.2)], speed=0.5,
                       appearance=make_appearance(rng), color=OTHER_COLOR)
    obs = [Obstacle(Circle(4.9, 0.85, 0.35), height=2.5, name="pillar", camera_label=None,
                    color=(0.7, 0.7, 0.72))]
    return Scenario("pillar", "target disappears behind a pillar; a bystander waits nearby",
                    World([target, bystander], obs), duration=28.0)


SCENARIOS: Dict[str, Callable[[int], Scenario]] = {
    "loop": loop, "obstacles": obstacles, "crossing": crossing, "pillar": pillar,
}
