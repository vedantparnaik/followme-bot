"""Every tunable number in one place.

Two profiles share the same brain:

- ``hardware()`` - the real 4-wheel skid-steer. It cannot turn gently (a pivot
  from rest fires a ~55% stiction kick), so it steers by curving while rolling
  and turns in place only with short pulse-and-settle nudges.
- ``ideal()`` - perfect motors. Smooth proportional steering and continuous
  search rotation.

Frames: rover frame is x forward, y left, origin at the chassis centre on the
ground. Angles are radians, positive = counter-clockwise (to the left).
Motor commands are left/right duty in percent, -100..100.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass


@dataclass
class Camera:
    width: int = 640
    height: int = 480
    fx: float = 461.0
    fy: float = 461.0
    cx: float = 320.0
    cy: float = 240.0
    height_m: float = 0.45       # lens above the ground
    forward_m: float = 0.30      # lens ahead of the chassis centre

    @property
    def hfov(self) -> float:
        return 2.0 * math.atan(self.width / 2.0 / self.fx)


@dataclass
class Body:
    length: float = 0.62
    width: float = 0.40
    v_at_100: float = 1.35       # m/s straight at 100% duty
    track: float = 0.46          # effective wheel track for yaw


@dataclass
class Follow:
    target_dist: float = 2.6     # chassis centre to person, metres
    dist_dead: float = 0.40
    kp_fwd: float = 25.0         # duty % per metre of range error
    max_speed: float = 50.0      # duty cap %
    max_reverse: float = 20.0    # backing up is blind, keep it slow
    person_h: float = 1.70
    person_w: float = 0.48       # shoulders; used when head/feet are clipped
    edge_margin: int = 4         # px; a box this close to the edge is clipped
    lost_grace_s: float = 0.9    # coast on the last fix through motion blur


@dataclass
class Steer:
    mode: str = "arc"            # "arc" (hardware) or "smooth" (ideal)
    # arc law
    kp_turn: float = 0.85        # fraction of throttle used to curve
    x_dead: float = 0.06         # no curve below this normalised error
    inner_min: float = 0.12      # slowest inner side as a fraction of outer
    creep: float = 18.0          # forward duty used only to turn without pivoting
    err_smooth: float = 0.40
    out_smooth: float = 0.18     # EMA on duties
    cmd_slew: float = 70.0       # duty %/s
    # smooth law
    kv: float = 0.8              # m/s per metre of range error
    kw: float = 2.2              # rad/s per rad of heading error
    v_max: float = 1.0
    w_max: float = 1.6


@dataclass
class Search:
    # arc mode: blind nudges toward where they left
    yaw_cmd: float = 6.0
    yaw_s: float = 0.16
    settle_s: float = 0.35
    pulses: int = 4
    # Only turn to look if they left toward the edge of the frame. Vanishing
    # near the centre means someone or something is in front of them: go to
    # where they were last seen (needs odometry), then look the way they walked.
    min_bearing_deg: float = 10.0
    pursue: bool = True
    pursue_s: float = 6.0        # give up driving to the last-seen point after this
    pursue_reach: float = 0.9    # "arrived" distance to the last-seen point
    # smooth mode: continuous rotation
    w: float = 0.7               # rad/s
    max_angle: float = 1.8       # rad, then stop and wait


@dataclass
class Avoid:
    enabled: bool = True
    stop_dist: float = 0.40      # front bumper to obstacle: do not go closer
    slow_dist: float = 1.80      # start slowing down inside this
    side_margin: float = 0.12    # corridor half-width = body.width/2 + margin
    span_deg: float = 70.0       # candidate headings +/- this
    step_deg: float = 5.0
    lookahead: float = 3.0
    w_goal: float = 1.0          # cost per rad away from the person
    w_clear: float = 2.2         # cost for a corridor that is inside slow_dist
    view_margin_deg: float = 6.0 # detours keep the person this far inside the frame
    target_mask_m: float = 0.55  # ignore range returns this close to the person
    commit_s: float = 0.6        # keep a detour side this long (no dithering)
    memory_s: float = 2.5        # remember obstacles that left the sensor cone
    person_pad: float = 0.35     # people move: treat them as this much closer
    sharp_turn_deg: float = 25.0 # headings past this need the swing circle clear
    # Stuck in BLOCKED: reverse straight a little (the way it came in is free).
    backoff_after_s: float = 1.5 # 0 disables
    backoff_m: float = 0.35
    backoff_duty: float = 26.0
    backoff_speed: float = 0.3   # m/s at backoff_duty, only used to time it
    swing_margin: float = 0.08   # clearance needed around the body to turn in place


@dataclass
class ReID:
    enabled: bool = True
    accept: float = 0.88         # appearance similarity to re-acquire
    margin: float = 0.04         # best must beat the runner-up by this
    confirm_frames: int = 5      # consecutive matching frames before switching
    reject: float = 0.80         # tracked ID this unlike the gallery = ID swap
    reject_frames: int = 3       # frames with the smoothed score below reject
    jump_m: float = 0.60         # range jump between consecutive frames = ID swap
    jump_shift: float = 0.60     # box centre move, in box widths, between frames
    rollback_s: float = 0.5      # on a swap, trust where they were this long ago
    gallery: int = 24
    min_box_h: int = 40
    lock_stable_s: float = 0.4   # candidate must persist this long to lock


@dataclass
class Config:
    camera: Camera = field(default_factory=Camera)
    body: Body = field(default_factory=Body)
    follow: Follow = field(default_factory=Follow)
    steer: Steer = field(default_factory=Steer)
    search: Search = field(default_factory=Search)
    avoid: Avoid = field(default_factory=Avoid)
    reid: ReID = field(default_factory=ReID)

    def to_dict(self) -> dict:
        return asdict(self)

    def flat(self) -> dict:
        """{"follow.target_dist": 2.6, ...} for UIs and tune files."""
        out = {}
        for sec in fields(self):
            obj = getattr(self, sec.name)
            for f in fields(obj):
                out[f"{sec.name}.{f.name}"] = getattr(obj, f.name)
        return out

    def set(self, key: str, value) -> None:
        """Set one ``section.name`` value, cast to the existing type."""
        sec, name = key.split(".", 1)
        obj = getattr(self, sec)
        if not is_dataclass(obj) or not hasattr(obj, name):
            raise KeyError(key)
        cur = getattr(obj, name)
        if isinstance(cur, bool):
            value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            value = int(float(value))
        elif isinstance(cur, float):
            value = float(value)
        else:
            value = str(value)
        setattr(obj, name, value)

    def update(self, flat: dict) -> None:
        for k, v in flat.items():
            try:
                self.set(k, v)
            except (KeyError, ValueError, AttributeError):
                pass


def hardware() -> Config:
    return Config()


def ideal() -> Config:
    c = Config()
    c.steer.mode = "smooth"
    c.follow.max_speed = 100.0
    c.follow.max_reverse = 40.0
    c.follow.lost_grace_s = 0.5
    return c


PROFILES = {"hardware": hardware, "ideal": ideal}
