"""The follow-me state machine. One ``step()`` per camera frame.

    IDLE --arm--> ACQUIRE --lock--> FOLLOW <--> AVOID
                                      |  ^        |
                            lost > grace  re-ID   +-> BLOCKED (nothing is free ahead)
                                      v  |
                    left the frame edge? -- no --> PURSUE (drive to last-seen point)
                                      | yes          | arrived / timed out
                                      v              v
                                    SEARCH --nothing found--> WAIT
    any --estop--> ESTOP (latched until reset)

IDLE     disarmed. Perception still runs so you can see who it would lock.
ACQUIRE  armed, nobody locked yet. Locks the biggest, most central person.
FOLLOW   hold ``target_dist`` behind them, steer toward them.
AVOID    same, but steering toward a free corridor instead of the person.
BLOCKED  wants to move forward, every corridor is blocked. Stands still, and
         backs off a little if it stays stuck.
PURSUE   target vanished mid-frame (behind a pillar, a crowd): drive, with
         avoidance, to where they were last seen. Needs odometry.
SEARCH   target gone: turn toward where they went (bounded).
WAIT     search exhausted: stand still, keep looking by appearance. Only a
         user re-lock picks a new person.
ESTOP    latched stop.

Inputs are plain data (detections + obstacle points + a timestamp), so the
same brain runs on the robot, in the ROS sim and in headless tests.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

from followme.config import Config, hardware
from followme.controller import make_drive
from followme.obstacles import (Memory, ObstaclePoint, Plan, Planner, RangeCone, mask_target,
                                spread, within)
from followme.perception import Detection, TargetFix, ground_fix, target_fix
from followme.target import TargetTracker


class State(str, Enum):
    IDLE = "IDLE"
    ACQUIRE = "ACQUIRE"
    FOLLOW = "FOLLOW"
    AVOID = "AVOID"
    BLOCKED = "BLOCKED"
    PURSUE = "PURSUE"
    SEARCH = "SEARCH"
    WAIT = "WAIT"
    ESTOP = "ESTOP"


@dataclass
class Output:
    l: float
    r: float
    state: State
    note: str = ""
    event: str = ""
    target_id: Optional[int] = None
    how: str = "none"
    fix: Optional[TargetFix] = None
    plan: Optional[Plan] = None
    obstacles: List[ObstaclePoint] = field(default_factory=list)
    reid_score: float = 0.0


class Brain:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or hardware()
        self.tracker = TargetTracker(self.cfg)
        self.planner = Planner(self.cfg)
        self.memory = Memory(self.cfg.avoid.memory_s)
        self.drive = make_drive(self.cfg)
        self.armed = False
        self.estopped = False
        self.state = State.IDLE
        self._last_t: Optional[float] = None
        self._last_range: Optional[float] = None
        self._last_bearing = 0.0
        self._searching = False
        self._search_done = False
        self._seen_at: Optional[tuple] = None    # odom-frame (x, y, t) of the last fix
        self._walk = (0.0, 0.0)                  # odom-frame velocity estimate, m/s
        self._pursue_until: Optional[float] = None
        self._seen_half: Optional[float] = None  # side coverage of the range sensors
        self._blocked_since: Optional[float] = None
        # Recent (t, bearing, seen_at, walk), to undo the frames an ID swap poisoned.
        self._history: deque = deque(maxlen=90)
        self._backoff_until = -1e9

    # ---- commands -----------------------------------------------------
    def arm(self) -> None:
        if not self.estopped:
            self.armed = True

    def disarm(self) -> None:
        self.armed = False
        self.drive.stop()

    def estop(self) -> None:
        self.estopped = True
        self.armed = False
        self.drive.stop()

    def relock(self) -> None:
        """Forget the target; the next ACQUIRE locks whoever is in front."""
        self.tracker.reset()
        self.planner.reset()
        self.memory.clear()
        self.drive.reset()
        self._clear_lost()
        self._last_range = None
        self._seen_at, self._walk = None, (0.0, 0.0)
        self._history.clear()

    def _roll_back(self, t: float) -> None:
        while self._history and self._history[-1][0] > t:
            self._history.pop()
        if self._history:
            _, self._last_bearing, self._seen_at, self._walk = self._history[-1]

    def _clear_lost(self) -> None:
        self._searching = self._search_done = False
        self._pursue_until = None

    def reset(self) -> None:
        """Clear the e-stop and the target. Stays disarmed."""
        self.estopped = False
        self.armed = False
        self.relock()

    def reconfigure(self) -> None:
        """Call after changing ``cfg.steer.mode``."""
        self.drive = make_drive(self.cfg)

    # ---- one frame ----------------------------------------------------
    def step(self, now: float, detections: Sequence[Detection],
             obstacles: Sequence[ObstaclePoint] = (),
             odom: Optional[tuple] = None,
             free: Sequence[RangeCone] = ()) -> Output:
        """``odom`` is (x, y, yaw) from any dead-reckoning source. Without it
        the brain is purely reactive (no obstacle memory). ``free`` are the
        range sensors' current cones, used to forget obstacles that moved."""
        cfg = self.cfg
        dt = 0.033 if self._last_t is None else min(max(now - self._last_t, 1e-3), 0.2)
        self._last_t = now

        if free:
            self._seen_half = max(abs(c.angle) + c.half for c in free)
        tr = self.tracker.update(detections, now)
        if tr.event.startswith("id swap"):
            self._roll_back(now - cfg.reid.rollback_s)
        fix = None
        if tr.det is not None:
            fix = target_fix(tr.det.box, cfg.camera, cfg.follow, self._last_range)
            self._last_range = fix.range_m
            self._last_bearing = fix.bearing

        pts = self._obstacle_points(detections, tr, obstacles)
        target_xy = None
        if fix is not None:
            target_xy = (fix.range_m * math.cos(fix.bearing), fix.range_m * math.sin(fix.bearing))
        beam = max((c.half for c in free), default=0.0)
        pts = mask_target(pts, target_xy, cfg.avoid.target_mask_m, beam)
        if odom is not None:
            self.memory.keep_s = cfg.avoid.memory_s
            pts = self.memory.update(now, odom, pts, free)
            pts = mask_target(pts, target_xy, cfg.avoid.target_mask_m, beam)
            if target_xy is not None and tr.how != "coast":
                self._remember_seen(now, odom, target_xy)
        if fix is not None:
            self._history.append((now, self._last_bearing, self._seen_at, self._walk))

        plan = None
        l = r = 0.0
        note = ""
        if self.estopped:
            state = State.ESTOP
            self.drive.stop()
        elif not self.armed:
            state = State.IDLE
            self.drive.stop()
            self._clear_lost()
            note = f"would follow #{tr.target_id}" if tr.target_id is not None else ""
        elif not self.tracker.locked:
            state = State.ACQUIRE
            self.drive.stop()
        elif fix is not None:
            self._clear_lost()
            range_err = fix.range_m - cfg.follow.target_dist
            plan = self._plan(fix.bearing, pts, now, fix.range_m)
            wants_forward = range_err > cfg.follow.dist_dead
            if plan.blocked and wants_forward:
                state = State.BLOCKED
                l, r, note = self._blocked(now, pts, plan)
            else:
                self._blocked_since = None
                l, r, note = self.drive.follow(now, dt, range_err, plan.heading,
                                               plan.speed_scale, fix.close_fill)
                state = State.AVOID if (plan.avoiding and wants_forward) else State.FOLLOW
            note = f"{note}  d={fix.range_m:.2f}m  b={math.degrees(fix.bearing):+.0f}deg"
            if fix.clipped:
                note += f"  {fix.clipped} clipped"
            if tr.how == "coast":
                note += "  coasting"
        else:
            if not self._searching and not self._search_done and self._pursue_until is None:
                if abs(self._last_bearing) >= math.radians(cfg.search.min_bearing_deg):
                    self._begin_search(now, self._last_bearing)
                elif cfg.search.pursue and odom is not None and self._seen_at is not None:
                    self._pursue_until = now + cfg.search.pursue_s
                else:
                    self._search_done = True     # vanished ahead, no odometry: wait
            if self._pursue_until is not None:
                goal = self._to_rover(odom, self._seen_at) if odom is not None else None
                dist = math.hypot(*goal) if goal else 0.0
                if goal is None or dist < cfg.search.pursue_reach or now > self._pursue_until:
                    self._pursue_until = None
                    self.drive.stop()
                    self._begin_search(now, self._walk_bearing(odom))
                else:
                    bearing = math.atan2(goal[1], goal[0])
                    plan = self._plan(bearing, pts, now, dist, keep_view=False)
                    people = [p for p in pts if p.source == "camera:person"]
                    if within(people, cfg.avoid.slow_dist):
                        self.drive.stop()
                        note = "someone in the way: waiting for them to pass"
                    elif plan.blocked:
                        l, r, note = self._blocked(now, pts, plan)
                        note = "to last-seen point: " + note
                    else:
                        self._blocked_since = None
                        l, r, note = self.drive.follow(now, dt, dist, plan.heading,
                                                       plan.speed_scale, False)
                        note = f"to last-seen point {dist:.1f} m  {note}"
                    state = State.PURSUE
            if self._searching and within(pts, self._swing()):
                self._searching, self._search_done = False, True
                self.drive.stop()
                note = "no room to turn"
            if self._pursue_until is not None:
                pass
            elif self._searching:
                l, r, done, note = self.drive.search(now, dt)
                state = State.SEARCH
                if done:
                    self._searching, self._search_done = False, True
                    state = State.WAIT
            else:
                state = State.WAIT
                note = note or "waiting for the same person to reappear"

        self.state = state
        return Output(l=l, r=r, state=state, note=note, event=tr.event,
                      target_id=tr.target_id, how=tr.how, fix=fix, plan=plan,
                      obstacles=pts, reid_score=tr.reid_score)

    def _blocked(self, now: float, pts, plan: Plan) -> tuple:
        """Stand still; if it stays blocked, back straight off a little so the
        next plan has room to turn. Only when the strip behind is known free."""
        a, b = self.cfg.avoid, self.cfg.body
        self.drive.stop()
        half = b.width / 2 + a.side_margin
        rear = b.length / 2 + a.backoff_m + 0.1
        rear_clear = not any(-rear < p.x < 0.0 and abs(p.y) < half for p in pts)
        if now < self._backoff_until and rear_clear:
            return -a.backoff_duty, -a.backoff_duty, "backing off"
        if self._blocked_since is None:
            self._blocked_since = now
        if a.backoff_after_s > 0 and now - self._blocked_since > a.backoff_after_s and rear_clear:
            self._backoff_until = now + a.backoff_m / a.backoff_speed
            self._blocked_since = self._backoff_until
            return -a.backoff_duty, -a.backoff_duty, "backing off"
        return 0.0, 0.0, f"blocked {plan.clearance:.2f} m ahead"

    def _swing(self) -> float:
        b = self.cfg.body
        return math.hypot(b.length, b.width) / 2 + self.cfg.avoid.swing_margin

    def _plan(self, bearing: float, pts, now: float, goal_range: float,
              keep_view: bool = True) -> Plan:
        """Corridors assume driving straight. A sharp heading on skid-steer is
        close to a pivot, so it also needs the swing circle clear."""
        plan = self.planner.plan(bearing, pts, now, goal_range=goal_range,
                                 seen_half=self._seen_half, keep_view=keep_view)
        sharp = abs(plan.heading) > math.radians(self.cfg.avoid.sharp_turn_deg)
        if not plan.blocked and sharp and within(pts, self._swing()):
            plan.blocked, plan.speed_scale = True, 0.0
        return plan

    def _begin_search(self, now: float, bearing: float) -> None:
        self.drive.begin_search(now, 1.0 if bearing >= 0 else -1.0)
        self._searching = True

    def _remember_seen(self, now: float, odom: tuple, xy: tuple) -> None:
        ox, oy, oyaw = odom
        c, s = math.cos(oyaw), math.sin(oyaw)
        wx, wy = ox + c * xy[0] - s * xy[1], oy + s * xy[0] + c * xy[1]
        if self._seen_at is not None:
            px, py, pt = self._seen_at
            dt = now - pt
            if 0.0 < dt < 0.5:
                vx, vy = (wx - px) / dt, (wy - py) / dt
                a = 0.1                            # heavy smoothing: range is noisy
                self._walk = ((1 - a) * self._walk[0] + a * vx, (1 - a) * self._walk[1] + a * vy)
        self._seen_at = (wx, wy, now)

    @staticmethod
    def _to_rover(odom: tuple, p: tuple) -> tuple:
        ox, oy, oyaw = odom
        dx, dy = p[0] - ox, p[1] - oy
        c, s = math.cos(oyaw), math.sin(oyaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _walk_bearing(self, odom: Optional[tuple]) -> float:
        """Which way to turn after reaching the last-seen point: the side they
        were walking toward, else the side they were last seen on."""
        vx, vy = self._walk
        if odom is None or math.hypot(vx, vy) < 0.15:
            return self._last_bearing
        yaw = odom[2]
        return math.atan2(-math.sin(yaw) * vx + math.cos(yaw) * vy,
                          math.cos(yaw) * vx + math.sin(yaw) * vy)

    def _obstacle_points(self, dets: Sequence[Detection], tr,
                         sensed: Sequence[ObstaclePoint]) -> List[ObstaclePoint]:
        """Range-sensor points plus camera ground fixes of everything but the target."""
        pts = list(sensed)
        cam = self.cfg.camera
        for d in dets:
            if tr.det is not None and (d is tr.det or
                                       (d.track_id is not None and d.track_id == tr.target_id)):
                continue
            g = ground_fix(d.box, cam, self.cfg.follow.edge_margin)
            if g is None:
                continue
            r = g.range_m
            if d.label == "person":
                if g.clipped_bottom:     # feet out of frame: the ground row only bounds it
                    r = min(r, target_fix(d.box, cam, self.cfg.follow).range_m)
                r = max(0.05, r - self.cfg.avoid.person_pad)
            pts.extend(spread(r, g.bearing, g.half_angle, "camera:" + d.label))
        return pts
