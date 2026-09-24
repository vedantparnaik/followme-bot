"""Drive models: left/right duty -> pose.

``IdealPlant``    duty maps straight to wheel speed (tiny lag). Perfect motors.
``HardwarePlant`` the real drive layer and chassis, as measured:
                  ~250 ms command dead time (camera -> Wi-Fi -> YOLO -> motor),
                  a 55% stiction kick for 0.18 s whenever a side starts from
                  rest, minimum duty floors (18% rolling, 26% pivoting) and a
                  400%/s slew. Same numbers as ``robot/pi/fpv_server.py``.
"""
from __future__ import annotations

import math
from collections import deque

from followme.config import Body


def _approach(cur, goal, step):
    return min(cur + step, goal) if goal > cur else max(cur - step, goal)


class _Base:
    def __init__(self, body: Body, x=0.0, y=0.0, yaw=0.0):
        self.body = body
        self.x, self.y, self.yaw = x, y, yaw
        self.vx = self.wz = 0.0
        self.applied = (0.0, 0.0)

    @property
    def pose(self):
        return self.x, self.y, self.yaw

    def _integrate(self, al, ar, dt):
        b = self.body
        vl, vr = al / 100.0 * b.v_at_100, ar / 100.0 * b.v_at_100
        self.vx = 0.5 * (vl + vr)
        self.wz = (vr - vl) / b.track
        self.yaw += self.wz * dt
        self.x += self.vx * math.cos(self.yaw) * dt
        self.y += self.vx * math.sin(self.yaw) * dt
        self.applied = (al, ar)

    def halt(self):
        """Hit something: the wheels stall."""
        self.vx = self.wz = 0.0


class IdealPlant(_Base):
    TAU = 0.05

    def __init__(self, body: Body, x=0.0, y=0.0, yaw=0.0):
        super().__init__(body, x, y, yaw)
        self._l = self._r = 0.0

    def step(self, l, r, dt, now):
        a = min(1.0, dt / self.TAU)
        self._l += a * (l - self._l)
        self._r += a * (r - self._r)
        self._integrate(self._l, self._r, dt)


class _Side:
    def __init__(self, p):
        self.p = p
        self.applied = 0.0
        self.kick_until = 0.0
        self.zero_since = -1e9

    def update(self, target, dt, now):
        p = self.p
        if target == 0.0:
            if self.applied != 0.0:
                self.zero_since = now
            self.applied = 0.0
            self.kick_until = 0.0
        elif self.applied != 0.0 and target * self.applied < 0:
            self.applied = _approach(self.applied, 0.0, p.SLEW * dt)
            if abs(self.applied) < 1.0:
                self.applied = 0.0
                self.zero_since = now
        elif self.applied == 0.0:
            if now - self.zero_since >= p.KICK_REARM_S:
                self.kick_until = now + p.KICK_S
                self.applied = p.KICK_DUTY if target > 0 else -p.KICK_DUTY
            else:
                self.applied = _approach(0.0, target, p.SLEW * dt)
        elif now < self.kick_until:
            self.applied = p.KICK_DUTY if target > 0 else -p.KICK_DUTY
        else:
            self.applied = _approach(self.applied, target, p.SLEW * dt)
        return self.applied


class HardwarePlant(_Base):
    CMD_DELAY = 0.25
    MIN_DUTY = 18.0
    TURN_MIN_DUTY = 26.0
    KICK_DUTY = 55.0
    KICK_S = 0.18
    KICK_REARM_S = 0.5
    SLEW = 400.0

    def __init__(self, body: Body, x=0.0, y=0.0, yaw=0.0, delay=None):
        super().__init__(body, x, y, yaw)
        if delay is not None:
            self.CMD_DELAY = delay
        self.q = deque()
        self.cur = (0.0, 0.0)
        self.left, self.right = _Side(self), _Side(self)

    @staticmethod
    def shape(cmd, floor):
        if abs(cmd) < 1.0:
            return 0.0
        mag = floor + (min(abs(cmd), 100.0) / 100.0) * (100.0 - floor)
        return mag if cmd > 0 else -mag

    def step(self, l, r, dt, now):
        self.q.append((now, l, r))
        while self.q and now - self.q[0][0] >= self.CMD_DELAY:
            _, a, b = self.q.popleft()
            self.cur = (a, b)
        cl, cr = self.cur
        floor = self.TURN_MIN_DUTY if cl * cr < 0 else self.MIN_DUTY
        al = self.left.update(self.shape(cl, floor), dt, now)
        ar = self.right.update(self.shape(cr, floor), dt, now)
        self._integrate(al, ar, dt)


PLANTS = {"ideal": IdealPlant, "hardware": HardwarePlant}
