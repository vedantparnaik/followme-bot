"""Steering laws: (range error, heading) -> left/right duty.

``ArcDrive`` is for the real skid-steer. Small heading errors cannot be
turned into small pivots on this chassis: a pivot from rest fires the Pi's
stiction kick and spins ~30 degrees. So the rover only turns by curving while
it rolls (both sides same sign, inner side slower). In the standoff band it
creeps forward on a curve instead of pivoting. Pivots are reserved for blind
search, as short nudges each followed by a mandatory settle so the camera can
catch up with ~250 ms of dead time.

``SmoothDrive`` is for perfect motors: proportional speed and yaw rate.
"""
from __future__ import annotations

from typing import Tuple

from followme.config import Config


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class _Output:
    """EMA + slew on duties. Stops are immediate, never ramped."""

    def __init__(self, cfg: Config, smooth: bool):
        self.cfg, self.smooth = cfg, smooth
        self.reset()

    def reset(self) -> None:
        self.ema_l = self.ema_r = 0.0
        self.l = self.r = 0.0

    def __call__(self, l: float, r: float, dt: float) -> Tuple[float, float]:
        if not self.smooth:
            self.l, self.r = l, r
            return l, r
        s = self.cfg.steer
        a = s.out_smooth
        self.ema_l = (1 - a) * self.ema_l + a * l
        self.ema_r = (1 - a) * self.ema_r + a * r
        if l == 0.0 and r == 0.0 and abs(self.ema_l) < 1.0 and abs(self.ema_r) < 1.0:
            self.ema_l = self.ema_r = 0.0
            self.l = self.r = 0.0
            return 0.0, 0.0
        step = s.cmd_slew * dt
        self.l = _clamp(self.ema_l, self.l - step, self.l + step)
        self.r = _clamp(self.ema_r, self.r - step, self.r + step)
        return self.l, self.r

    def stop(self) -> None:
        self.reset()


class ArcDrive:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out = _Output(cfg, smooth=True)
        self.reset()

    def reset(self) -> None:
        self.err_f = 0.0
        self.yaw_until = self.settle_until = -1e9
        self.yaw_dir = 0.0
        self.pulses_left = 0
        self.out.reset()

    def _arc(self, speed: float, err: float) -> Tuple[float, float]:
        """Both sides same sign. Goal to the right (err>0): right side slower."""
        s = self.cfg.steer
        if speed == 0.0:
            return 0.0, 0.0
        mix = _clamp(err * s.kp_turn, -1.0, 1.0)
        inner = max(s.inner_min, 1.0 - abs(mix))
        if err > s.x_dead:
            return speed, speed * inner
        if err < -s.x_dead:
            return speed * inner, speed
        return speed, speed

    def follow(self, now: float, dt: float, range_err: float, heading: float,
               speed_scale: float, close_fill: bool = False) -> Tuple[float, float, str]:
        c, s = self.cfg, self.cfg.steer
        # Normalise like an image error: +1 = right edge of the frame.
        steer_err = _clamp(-heading / (c.camera.hfov / 2.0), -1.0, 1.0)
        derr = 0.0 if abs(range_err) < c.follow.dist_dead else range_err
        fwd = _clamp(c.follow.kp_fwd * derr, -c.follow.max_reverse, c.follow.max_speed)
        if close_fill:
            fwd = min(fwd, -0.5 * c.follow.max_reverse)
        if fwd > 0:
            fwd *= speed_scale
        self.err_f = (1 - s.err_smooth) * self.err_f + s.err_smooth * steer_err

        if abs(fwd) >= 1.0:
            l, r = self._arc(fwd, self.err_f)
            note = "arc"
        elif abs(self.err_f) > s.x_dead:
            if range_err < -0.25:
                l, r = self._arc(-s.creep * 0.6, self.err_f)
                note = "back-arc"
            elif speed_scale > 0.0:
                l, r = self._arc(s.creep * max(speed_scale, 0.6), self.err_f)
                note = "creep-arc"
            else:
                l = r = 0.0
                note = "hold"
        else:
            l = r = 0.0
            note = "hold"
        l, r = self.out(l, r, dt)
        return l, r, note

    def begin_search(self, now: float, direction: float) -> None:
        self.yaw_dir = 1.0 if direction >= 0 else -1.0
        self.pulses_left = self.cfg.search.pulses
        self.yaw_until = self.settle_until = -1e9

    def search(self, now: float, dt: float) -> Tuple[float, float, bool, str]:
        """Blind nudges toward where they left. Returns (l, r, done, note)."""
        sr = self.cfg.search
        if now < self.yaw_until:
            l, r = -sr.yaw_cmd * self.yaw_dir, sr.yaw_cmd * self.yaw_dir
            note = "nudge"
        elif now < self.settle_until:
            l = r = 0.0
            note = "settle"
        elif self.pulses_left > 0:
            self.pulses_left -= 1
            self.yaw_until = now + sr.yaw_s
            self.settle_until = self.yaw_until + sr.settle_s
            l, r = -sr.yaw_cmd * self.yaw_dir, sr.yaw_cmd * self.yaw_dir
            note = "nudge"
        else:
            self.out.stop()
            return 0.0, 0.0, True, "search done"
        # Pulses go straight to the Pi: the kick is the point, don't smooth it.
        self.out.stop()
        return l, r, False, note

    def stop(self) -> None:
        self.out.stop()
        self.err_f = 0.0


class SmoothDrive:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.search_dir = 0.0
        self.search_t0 = -1e9

    def _duties(self, v: float, w: float) -> Tuple[float, float]:
        b = self.cfg.body
        vl = v - w * b.track / 2.0
        vr = v + w * b.track / 2.0
        k = 100.0 / b.v_at_100
        cap = self.cfg.follow.max_speed
        l, r = vl * k, vr * k
        m = max(abs(l), abs(r))
        if m > cap:
            l, r = l * cap / m, r * cap / m
        return l, r

    def follow(self, now: float, dt: float, range_err: float, heading: float,
               speed_scale: float, close_fill: bool = False) -> Tuple[float, float, str]:
        c, s = self.cfg, self.cfg.steer
        derr = 0.0 if abs(range_err) < 0.5 * c.follow.dist_dead else range_err
        v_rev = c.follow.max_reverse / 100.0 * c.body.v_at_100
        v = _clamp(s.kv * derr, -v_rev, s.v_max)
        if close_fill:
            v = min(v, -0.5 * v_rev)
        if v > 0:
            v *= speed_scale
        w = _clamp(s.kw * heading, -s.w_max, s.w_max)
        l, r = self._duties(v, w)
        note = "track" if (v or w) else "hold"
        return l, r, note

    def begin_search(self, now: float, direction: float) -> None:
        self.search_dir = 1.0 if direction >= 0 else -1.0
        self.search_t0 = now

    def search(self, now: float, dt: float) -> Tuple[float, float, bool, str]:
        sr = self.cfg.search
        if (now - self.search_t0) * sr.w >= sr.max_angle:
            return 0.0, 0.0, True, "search done"
        l, r = self._duties(0.0, sr.w * self.search_dir)
        return l, r, False, "rotate"

    def stop(self) -> None:
        pass


def make_drive(cfg: Config):
    return SmoothDrive(cfg) if cfg.steer.mode == "smooth" else ArcDrive(cfg)
