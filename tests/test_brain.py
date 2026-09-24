import math

from followme import Brain, State
from followme.obstacles import ObstaclePoint
from tests.conftest import person


def steps(brain, dets_fn, seconds, t0=0.0, obstacles=(), odom=(0.0, 0.0, 0.0)):
    t, out = t0, None
    while t < t0 + seconds:
        out = brain.step(t, dets_fn(t), obstacles, odom=odom)
        t += 1 / 30
    return out, t


def armed(cfg, r=4.0):
    b = Brain(cfg)
    b.arm()
    out, t = steps(b, lambda t: [person(r)], 1.0)
    return b, out, t


def test_idle_until_armed(cfg):
    b = Brain(cfg)
    out, _ = steps(b, lambda t: [person(4.0)], 1.0)
    assert out.state == State.IDLE and out.l == out.r == 0
    assert "would follow" in out.note


def test_follows_forward_when_far(cfg):
    b, out, t = armed(cfg, 4.0)
    assert out.state == State.FOLLOW
    assert out.l > 0 and out.r > 0


def test_holds_at_target_distance(cfg):
    b, out, t = armed(cfg, cfg.follow.target_dist)
    out, _ = steps(b, lambda t: [person(cfg.follow.target_dist)], 1.0, t0=t)
    assert out.state == State.FOLLOW and abs(out.l) < 1 and abs(out.r) < 1


def test_estop_is_latched(cfg):
    b, out, t = armed(cfg)
    b.estop()
    b.arm()
    out, t = steps(b, lambda t: [person(4.0)], 0.5, t0=t)
    assert out.state == State.ESTOP and out.l == out.r == 0
    b.reset()
    b.arm()
    out, _ = steps(b, lambda t: [person(4.0)], 1.0, t0=t)
    assert out.state == State.FOLLOW


def test_blocked_in_a_dead_end(cfg):
    b, out, t = armed(cfg)
    w = [ObstaclePoint(0.6, y / 20, "sonar") for y in range(-10, 11)]
    w += [ObstaclePoint(x / 20, s * 0.5, "sonar") for x in range(-6, 13) for s in (-1, 1)]
    out, _ = steps(b, lambda t: [person(4.0)], 0.5, t0=t, obstacles=w)
    assert out.state == State.BLOCKED


def test_lost_at_frame_edge_searches_that_way(cfg):
    bearing = math.radians(25)
    b = Brain(cfg)
    b.arm()
    out, t = steps(b, lambda t: [person(3.5, bearing)], 1.0)
    out, _ = steps(b, lambda t: [], 1.5, t0=t)
    assert out.state == State.SEARCH
    assert out.r >= out.l          # turning left, toward where they went


def test_lost_straight_ahead_pursues_last_seen_point(cfg):
    b, out, t = armed(cfg)
    out, _ = steps(b, lambda t: [], 1.5, t0=t)
    assert out.state == State.PURSUE
    assert out.l > 0 and out.r > 0
