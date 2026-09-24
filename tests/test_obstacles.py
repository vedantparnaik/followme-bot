import math

import numpy as np

from followme.obstacles import (Memory, ObstaclePoint, Planner, RangeCone, corridor_clearance,
                                mask_target)


def wall(x, y0, y1, step=0.05, source="sonar"):
    n = int(round((y1 - y0) / step)) + 1
    return [ObstaclePoint(x, y0 + i * step, source) for i in range(n)]


def test_corridor_clearance():
    pts = np.array([[2.0, 0.0], [1.0, 1.0]])
    assert math.isclose(corridor_clearance(pts, 0.0, 0.3, 0.3, 3.0), 1.7)
    assert corridor_clearance(np.empty((0, 2)), 0.0, 0.3, 0.3, 3.0) == 3.0


def test_free_space_goes_straight(cfg):
    plan = Planner(cfg).plan(0.1, [], 0.0)
    assert plan.heading == 0.1 and plan.speed_scale == 1.0 and not plan.blocked


def test_detours_around_a_box_on_the_right(cfg):
    pts = wall(1.6, -0.6, 0.1)                 # box ahead, sticking out to the left a bit
    plan = Planner(cfg).plan(0.0, pts, 0.0, goal_range=4.0)
    assert not plan.blocked and plan.avoiding
    assert plan.heading > 0                   # goes left, the short way round


def test_blocked_in_a_dead_end(cfg):
    pts = wall(0.6, -0.5, 0.5)
    pts += [ObstaclePoint(x / 20, s * 0.5) for x in range(-6, 13) for s in (-1, 1)]
    plan = Planner(cfg).plan(0.0, pts, 0.0, goal_range=4.0)
    assert plan.blocked and plan.speed_scale == 0.0


def test_unseen_headings_are_not_free(cfg):
    pts = wall(0.7, -0.8, 0.8)
    wide = Planner(cfg).plan(0.0, pts, 0.0, goal_range=4.0)
    narrow = Planner(cfg).plan(0.0, pts, 0.0, goal_range=4.0, seen_half=math.radians(20))
    assert not wide.blocked
    assert narrow.blocked


def test_target_is_masked():
    pts = [ObstaclePoint(3.0, 0.0), ObstaclePoint(1.0, 0.0)]
    left = mask_target(pts, (3.0, 0.1), 0.5)
    assert left == [pts[1]]


def test_memory_keeps_points_while_turning_and_forgets_them_later():
    m = Memory(keep_s=2.0)
    m.update(0.0, (0.0, 0.0, 0.0), [ObstaclePoint(1.0, 0.0)])
    out = m.update(0.5, (0.0, 0.0, math.pi / 2), [])     # turned left 90 deg
    assert len(out) == 1
    assert abs(out[0].x) < 0.05 and abs(out[0].y + 1.0) < 0.05
    assert m.update(3.0, (0.0, 0.0, 0.0), []) == []


def test_memory_clears_what_a_sensor_sees_through():
    m = Memory(keep_s=5.0)
    m.update(0.0, (0.0, 0.0, 0.0), [ObstaclePoint(1.0, 0.0)])
    cone = RangeCone(0.3, 0.0, 0.0, math.radians(10), 4.0)   # reads 4 m: nothing there
    assert m.update(0.1, (0.0, 0.0, 0.0), [], free=[cone]) == []


def test_camera_people_are_not_remembered():
    m = Memory(keep_s=5.0)
    m.update(0.0, (0.0, 0.0, 0.0), [ObstaclePoint(1.0, 0.0, "camera:person")])
    assert m.update(0.1, (0.0, 0.0, 0.0), []) == []
