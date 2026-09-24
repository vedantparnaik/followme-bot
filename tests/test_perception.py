import math

import pytest

from followme.config import Camera, Follow
from followme.perception import ground_fix, target_fix
from tests.conftest import person_box

CAM, FOL = Camera(), Follow()


@pytest.mark.parametrize("r", [3.0, 4.0, 6.0])
def test_range_from_box_height(r):
    fix = target_fix(person_box(r), CAM, FOL)
    assert fix.range_m == pytest.approx(r, abs=0.05)
    assert fix.clipped == ""


def test_bearing_sign_left_is_positive():
    fix = target_fix(person_box(3.0, math.radians(15)), CAM, FOL)
    assert fix.bearing == pytest.approx(math.radians(15), abs=1e-3)
    assert fix.err_x < 0          # left of centre in the image


def test_clipped_head_falls_back_to_width():
    x, y, w, h = person_box(1.4)
    cut = (x, 0.0, w, h + y)      # head above the frame
    fix = target_fix(cut, CAM, FOL)
    assert fix.clipped == "head"
    assert fix.range_m == pytest.approx(1.4, abs=0.05)


def test_ground_fix_from_bottom_row():
    g = ground_fix(person_box(2.5), CAM)
    assert g.range_m == pytest.approx(2.5, abs=0.02)
    assert not g.clipped_bottom


def test_ground_fix_above_horizon_is_none():
    assert ground_fix((300, 100, 40, 100), CAM) is None
