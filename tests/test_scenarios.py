"""End to end: every scenario, headless, must end on the right person with no
collisions and no person contacts."""
import pytest

from simkit.scenarios import SCENARIOS
from simkit.sim import Sim

CASES = [("hardware", "sonar3"), ("hardware", "lidar"), ("ideal", "sonar3")]


@pytest.mark.parametrize("plant,sensor", CASES)
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("seed", [0, 1])
def test_scenario_passes(scenario, plant, sensor, seed):
    m = Sim(SCENARIOS[scenario](seed), seed=seed, plant=plant, profile=plant, sensor=sensor).run()
    assert m.collisions == 0
    assert m.person_contacts == 0
    assert m.wrong_follow_s < 0.5
    assert m.passed(), m.row()


def test_camera_only_follows_without_obstacles():
    m = Sim(SCENARIOS["crossing"](0), seed=0, sensor="none").run()
    assert m.passed(), m.row()
