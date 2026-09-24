import numpy as np

from followme.reid import Gallery, similarity
from followme.target import TargetTracker
from tests.conftest import feature, person


def run(tracker, frames, t0=0.0, dt=1 / 30):
    t, res = t0, None
    for dets in frames:
        res = tracker.update(dets, t)
        t += dt
    return res, t


def locked_tracker(cfg, seconds=2.0, look=1):
    tr = TargetTracker(cfg)
    rng = np.random.default_rng(0)
    res, t = run(tr, [[person(3.0, look=look, noise=0.5, rng=rng)]] * int(seconds * 30))
    assert tr.locked and res.target_id == 1
    return tr, t


def test_gallery_prefers_the_same_look():
    g = Gallery()
    for _ in range(5):
        g.add(feature(1))
    assert g.score(feature(1)) > 0.99
    assert g.score(feature(2)) < g.score(feature(1))
    assert abs(similarity(feature(3), feature(3)) - 1.0) < 1e-9


def test_lock_needs_a_stable_candidate(cfg):
    tr = TargetTracker(cfg)
    res, _ = run(tr, [[person(3.0)]] * 3)
    assert not tr.locked
    res, _ = run(tr, [[person(3.0)]] * 30, t0=0.1)
    assert tr.locked and res.target_id == 1


def test_teleport_is_an_id_swap(cfg):
    tr, t = locked_tracker(cfg)
    res = tr.update([person(1.8, tid=1)], t)          # same ID, 1.2 m nearer in one frame
    assert res.event.startswith("id swap")
    assert tr.target_id is None and 1 in tr.known_others


def test_appearance_mismatch_is_an_id_swap(cfg):
    tr, t = locked_tracker(cfg)
    res, _ = run(tr, [[person(3.0, tid=1, look=7)]] * 10, t0=t)
    assert tr.target_id is None


def test_reid_same_person_new_id(cfg):
    tr, t = locked_tracker(cfg)
    run(tr, [[]] * 60, t0=t)                            # 2 s gone
    res, _ = run(tr, [[person(3.5, tid=9, look=1)]] * 10, t0=t + 2.0)
    assert res.target_id == 9 and res.how in ("reid", "id")


def test_never_locks_a_stranger_after_loss(cfg):
    tr, t = locked_tracker(cfg)
    run(tr, [[]] * 60, t0=t)
    res, _ = run(tr, [[person(3.0, tid=5, look=4)]] * 90, t0=t + 2.0)
    assert res.how == "lost" and res.det is None
