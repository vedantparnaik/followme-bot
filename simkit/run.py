"""Headless scenario runner.

    python -m simkit.run                       # every scenario, hardware plant, sonar
    python -m simkit.run -s pillar -v          # one scenario, print the event log
    python -m simkit.run --plant ideal         # perfect motors
    python -m simkit.run --sensor none         # camera only
    python -m simkit.run -s obstacles --gif docs/media/obstacles.gif
"""
from __future__ import annotations

import argparse
import sys

from simkit.scenarios import SCENARIOS
from simkit.sim import Sim


def run_one(name, plant, profile, sensor, seed, record=False):
    sc = SCENARIOS[name](seed)
    sim = Sim(sc, profile=profile, plant=plant, sensor=sensor, seed=seed, record=record)
    return sim, sim.run()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-s", "--scenario", default="all", choices=["all", *SCENARIOS])
    ap.add_argument("--plant", default="hardware", choices=["hardware", "ideal"])
    ap.add_argument("--profile", default=None, choices=["hardware", "ideal"],
                    help="brain tuning (defaults to the plant)")
    ap.add_argument("--sensor", default="sonar3", choices=["sonar3", "lidar", "none"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="run seeds seed..seed+N-1")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the event log")
    ap.add_argument("--gif", help="write a top-down animation (one scenario)")
    a = ap.parse_args(argv)
    profile = a.profile or a.plant
    names = list(SCENARIOS) if a.scenario == "all" else [a.scenario]

    cols = ["pass", "collisions", "person_contacts", "min_clear_m", "min_person_m",
            "follow_s", "wrong_s", "reid", "range_err_m", "max_dist_m", "final"]
    print(f"plant={a.plant} brain={profile} sensor={a.sensor}")
    print(f"{'scenario':10s} {'seed':>4s} " + " ".join(f"{c:>15s}" for c in cols))
    ok = True
    for name in names:
        for seed in range(a.seed, a.seed + a.seeds):
            sim, m = run_one(name, a.plant, profile, a.sensor, seed, record=bool(a.gif))
            row = m.row()
            ok &= row["pass"]
            print(f"{name:10s} {seed:4d} " + " ".join(f"{str(row[c]):>15s}" for c in cols))
            if a.verbose:
                for e in m.events:
                    print("    " + e)
                print("    time in state: " + ", ".join(
                    f"{k} {v:.1f}s" for k, v in sorted(m.state_s.items(), key=lambda kv: -kv[1])))
            if a.gif:
                from simkit.render import write_gif
                write_gif(sim, a.gif)
                print(f"wrote {a.gif}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
