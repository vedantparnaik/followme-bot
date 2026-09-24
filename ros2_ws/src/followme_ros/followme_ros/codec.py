"""JSON on std_msgs/String between the world and the brain.

Detections and range points are small and change shape often while
developing; JSON keeps the package free of custom message builds. Swap for
vision_msgs / sensor_msgs later without touching the core.
"""
from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np

from followme.obstacles import ObstaclePoint, RangeCone
from followme.perception import Detection


def encode_detections(t: float, dets: List[Detection]) -> str:
    return json.dumps({"t": t, "dets": [
        {"box": [round(v, 2) for v in d.box], "id": d.track_id, "label": d.label,
         "conf": round(d.conf, 3),
         "feat": None if d.feature is None else [round(float(v), 4) for v in d.feature]}
        for d in dets]})


def decode_detections(s: str) -> Tuple[float, List[Detection]]:
    m = json.loads(s)
    return m["t"], [Detection(box=tuple(d["box"]), track_id=d["id"], label=d["label"],
                              conf=d["conf"],
                              feature=None if d["feat"] is None else np.array(d["feat"]))
                    for d in m["dets"]]


def encode_range(t: float, pts: List[ObstaclePoint], cones: List[RangeCone]) -> str:
    return json.dumps({"t": t,
                       "pts": [[round(p.x, 3), round(p.y, 3), p.source] for p in pts],
                       "cones": [[c.x, c.y, round(c.angle, 4), round(c.half, 4),
                                  round(c.range_m, 3)] for c in cones]})


def decode_range(s: str) -> Tuple[float, List[ObstaclePoint], List[RangeCone]]:
    m = json.loads(s)
    return (m["t"], [ObstaclePoint(x, y, src) for x, y, src in m["pts"]],
            [RangeCone(*c) for c in m["cones"]])
