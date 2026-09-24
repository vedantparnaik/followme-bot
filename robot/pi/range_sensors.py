"""Three HC-SR04 ultrasonic sensors: front-left, front, front-right.

Optional. They cover the front arc for things the camera can't classify
(pallets, low boxes, glass doors).

Wiring (BCM numbering; none of these clash with the motor pins):

    sensor        angle   TRIG   ECHO
    front-left    +30 deg  23     24
    front          0 deg   17     27
    front-right   -30 deg  22     25

The HC-SR04 ECHO pin is 5 V. The Pi's GPIO is 3.3 V and is NOT 5 V tolerant:
put a divider on every ECHO line (1 kOhm from ECHO to the GPIO pin, 2 kOhm from
the GPIO pin to ground). VCC 5 V, GND common with the Pi.

Sensors fire one at a time so they don't hear each other's pings. A full
sweep of three is ~40-75 ms, so readings arrive at ~15 Hz. Each reading is
the median of the last three, which removes the occasional ghost echo.

Enable in fpv_server.py with FOLLOWME_SONAR=1.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List

SPEED_OF_SOUND = 343.0      # m/s at ~20 C
MAX_RANGE_M = 4.0
ECHO_TIMEOUT_S = 2 * MAX_RANGE_M / SPEED_OF_SOUND + 0.002
GAP_S = 0.012               # between pings: let the last echo die out
FORWARD_M = 0.31            # sensor face ahead of the chassis centre
HALF_ANGLE_DEG = 12.0       # usable beam half-width


@dataclass
class Sonar:
    name: str
    angle_deg: float
    trig: int
    echo: int


DEFAULT_LAYOUT = [
    Sonar("front-left", 30.0, 23, 24),
    Sonar("front", 0.0, 17, 27),
    Sonar("front-right", -30.0, 22, 25),
]


class RangeSensors:
    def __init__(self, layout: List[Sonar] = DEFAULT_LAYOUT):
        import RPi.GPIO as GPIO    # only on the Pi
        self.GPIO = GPIO
        self.layout = layout
        for s in layout:
            GPIO.setup(s.trig, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(s.echo, GPIO.IN)
        self._hist = {s.name: deque(maxlen=3) for s in layout}
        self._lock = threading.Lock()
        self._last = {s.name: (MAX_RANGE_M, 0.0) for s in layout}
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def close(self):
        self._stop.set()

    def _ping(self, s: Sonar) -> float:
        G = self.GPIO
        G.output(s.trig, G.HIGH)
        time.sleep(10e-6)
        G.output(s.trig, G.LOW)
        t0 = time.perf_counter()
        while G.input(s.echo) == 0:
            if time.perf_counter() - t0 > ECHO_TIMEOUT_S:
                return MAX_RANGE_M
        start = time.perf_counter()
        while G.input(s.echo) == 1:
            if time.perf_counter() - start > ECHO_TIMEOUT_S:
                return MAX_RANGE_M
        r = (time.perf_counter() - start) * SPEED_OF_SOUND / 2.0
        return min(max(r, 0.02), MAX_RANGE_M)

    def _loop(self):
        while not self._stop.is_set():
            for s in self.layout:
                r = self._ping(s)
                h = self._hist[s.name]
                h.append(r)
                med = sorted(h)[len(h) // 2]
                with self._lock:
                    self._last[s.name] = (med, time.time())
                time.sleep(GAP_S)

    def read(self) -> dict:
        """JSON-ready. Angles in radians, + left; ranges in metres from the
        sensor face; ``max_r`` means nothing was heard."""
        with self._lock:
            last = dict(self._last)
        return {
            "t": time.time(),
            "forward_m": FORWARD_M,
            "half_angle": math.radians(HALF_ANGLE_DEG),
            "max_r": MAX_RANGE_M,
            "readings": [{"name": s.name, "angle": math.radians(s.angle_deg),
                          "r": round(last[s.name][0], 3), "age": round(time.time() - last[s.name][1], 3)}
                         for s in self.layout],
        }
