#!/usr/bin/env python3
"""Fake Pi: serves a video file as the camera stream and accepts /cmd.

For testing the laptop app on recorded footage:

    python tools/fake_pi.py --video walk.mov           # serves :8000
    FOLLOWME_PI=127.0.0.1:8000 python robot/mac/follow_server.py

Commands are only logged (the video can't react to them). /odom integrates
the commanded duty the same way the Pi does.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

V_AT_100, TRACK_EFF, DEADMAN_S = 1.35, 0.46, 0.25

_frame = {"jpg": None, "n": 0}
_cond = threading.Condition()
_cmd = {"l": 0.0, "r": 0.0, "t": 0.0}
_odom = {"x": 0.0, "y": 0.0, "yaw": 0.0, "t": 0.0}


def video_loop(path: str, width: int, height: int):
    while True:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / min(fps, 30.0)
        while True:
            t0 = time.time()
            ok, img = cap.read()
            if not ok:
                break
            img = cv2.resize(img, (width, height))
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _cond:
                    _frame["jpg"], _frame["n"] = enc.tobytes(), _frame["n"] + 1
                    _cond.notify_all()
            time.sleep(max(0.0, period - (time.time() - t0)))
        cap.release()


def odom_loop():
    last = time.time()
    while True:
        now = time.time()
        dt, last = now - last, now
        l, r = (_cmd["l"], _cmd["r"]) if now - _cmd["t"] < DEADMAN_S else (0.0, 0.0)
        vl, vr = l / 100 * V_AT_100, r / 100 * V_AT_100
        v, w = (vl + vr) / 2, (vr - vl) / TRACK_EFF
        _odom["yaw"] += w * dt
        _odom["x"] += v * math.cos(_odom["yaw"]) * dt
        _odom["y"] += v * math.sin(_odom["yaw"]) * dt
        _odom["t"] = now
        time.sleep(0.02)


class Handler(BaseHTTPRequestHandler):
    log_cmds = False

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/cmd":
            q = parse_qs(u.query)
            l, r = float(q.get("l", ["0"])[0]), float(q.get("r", ["0"])[0])
            _cmd.update(l=l, r=r, t=time.time())
            if self.log_cmds and (l or r):
                print(f"cmd l={l:6.1f} r={r:6.1f}", flush=True)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        elif u.path == "/odom":
            self._json(dict(_odom, applied_l=_cmd["l"], applied_r=_cmd["r"]))
        elif u.path == "/range":
            self._json({"error": "fake pi has no range sensors"}, 404)
        elif u.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = 0
            try:
                while True:
                    with _cond:
                        while _frame["n"] == last:
                            _cond.wait(timeout=5)
                        jpg, last = _frame["jpg"], _frame["n"]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--log-cmds", action="store_true")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    Handler.log_cmds = a.log_cmds
    threading.Thread(target=video_loop, args=(a.video, w, h), daemon=True).start()
    threading.Thread(target=odom_loop, daemon=True).start()
    print(f"fake pi on http://127.0.0.1:{a.port}/  (video: {a.video})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
