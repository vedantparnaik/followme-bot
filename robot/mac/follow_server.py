#!/usr/bin/env python3
"""Laptop side of the rover.

Runs YOLO + ByteTrack on the Pi's camera stream, computes colour features,
reads /odom and /range from the Pi, and feeds it all to ``followme.Brain``.
The duty it returns goes back to the Pi's /cmd.

    FOLLOWME_PI=raspberrypi.local:8000 python robot/mac/follow_server.py

UI at http://127.0.0.1:8080/. Non-person YOLO classes (chairs, bags, bikes)
are passed in as camera obstacles; anything YOLO has no class for needs the
ultrasonics.

Safety:
- starts disarmed (the brain runs, but 0,0 is sent)
- E-STOP latches until "Reset e-stop"
- when the target is lost, PURSUE/SEARCH are bounded and then it waits in
  place; only "Re-lock" picks a new person
- the Pi stops the motors if this app dies or the Wi-Fi drops (0.25 s deadman)
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from features import person_feature, to_hsv            # noqa: E402
from followme import Brain, Detection, State, hardware  # noqa: E402
from followme.obstacles import RangeCone, spread        # noqa: E402

# config
PORT = int(os.environ.get("FOLLOWME_PORT", "8080"))
PI = os.environ.get("FOLLOWME_PI", "raspberrypi.local:8000").replace("http://", "").rstrip("/")
MODEL = os.environ.get("FOLLOWME_MODEL", "yolov8n.pt")
CONF = 0.40
IMGSZ = 416
SEND_HZ = 20.0
RANGE_HZ, ODOM_HZ = 10.0, 20.0
STALE_S = 0.4
TUNE_PATH = os.path.join(HERE, "follow_tune.json")

cfg = hardware()
cfg.follow.max_speed = 15.0     # start gentle on real hardware; raise from the UI


def load_tune():
    try:
        with open(TUNE_PATH) as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"ignoring {TUNE_PATH}: {e}", flush=True)


def save_tune():
    keys = ("camera.fx", "camera.fy", "follow.target_dist", "follow.max_speed",
            "steer.kp_turn", "steer.x_dead", "avoid.enabled", "avoid.stop_dist")
    flat = cfg.flat()
    with open(TUNE_PATH, "w") as f:
        json.dump({k: flat[k] for k in keys}, f, indent=2)


load_tune()
brain = Brain(cfg)
_brain_lock = threading.Lock()
_stop = threading.Event()
print(f"Pi at {PI}", flush=True)


def fetch_json(path, timeout=0.5):
    with urllib.request.urlopen(f"http://{PI}{path}", timeout=timeout) as r:
        return json.loads(r.read())


# inputs from the Pi
_raw_lock = threading.Lock()
_raw = {"img": None, "n": 0}


def mjpeg_reader():
    while not _stop.is_set():
        try:
            with urllib.request.urlopen(f"http://{PI}/stream", timeout=5) as r:
                buf = b""
                while not _stop.is_set():
                    chunk = r.read(8192)
                    if not chunk:
                        break
                    buf += chunk
                    a = buf.find(b"\xff\xd8")
                    b = buf.find(b"\xff\xd9", a + 2) if a != -1 else -1
                    if a != -1 and b != -1:
                        jpg, buf = buf[a:b + 2], buf[b + 2:]
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            with _raw_lock:
                                _raw["img"], _raw["n"] = img, _raw["n"] + 1
                    if len(buf) > 4_000_000:
                        buf = b""
        except Exception:
            time.sleep(0.5)


_side = {"odom": None, "odom_t": 0.0, "range": None, "range_t": 0.0, "has_range": None}


def odom_poller():
    while not _stop.is_set():
        try:
            o = fetch_json("/odom")
            _side["odom"], _side["odom_t"] = (o["x"], o["y"], o["yaw"]), time.time()
        except Exception:
            pass
        time.sleep(1.0 / ODOM_HZ)


def range_poller():
    while not _stop.is_set():
        try:
            _side["range"], _side["range_t"] = fetch_json("/range"), time.time()
            _side["has_range"] = True
        except urllib.error.HTTPError:
            _side["has_range"] = False      # Pi runs without sonar: stop asking often
            time.sleep(5.0)
        except Exception:
            pass
        time.sleep(1.0 / RANGE_HZ)


def range_inputs():
    """Pi /range JSON -> (points, free cones) in the rover frame."""
    rg = _side["range"]
    if rg is None or time.time() - _side["range_t"] > STALE_S:
        return [], []
    fwd, half, max_r = rg["forward_m"], rg["half_angle"], rg["max_r"]
    pts, cones = [], []
    for s in rg["readings"]:
        a, r = s["angle"], s["r"]
        cones.append(RangeCone(fwd, 0.0, a, half, r))
        if r < max_r - 0.05:
            for p in spread(r, a, half, "sonar"):
                p.x += fwd
                pts.append(p)
    return pts, cones


def current_odom():
    if _side["odom"] is None or time.time() - _side["odom_t"] > STALE_S:
        return None
    return _side["odom"]


# output to the Pi
class Commander:
    def __init__(self):
        self.lock = threading.Lock()
        self.l = self.r = 0.0
        self.online = False
        threading.Thread(target=self._loop, daemon=True).start()

    def set(self, l, r):
        with self.lock:
            self.l, self.r = l, r

    def _loop(self):
        while not _stop.is_set():
            with self.lock:
                l, r = (self.l, self.r) if brain.armed else (0.0, 0.0)
            try:
                q = urlencode({"l": f"{l:.1f}", "r": f"{r:.1f}"})
                urllib.request.urlopen(f"http://{PI}/cmd?{q}", timeout=0.5).read()
                self.online = True
            except Exception:
                self.online = False
            time.sleep(1.0 / SEND_HZ)

    def stop_now(self):
        self.set(0.0, 0.0)
        try:
            urllib.request.urlopen(f"http://{PI}/cmd?l=0&r=0", timeout=0.5).read()
        except Exception:
            pass


commander = Commander()

# perception + brain
_state_lock = threading.Lock()
S = {"state": "IDLE", "note": "starting", "event": "", "events": [], "target_id": None,
     "range_m": None, "bearing_deg": None, "clipped": "", "reid": 0.0, "l": 0.0, "r": 0.0,
     "fps": 0.0, "ndet": 0, "box_h": 0, "n_obstacles": 0}
_out_lock = threading.Lock()
_out = {"jpg": None}

STATE_BGR = {
    State.IDLE: (160, 160, 160), State.ACQUIRE: (60, 220, 230), State.FOLLOW: (90, 220, 60),
    State.AVOID: (40, 160, 255), State.BLOCKED: (40, 60, 255), State.PURSUE: (220, 200, 40),
    State.SEARCH: (255, 150, 40), State.WAIT: (220, 110, 150), State.ESTOP: (40, 0, 255),
}


def detections_from(res, img, names):
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xy)
    hsv = to_hsv(img)
    dets = []
    for b, c, p, tid in zip(xy, cls, conf, ids):
        x0, y0, x1, y1 = (float(v) for v in b[:4])
        box = (x0, y0, x1 - x0, y1 - y0)
        label = names.get(int(c), str(c))
        feat = person_feature(hsv, box) if label == "person" else None
        dets.append(Detection(box=box, track_id=None if tid is None else int(tid),
                              label=label, conf=float(p), feature=feat))
    return dets


def draw(img, dets, out):
    H, W = img.shape[:2]
    for d in dets:
        x, y, w, h = (int(v) for v in d.box)
        col = (90, 90, 90) if d.label == "person" else (60, 60, 160)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 1)
        tag = d.label if d.track_id is None else f"{d.label} #{d.track_id}"
        cv2.putText(img, tag, (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    tgt = next((d for d in dets if out.target_id is not None and d.track_id == out.target_id
                and d.label == "person"), None)
    if tgt is not None and out.fix is not None:
        x, y, w, h = (int(v) for v in tgt.box)
        col = (0, 165, 255) if out.fix.clipped else (0, 230, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
        cv2.putText(img, f"{out.fix.range_m:.1f}m #{out.target_id}", (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.line(img, (W // 2, 0), (W // 2, H), (60, 60, 60), 1)
    _minimap(img, out)
    col = STATE_BGR[out.state]
    hud = f"{out.state.value}  L{out.l:5.1f} R{out.r:5.1f}"
    cv2.putText(img, hud, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)


def _minimap(img, out, size=150, scale=40.0):
    """Top-down inset, rover at the bottom centre: obstacles, the fan, the heading."""
    H, W = img.shape[:2]
    x0, y0 = W - size - 8, H - size - 8
    roi = img[y0:y0 + size, x0:x0 + size]
    roi[:] = (roi * 0.35).astype(np.uint8)
    ox, oy = size // 2, size - 12

    def px(x, y):
        return int(ox - y * scale), int(oy - x * scale)

    if out.plan is not None:
        for h, clr in out.plan.fan:
            c = (0, 110, 0) if clr > 1.0 else (0, 90, 160)
            cv2.line(roi, px(0.31 * math.cos(h), 0.31 * math.sin(h)),
                     px((0.31 + clr) * math.cos(h), (0.31 + clr) * math.sin(h)), c, 1)
        h = out.plan.heading
        cv2.arrowedLine(roi, (ox, oy), px(1.2 * math.cos(h), 1.2 * math.sin(h)),
                        STATE_BGR[out.state], 2)
    for p in out.obstacles:
        cv2.circle(roi, px(p.x, p.y), 1, (80, 80, 255), -1)
    if out.fix is not None:
        cv2.circle(roi, px(out.fix.range_m * math.cos(out.fix.bearing),
                           out.fix.range_m * math.sin(out.fix.bearing)), 5, (0, 230, 0), -1)
    cv2.rectangle(roi, (ox - 8, oy - 12), (ox + 8, oy + 12), (0, 200, 255), 1)


def vision_loop():
    model = YOLO(MODEL)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    last_n, fps_t, frames = -1, time.time(), 0
    events = []
    while not _stop.is_set():
        with _raw_lock:
            img, n = _raw["img"], _raw["n"]
        if img is None or n == last_n:
            time.sleep(0.005)
            continue
        last_n = n
        img = img.copy()
        res = model.track(img, conf=CONF, imgsz=IMGSZ, persist=True,
                          tracker="bytetrack.yaml", verbose=False)[0]
        dets = detections_from(res, img, names)
        pts, cones = range_inputs()
        now = time.time()
        with _brain_lock:
            out = brain.step(now, dets, pts, odom=current_odom(), free=cones)
        commander.set(out.l, out.r)
        if out.event:
            events = ([f"{time.strftime('%H:%M:%S')} {out.state.value}: {out.event}"] + events)[:8]

        draw(img, dets, out)
        frames += 1
        if now - fps_t >= 1.0:
            with _state_lock:
                S["fps"] = frames / (now - fps_t)
            frames, fps_t = 0, now
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with _out_lock:
                _out["jpg"] = enc.tobytes()
        tgt = next((d for d in dets if d.track_id == out.target_id and d.label == "person"), None)
        fix = out.fix
        with _state_lock:
            S.update({
                "state": out.state.value, "note": out.note, "event": out.event, "events": events,
                "target_id": out.target_id, "reid": out.reid_score,
                "range_m": None if fix is None else fix.range_m,
                "bearing_deg": None if fix is None else math.degrees(fix.bearing),
                "clipped": "" if fix is None else fix.clipped,
                "l": out.l, "r": out.r, "ndet": len(dets),
                "box_h": tgt.box[3] if (tgt is not None and fix is not None and not fix.clipped) else 0,
                "n_obstacles": len(out.obstacles),
            })


# web UI
PAGE = """<!doctype html><meta charset=utf-8>
<title>followme-bot</title>
<style>
 body{background:#111;color:#ddd;font:14px/1.45 -apple-system,system-ui,sans-serif;
      margin:0;padding:16px;display:flex;gap:16px;flex-wrap:wrap}
 img{width:720px;max-width:100%;border-radius:8px;background:#000}
 .panel{min-width:300px;flex:1}
 button{font:600 14px/1 inherit;padding:10px 14px;border-radius:8px;border:0;
        margin:0 6px 8px 0;cursor:pointer;background:#2a2a2a;color:#eee}
 button.arm{background:#1d6f2b;color:#fff} button.stop{background:#8c1c1c;color:#fff}
 table{border-collapse:collapse;width:100%;margin-top:8px}
 td{padding:3px 6px;border-bottom:1px solid #222} td:first-child{color:#888;width:40%}
 label{display:block;margin:10px 0 2px;color:#888}
 input[type=range]{width:100%} .big{font-size:20px;font-weight:700}
 #events{color:#aaa;font:12px ui-monospace,monospace;white-space:pre;margin-top:8px}
</style>
<img id=v>
<div class=panel>
  <div><button class=arm id=arm>ARM</button>
       <button class=stop id=stop>E-STOP</button>
       <button id=reset>Reset e-stop</button>
       <button id=relock>Re-lock target</button></div>
  <table>
    <tr><td>state</td><td class=big id=s_state>-</td></tr>
    <tr><td>note</td><td id=s_note>-</td></tr>
    <tr><td>target</td><td id=s_tgt>-</td></tr>
    <tr><td>range, bearing</td><td id=s_fix>-</td></tr>
    <tr><td>command L,R</td><td id=s_cmd>-</td></tr>
    <tr><td>detections, obstacle pts</td><td id=s_ndet>-</td></tr>
    <tr><td>vision fps</td><td id=s_fps>-</td></tr>
    <tr><td>pi link / odom / sonar</td><td id=s_link>-</td></tr>
  </table>
  <div id=events></div>
  <label>follow distance: <b id=l_td></b> m</label>
  <input type=range id=td min=1.5 max=5 step=0.1>
  <label>max speed: <b id=l_ms></b> %</label>
  <input type=range id=ms min=6 max=60 step=1>
  <label>stop short of obstacles: <b id=l_sd></b> m</label>
  <input type=range id=sd min=0.25 max=1.0 step=0.05>
  <label><input type=checkbox id=av> obstacle avoidance</label>
  <label>calibrate: stand this far from the camera, fully in frame, click</label>
  <input id=cd value="3.0" style="width:70px"> m <button id=cal>Calibrate</button>
  <div id=calout style="color:#888"></div>
</div>
<script>
const $=i=>document.getElementById(i);
let armed=false;
(function video(){
  const v=$('v'); let busy=false;
  async function tick(){
    if(!busy){ busy=true;
      try{ const r=await fetch('/frame.jpg?t='+Date.now(),{cache:'no-store'});
        if(r.ok){ const url=URL.createObjectURL(await r.blob()); const old=v.dataset.url;
          v.src=url; v.dataset.url=url; if(old) URL.revokeObjectURL(old); } }catch(e){}
      busy=false; }
    setTimeout(tick,45); }
  tick();
})();
const post=u=>fetch(u,{method:'POST'});
$('arm').onclick=()=>post(armed?'/disarm':'/arm');
$('stop').onclick=()=>post('/estop');
$('reset').onclick=()=>post('/reset');
$('relock').onclick=()=>post('/relock');
$('cal').onclick=()=>fetch('/calibrate?dist='+$('cd').value,{method:'POST'})
  .then(r=>r.json()).then(j=>{$('calout').textContent=j.msg;});
function slider(el,lab,key){ const e=$(el);
  e.oninput=()=>{$(lab).textContent=e.value; post('/set?'+key+'='+e.value);}; }
slider('td','l_td','follow.target_dist'); slider('ms','l_ms','follow.max_speed');
slider('sd','l_sd','avoid.stop_dist');
$('av').onchange=()=>post('/set?avoid.enabled='+($('av').checked?1:0));
const COL={IDLE:'#999',ACQUIRE:'#e6dc3c',FOLLOW:'#3cdc5a',AVOID:'#ffa028',BLOCKED:'#ff3c28',
  PURSUE:'#28c8dc',SEARCH:'#2896ff',WAIT:'#966edc',ESTOP:'#ff0028'};
function sync(el,lab,v){ if(document.activeElement!==$(el)) $(el).value=v; $(lab).textContent=v; }
setInterval(()=>fetch('/status').then(r=>r.json()).then(j=>{
  armed=j.armed; $('arm').textContent=armed?'DISARM':'ARM';
  $('s_state').textContent=j.state; $('s_state').style.color=COL[j.state]||'#ddd';
  $('s_note').textContent=j.note;
  $('s_tgt').textContent=j.target_id==null?'-':'#'+j.target_id+'  appearance '+j.reid.toFixed(2);
  $('s_fix').textContent=j.range_m==null?'-':j.range_m.toFixed(2)+' m, '+j.bearing_deg.toFixed(0)+' deg'+(j.clipped?'  ('+j.clipped+' clipped)':'');
  $('s_cmd').textContent=j.l.toFixed(1)+' , '+j.r.toFixed(1);
  $('s_ndet').textContent=j.ndet+' , '+j.n_obstacles;
  $('s_fps').textContent=j.fps.toFixed(1);
  $('s_link').textContent=(j.online?'ok':'DOWN')+' / '+(j.odom?'ok':'-')+' / '+(j.sonar?'ok':'-');
  $('s_link').style.color=j.online?'#5c5':'#f55';
  $('events').textContent=j.events.join('\\n');
  sync('td','l_td',j.p['follow.target_dist']); sync('ms','l_ms',j.p['follow.max_speed']);
  sync('sd','l_sd',j.p['avoid.stop_dist']); $('av').checked=!!j.p['avoid.enabled'];
}),200);
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        with _brain_lock:
            if u.path == "/arm":
                brain.arm()
            elif u.path == "/disarm":
                brain.disarm()
                commander.stop_now()
            elif u.path == "/estop":
                brain.estop()
                commander.stop_now()
            elif u.path == "/reset":
                brain.reset()
            elif u.path == "/relock":
                brain.relock()
            elif u.path == "/set":
                try:
                    for k, v in q.items():
                        cfg.set(k, v[0])
                except (KeyError, ValueError) as e:
                    return self._json({"ok": False, "error": str(e)}, 400)
                save_tune()
            elif u.path == "/calibrate":
                return self._calibrate(q)
            else:
                return self.send_error(404)
        self._json({"ok": True, "armed": brain.armed})

    def _calibrate(self, q):
        d = float(q.get("dist", ["3.0"])[0])
        with _state_lock:
            h = S["box_h"]
        if not h:
            return self._json({"msg": "need a locked, unclipped person"})
        cfg.camera.fx = cfg.camera.fy = h * d / cfg.follow.person_h
        save_tune()
        return self._json({"msg": f"focal = {cfg.camera.fx:.0f}px (from {h:.0f}px at {d}m)"})

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/status":
            with _state_lock:
                st = dict(S)
            st.update(armed=brain.armed, online=commander.online,
                      odom=current_odom() is not None, sonar=bool(_side["has_range"]),
                      p=cfg.flat())
            return self._json(st)
        if u.path == "/frame.jpg":
            with _out_lock:
                jpg = _out["jpg"]
            if not jpg:
                return self.send_error(503)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpg)
            return
        self.send_error(404)


def main():
    for fn in (mjpeg_reader, odom_poller, range_poller, vision_loop):
        threading.Thread(target=fn, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"follow server on http://127.0.0.1:{PORT}/  (starts DISARMED)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        brain.disarm()
        commander.stop_now()
        print("stopped, motors zeroed", flush=True)


if __name__ == "__main__":
    main()
