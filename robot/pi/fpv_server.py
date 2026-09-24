#!/usr/bin/env python3
"""Combined FPV + teleop server for the 4-motor skid-steer chassis.

Serves ONE page with the live webcam feed as a full-screen background and the
driving controls overlaid. Open http://<pi-ip>:8000/ on your Mac.

- Camera: one ffmpeg reads native MJPEG from the BRIO into a shared buffer.
  Mounted front-centre (between FL and FR), ~1.5 ft above the wheels, so the
  view faces the direction of travel: no W/S inversion.
- Motors: background control loop applies latest command; STOPS if no command
  within TIMEOUT (deadman). Stuck keys self-heal after ~0.7s.

Drive mapping (verified on the bench, see docs/hardware.md):
  LEFT  side: FL RPWM=GPIO12 LPWM=GPIO13   RL RPWM=GPIO5  LPWM=GPIO6   normal
  RIGHT side: FR RPWM=GPIO18 LPWM=GPIO19   RR RPWM=GPIO16 LPWM=GPIO20  inverted
  Right-side motors are mounted mirrored, hence the invert on that side only.

Endpoints:  /  /stream  /snapshot  /audio  /cmd?l=&r=  /stats
            /odom   dead-reckoned pose from the applied duty (x, y, yaw)
            /range  ultrasonic readings (only with FOLLOWME_SONAR=1)
            POST /rec/start  POST /rec/stop  /rec/status  /recordings  /rec/file

/odom is integrated from the duty applied to each side (no encoders), so it
drifts, yaw especially since a skid-steer slips when it turns. The follow
brain only uses it for a few seconds of obstacle memory and for driving to
the last-seen point. Calibrate FOLLOWME_V_AT_100 (m/s at 100% duty,
straight) and FOLLOWME_TRACK_EFF (effective track width in m, larger than
the real one on a skid-steer).
"""

import json
import math
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import RPi.GPIO as GPIO

# config
DEVICE = "/dev/video0"
WIDTH, HEIGHT, FPS = 640, 480, 30   # low-res for snappy FPV latency
AUDIO_DEV = "plughw:3,0"            # BRIO built-in mic (ALSA card 3)
PORT = 8000
TIMEOUT = 0.25  # motor deadman (s)

# Breaking these gearmotors free from rest takes ~28% duty, but keeping them
# turning takes far less. So a side starting from zero gets a brief KICK_DUTY
# pulse, then drops to the steady range MIN_DUTY..100. Without the kick the
# floor would have to sit at the (much higher) breakaway value and the rover
# could never move slowly -- which is what made RL look dead at 20-30%.
MIN_DUTY = 18.0         # steady-state floor once rolling
TURN_MIN_DUTY = 26.0    # pivots scrub all four tyres sideways, so they need more
KICK_DUTY = 55.0        # stiction-breaking pulse
KICK_S = 0.18
# Only kick a side that has been at rest for KICK_REARM_S. A command
# oscillating around zero would otherwise re-kick every few frames and make
# the rover lurch side to side instead of holding still.
KICK_REARM_S = 0.5
SLEW = 400.0            # max duty change per second; 0->100 in 250ms

V_AT_100 = float(os.environ.get("FOLLOWME_V_AT_100", "1.35"))
TRACK_EFF = float(os.environ.get("FOLLOWME_TRACK_EFF", "0.46"))
SONAR = os.environ.get("FOLLOWME_SONAR") == "1"

# motors
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


class Motor:
    def __init__(self, name, rpwm, lpwm, invert=False, freq=1000):
        self.name = name
        self.invert = invert
        GPIO.setup(rpwm, GPIO.OUT)
        GPIO.setup(lpwm, GPIO.OUT)
        self.rpwm = GPIO.PWM(rpwm, freq)
        self.lpwm = GPIO.PWM(lpwm, freq)
        self.rpwm.start(0)
        self.lpwm.start(0)

    def drive(self, speed):
        if self.invert:
            speed = -speed
        speed = max(-100.0, min(100.0, speed))
        if speed >= 0:
            self.lpwm.ChangeDutyCycle(0)
            self.rpwm.ChangeDutyCycle(speed)
        else:
            self.rpwm.ChangeDutyCycle(0)
            self.lpwm.ChangeDutyCycle(-speed)


def shape(cmd, floor):
    """Remap a -100..100 request onto the range the motors can actually act on."""
    if abs(cmd) < 1.0:
        return 0.0
    mag = floor + (min(abs(cmd), 100.0) / 100.0) * (100.0 - floor)
    return mag if cmd > 0 else -mag


def approach(cur, goal, step):
    if goal > cur:
        return min(cur + step, goal)
    return max(cur - step, goal)


class Side:
    """One side of the skid-steer: its motors plus the kick/ramp state."""

    def __init__(self, motors):
        self.motors = motors
        self.applied = 0.0
        self.kick_until = 0.0
        self.zero_since = 0.0

    def update(self, target, dt, now):
        if target == 0.0:
            if self.applied != 0.0:
                self.zero_since = now
            self.applied = 0.0  # stops are immediate, never ramped
            self.kick_until = 0.0
        elif self.applied != 0.0 and target * self.applied < 0:
            # Direction reversal: coast through zero before kicking the other
            # way, rather than slamming the H-bridge across.
            self.applied = approach(self.applied, 0.0, SLEW * dt)
            if abs(self.applied) < 1.0:
                self.applied = 0.0
                self.zero_since = now
        elif self.applied == 0.0:
            if now - self.zero_since >= KICK_REARM_S:
                self.kick_until = now + KICK_S
                self.applied = KICK_DUTY if target > 0 else -KICK_DUTY
            else:
                # Only just stopped, so the wheels are still turning: ramp in
                # normally rather than hitting it with another kick.
                self.applied = approach(0.0, target, SLEW * dt)
        elif now < self.kick_until:
            self.applied = KICK_DUTY if target > 0 else -KICK_DUTY
        else:
            self.applied = approach(self.applied, target, SLEW * dt)

        for m in self.motors:
            m.drive(self.applied)


left_side = Side([Motor("FL", 12, 13, invert=False),
                  Motor("RL", 5, 6, invert=False)])
right_side = Side([Motor("FR", 18, 19, invert=True),
                   Motor("RR", 16, 20, invert=True)])

_target = {"l": 0.0, "r": 0.0, "ts": 0.0}
_applied = {"l": 0.0, "r": 0.0}
_mlock = threading.Lock()
_odom = {"x": 0.0, "y": 0.0, "yaw": 0.0, "t": 0.0}
_olock = threading.Lock()


def integrate_odom(al, ar, dt, now):
    vl, vr = al / 100.0 * V_AT_100, ar / 100.0 * V_AT_100
    v, w = 0.5 * (vl + vr), (vr - vl) / TRACK_EFF
    with _olock:
        _odom["yaw"] = math.atan2(math.sin(_odom["yaw"] + w * dt), math.cos(_odom["yaw"] + w * dt))
        _odom["x"] += v * math.cos(_odom["yaw"]) * dt
        _odom["y"] += v * math.sin(_odom["yaw"]) * dt
        _odom["t"] = now


def control_loop():
    last = time.time()
    while True:
        now = time.time()
        dt = max(now - last, 1e-3)
        last = now

        with _mlock:
            l, r, ts = _target["l"], _target["r"], _target["ts"]
        if now - ts > TIMEOUT:
            l = r = 0.0

        # A pivot (sides commanded opposite ways) has to overcome tyre scrub.
        floor = TURN_MIN_DUTY if (l * r) < 0 else MIN_DUTY
        left_side.update(shape(l, floor), dt, now)
        right_side.update(shape(r, floor), dt, now)
        _applied["l"], _applied["r"] = left_side.applied, right_side.applied
        integrate_odom(left_side.applied, right_side.applied, dt, now)
        time.sleep(0.02)


threading.Thread(target=control_loop, daemon=True).start()

ranger = None
if SONAR:
    from range_sensors import RangeSensors
    ranger = RangeSensors()

# camera
_cond = threading.Condition()
_latest = {"jpg": None, "n": 0}


def camera_thread():
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "v4l2", "-input_format", "mjpeg",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
        "-i", DEVICE, "-f", "mjpeg", "-c:v", "copy", "pipe:1",
    ]
    while True:
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        except Exception:
            time.sleep(2)
            continue
        buf = b""
        try:
            while True:
                chunk = p.stdout.read(8192)
                if not chunk:
                    break
                buf += chunk
                while True:
                    a = buf.find(b"\xff\xd8")
                    if a == -1:
                        if len(buf) > 4_000_000:
                            buf = b""
                        break
                    b = buf.find(b"\xff\xd9", a + 2)
                    if b == -1:
                        buf = buf[a:]
                        break
                    jpg = buf[a:b + 2]
                    buf = buf[b + 2:]
                    with _cond:
                        _latest["jpg"] = jpg
                        _latest["n"] += 1
                        _cond.notify_all()
        except Exception:
            pass
        finally:
            try:
                p.kill()
            except Exception:
                pass
        time.sleep(1)


threading.Thread(target=camera_thread, daemon=True).start()

# audio
# One ffmpeg per /audio request produces a low-latency Opus/WebM stream from the
# mic. Only one reader can hold the ALSA device, so a new request kills the old.
_aproc = None
_alock = threading.Lock()


def start_audio():
    global _aproc
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-fflags", "nobuffer",
        "-f", "alsa", "-ac", "1", "-ar", "48000", "-i", AUDIO_DEV,
        "-c:a", "libopus", "-b:a", "48k", "-application", "lowdelay",
        "-frame_duration", "20",
        "-f", "webm", "-flush_packets", "1", "-cluster_time_limit", "100",
        "pipe:1",
    ]
    with _alock:
        if _aproc and _aproc.poll() is None:
            _aproc.terminate()
            try:
                _aproc.wait(1)
            except Exception:
                _aproc.kill()
        _aproc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        return _aproc


# system stats helpers
_CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
REC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")


def _cpu_times():
    with open("/proc/stat") as f:
        vals = list(map(int, f.readline().split()[1:]))
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


def _mem_mb():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            d[k] = int(v.split()[0])
    total = d.get("MemTotal", 0) / 1024.0
    avail = d.get("MemAvailable", 0) / 1024.0
    return total - avail, total


def _temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except Exception:
        return 0.0


def _load1():
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except Exception:
        return 0.0


def _disk_bytes():
    r = w = 0
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                name = p[2]
                if name.startswith(("loop", "ram")):
                    continue
                if name.startswith("mmcblk") and "p" in name:  # skip partitions
                    continue
                if name[-1].isdigit() and not name.startswith("mmcblk"):
                    continue
                r += int(p[5])
                w += int(p[9])
    except Exception:
        pass
    return r * 512, w * 512


def _net_bytes(dev="wlan0"):
    try:
        base = f"/sys/class/net/{dev}/statistics/"
        with open(base + "rx_bytes") as f:
            rx = int(f.read())
        with open(base + "tx_bytes") as f:
            tx = int(f.read())
        return rx, tx
    except Exception:
        return 0, 0


def _proc_ticks(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            p = f.read().split()
        return int(p[13]) + int(p[14])
    except Exception:
        return 0


def _throttled():
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"],
                             capture_output=True, text=True, timeout=1).stdout
        return out.strip().split("=")[-1]
    except Exception:
        return ""


# recorder
class Recorder:
    """Records synchronized A/V (video from the shared frame buffer + mic audio),
    a motor-command log, and periodic Pi system stats into recordings/<ts>/."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.name = None
        self.dir = None
        self.t0 = 0.0
        self.proc = None
        self._stop = threading.Event()
        self._sthread = None
        self.motor_fp = None
        self.motor_lock = threading.Lock()

    def status(self):
        with self.lock:
            if not self.active:
                return {"recording": False}
            size = 0
            try:
                for f in os.listdir(self.dir):
                    size += os.path.getsize(os.path.join(self.dir, f))
            except Exception:
                pass
            return {"recording": True, "name": self.name,
                    "secs": round(time.time() - self.t0, 1),
                    "size_mb": round(size / 1e6, 2)}

    def start(self):
        global _aproc
        with self.lock:
            if self.active:
                return self.name
            # recorder owns the mic: stop the live audio monitor first
            with _alock:
                if _aproc and _aproc.poll() is None:
                    _aproc.terminate()
                    try:
                        _aproc.wait(1)
                    except Exception:
                        _aproc.kill()
                _aproc = None
            time.sleep(0.15)  # let ALSA release the device
            self.name = time.strftime("%Y%m%d_%H%M%S")
            self.dir = os.path.join(REC_DIR, self.name)
            os.makedirs(self.dir, exist_ok=True)
            self.t0 = time.time()
            self._stop.clear()
            av = os.path.join(self.dir, "av.mkv")
            # Video: read the already-running MJPEG stream over localhost (full
            # frame rate, no device conflict). Audio: mic. Wallclock timestamps
            # keep A/V in sync. Stopped gracefully with SIGINT.
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
                "-use_wallclock_as_timestamps", "1",
                "-f", "mpjpeg", "-i", f"http://127.0.0.1:{PORT}/stream",
                "-f", "alsa", "-ac", "1", "-ar", "48000", "-i", AUDIO_DEV,
                "-c:v", "copy", "-c:a", "libopus", "-b:a", "48k",
                "-f", "matroska", av,
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
            self.motor_fp = open(os.path.join(self.dir, "motor.csv"), "w", buffering=1)
            self.motor_fp.write("t_s,l,r\n")
            with _mlock:
                l, r = _target["l"], _target["r"]
            self._write_motor(l, r)
            with open(os.path.join(self.dir, "meta.json"), "w") as f:
                json.dump({"name": self.name, "start": self.t0,
                           "video": f"{WIDTH}x{HEIGHT}", "fps": FPS,
                           "audio": "opus 48k mono"}, f, indent=2)
            self.active = True
            self._sthread = threading.Thread(target=self._stats_loop, daemon=True)
            self._sthread.start()
            return self.name

    def _write_motor(self, l, r):
        if self.motor_fp:
            with self.motor_lock:
                try:
                    self.motor_fp.write(f"{time.time() - self.t0:.3f},{l:.1f},{r:.1f}\n")
                except Exception:
                    pass

    def log_motor(self, l, r):
        if self.active:
            self._write_motor(l, r)

    def _stats_loop(self):
        pid = self.proc.pid if self.proc else 0
        f = open(os.path.join(self.dir, "stats.csv"), "w", buffering=1)
        f.write("t_s,cpu_pct,mem_used_mb,mem_total_mb,temp_c,load1,"
                "disk_read_kBps,disk_write_kBps,net_rx_kBps,net_tx_kBps,"
                "rec_cpu_pct,throttled\n")
        pt = _cpu_times()
        pd = _disk_bytes()
        pn = _net_bytes()
        pp = _proc_ticks(pid)
        tprev = time.time()
        while not self._stop.wait(1.0):
            now = time.time()
            dt = max(1e-3, now - tprev)
            tprev = now
            ct = _cpu_times()
            dtot = ct[0] - pt[0]
            didle = ct[1] - pt[1]
            cpu = 100.0 * (dtot - didle) / dtot if dtot > 0 else 0.0
            pt = ct
            used, total = _mem_mb()
            cd = _disk_bytes()
            dr = (cd[0] - pd[0]) / 1024.0 / dt
            dw = (cd[1] - pd[1]) / 1024.0 / dt
            pd = cd
            cn = _net_bytes()
            nrx = (cn[0] - pn[0]) / 1024.0 / dt
            ntx = (cn[1] - pn[1]) / 1024.0 / dt
            pn = cn
            cp = _proc_ticks(pid)
            rec_cpu = 100.0 * ((cp - pp) / _CLK) / dt
            pp = cp
            f.write(f"{now - self.t0:.1f},{cpu:.1f},{used:.0f},{total:.0f},"
                    f"{_temp_c():.1f},{_load1():.2f},{dr:.1f},{dw:.1f},"
                    f"{nrx:.1f},{ntx:.1f},{rec_cpu:.1f},{_throttled()}\n")
        f.close()

    def stop(self):
        with self.lock:
            if not self.active:
                return None
            self.active = False
            self._stop.set()
            name = self.name
            t_stop = time.time()
        try:
            self.proc.send_signal(signal.SIGINT)  # graceful: writes MKV trailer
            self.proc.wait(6)
        except Exception:
            try:
                self.proc.terminate()
                self.proc.wait(3)
            except Exception:
                pass
        if self._sthread:
            self._sthread.join(3)
        if self.motor_fp:
            try:
                self.motor_fp.close()
            except Exception:
                pass
            self.motor_fp = None
        try:
            meta = os.path.join(self.dir, "meta.json")
            with open(meta) as fh:
                m = json.load(fh)
            m["stop"] = t_stop
            m["duration_s"] = round(t_stop - m["start"], 1)
            with open(meta, "w") as fh:
                json.dump(m, fh, indent=2)
        except Exception:
            pass
        return name


os.makedirs(REC_DIR, exist_ok=True)
recorder = Recorder()

# live system stats (for the /stats endpoint)
_sysstats = {}


def sysstats_loop():
    pt = _cpu_times()
    pn = _net_bytes()
    pd = _disk_bytes()
    tprev = time.time()
    while True:
        time.sleep(1.0)
        now = time.time()
        dt = max(1e-3, now - tprev)
        tprev = now
        ct = _cpu_times()
        dtot = ct[0] - pt[0]
        didle = ct[1] - pt[1]
        cpu = 100.0 * (dtot - didle) / dtot if dtot > 0 else 0.0
        pt = ct
        used, total = _mem_mb()
        cn = _net_bytes()
        nrx = (cn[0] - pn[0]) / 1024.0 / dt
        ntx = (cn[1] - pn[1]) / 1024.0 / dt
        pn = cn
        cd = _disk_bytes()
        dr = (cd[0] - pd[0]) / 1024.0 / dt
        dw = (cd[1] - pd[1]) / 1024.0 / dt
        pd = cd
        _sysstats.update({
            "cpu_pct": round(cpu, 1),
            "mem_used_mb": round(used), "mem_total_mb": round(total),
            "mem_pct": round(100.0 * used / total, 1) if total else 0,
            "temp_c": round(_temp_c(), 1), "load1": round(_load1(), 2),
            "net_rx_kBps": round(nrx, 1), "net_tx_kBps": round(ntx, 1),
            "disk_read_kBps": round(dr, 1), "disk_write_kBps": round(dw, 1),
            "throttled": _throttled(), "recording": recorder.active,
        })


threading.Thread(target=sysstats_loop, daemon=True).start()

# page
PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>FPV Rover</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#000;color:#eee;font-family:system-ui,sans-serif;
  user-select:none;-webkit-user-select:none;overflow:hidden;touch-action:none}
#cam{position:fixed;inset:0;width:100%;height:100%;object-fit:contain;background:#000;z-index:0}
#hud{position:fixed;top:8px;left:8px;z-index:3;font-family:monospace;font-size:13px;
  background:rgba(0,0,0,.45);padding:6px 10px;border-radius:8px;line-height:1.5}
#hud b{color:#9cf}
#dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#e01b24;margin-right:5px;vertical-align:baseline}
#dot.ok{background:#26a269}
#pad{position:fixed;left:0;right:0;bottom:14px;z-index:3;text-align:center}
.row{margin:5px}
.k{display:inline-block;width:64px;height:64px;line-height:64px;margin:4px;border:1px solid rgba(255,255,255,.25);
  border-radius:12px;background:rgba(20,22,30,.6);font-size:20px;cursor:pointer;backdrop-filter:blur(3px)}
.k.on{background:rgba(38,162,105,.85);border-color:#26a269;color:#fff}
.k.stop{background:rgba(90,20,20,.7);border-color:#a33}
.k.stop.on{background:#e01b24}
#spwrap{position:fixed;top:8px;right:8px;z-index:3;background:rgba(0,0,0,.45);padding:6px 10px;border-radius:8px;font-size:13px}
input[type=range]{width:130px;vertical-align:middle}
.btn{margin-left:8px;padding:3px 8px;border-radius:6px;border:1px solid rgba(255,255,255,.3);
  background:rgba(20,22,30,.7);color:#eee;font-size:12px;cursor:pointer}
#aud.on{background:rgba(38,162,105,.85);border-color:#26a269;color:#fff}
#rec.on{background:#e01b24;border-color:#ff5b5b;color:#fff;animation:blink 1s step-start infinite}
.btn:disabled{opacity:.45;cursor:default}
@keyframes blink{50%{opacity:.45}}
#clipbox{display:none;position:fixed;top:44px;right:8px;z-index:4;max-width:360px;max-height:60vh;
  overflow:auto;background:rgba(0,0,0,.8);padding:8px 10px;border-radius:8px;font-size:12px;font-family:monospace}
#clipbox a{color:#9cf;text-decoration:none}
.clip{margin:4px 0;padding-bottom:4px;border-bottom:1px solid rgba(255,255,255,.15);line-height:1.7}
</style></head>
<body>
<img id=cam src="/stream" alt="camera">
<audio id=snd></audio>
<div id=hud><span id=dot></span><span id=connt>connecting</span><br>SPD <b id=sv>0</b>% &nbsp; l <b id=lv>0</b> r <b id=rv>0</b><br>&#127918; <b id=gp>none</b></div>
<div id=spwrap>speed <input type=range id=spd min=10 max=100 value=40> <b id=spdv>40</b>%
  <button id=aud class=btn>&#128264; audio off</button>
  <button id=rec class=btn>&#9679; REC</button><span id=recinfo></span>
  <button id=clips class=btn>clips</button></div>
<div id=clipbox></div>
<div id=pad>
  <div class=row><span class=k id=kw data-k=w>W</span></div>
  <div class=row><span class=k id=ka data-k=a>A</span><span class=k id=ks data-k=s>S</span><span class=k id=kd data-k=d>D</span></div>
  <div class=row><span class=k id=kq data-k=q>Q</span><span class="k stop" id=kstop>STOP</span><span class=k id=ke data-k=e>E</span></div>
</div>
<script>
const DRIVE=['w','a','s','d','q','e'];
const keys={}, kbdSeen={};
let speed=40, ok=true, lastSent="";
const spd=document.getElementById('spd'), spdv=document.getElementById('spdv');
spd.oninput=()=>{speed=+spd.value; spdv.textContent=speed;};
function compute(){
  let fwd=(keys['w']?1:0)-(keys['s']?1:0);
  let turn=(keys['d']?1:0)-(keys['a']?1:0)+(keys['e']?1:0)-(keys['q']?1:0);
  let l=fwd+turn, r=fwd-turn;
  const m=Math.max(1,Math.abs(l),Math.abs(r));
  return [l/m*speed, r/m*speed];
}
// DualShock 4 / standard gamepad: LEFT STICK does everything (single-stick arcade).
// Y (axes[1]) = throttle: stick up reads -1, so it is negated to match W.
// X (axes[0]) = steering: right (=+1) matches D.
// Overrides the keyboard only while the stick is pushed past the deadzone.
const GP_DEAD=0.12;
function activePad(){
  const pads=navigator.getGamepads?navigator.getGamepads():[];
  for(const p of pads){ if(p) return p; }
  return null;
}
function computeGamepad(){
  const p=activePad(), gpEl=document.getElementById('gp');
  if(!p){ if(gpEl)gpEl.textContent='none'; return null; }
  if(gpEl)gpEl.textContent='ready';
  const dz=v=>Math.abs(v)<GP_DEAD?0:v;
  let fwd=-dz(p.axes[1]||0), turn=dz(p.axes[0]||0);
  if(!fwd&&!turn) return null;
  let l=fwd+turn, r=fwd-turn;
  const m=Math.max(1,Math.abs(l),Math.abs(r));
  return [l/m*speed, r/m*speed];
}
function driveVals(){ return computeGamepad() || compute(); }
function paint(){
  for(const [k,id] of [['w','kw'],['a','ka'],['s','ks'],['d','kd'],['q','kq'],['e','ke']])
    document.getElementById(id).classList.toggle('on',!!keys[k]);
}
function setConn(o){ok=o;const d=document.getElementById('dot');d.classList.toggle('ok',o);
  document.getElementById('connt').textContent=o?'connected':'DISCONNECTED';}
function send(l,r){
  fetch('/cmd?l='+l.toFixed(1)+'&r='+r.toFixed(1)).then(()=>{if(!ok)setConn(true);}).catch(()=>setConn(false));
}
function refresh(force){
  const [l,r]=driveVals();
  document.getElementById('sv').textContent=Math.round(Math.max(Math.abs(l),Math.abs(r)));
  document.getElementById('lv').textContent=l.toFixed(0);
  document.getElementById('rv').textContent=r.toFixed(0);
  paint();
  const msg=l.toFixed(1)+','+r.toFixed(1), moving=(l||r);
  if(force||moving||msg!==lastSent){ lastSent=msg; send(l,r); }
}
function clearAll(){for(const k in keys)delete keys[k];for(const k in kbdSeen)delete kbdSeen[k];refresh(true);}
document.addEventListener('keydown',e=>{const k=e.key.toLowerCase();
  if(k===' '){clearAll();e.preventDefault();return;}
  if(DRIVE.includes(k)){kbdSeen[k]=performance.now();if(!keys[k]){keys[k]=1;refresh();}e.preventDefault();}});
document.addEventListener('keyup',e=>{const k=e.key.toLowerCase();
  if(DRIVE.includes(k)){delete keys[k];delete kbdSeen[k];e.preventDefault();refresh();}});
setInterval(()=>{const now=performance.now();
  for(const k in kbdSeen){if(now-kbdSeen[k]>700){delete keys[k];delete kbdSeen[k];}}
  refresh();},90);
function bindBtn(id,k){const el=document.getElementById(id);
  const d=e=>{e.preventDefault();if(!keys[k]){keys[k]=1;refresh();}};
  const u=e=>{e.preventDefault();delete keys[k];refresh();};
  el.addEventListener('pointerdown',d);el.addEventListener('pointerup',u);
  el.addEventListener('pointerleave',u);el.addEventListener('pointercancel',u);}
['kw','ka','ks','kd','kq','ke'].forEach(id=>bindBtn(id,document.getElementById(id).dataset.k));
document.getElementById('kstop').addEventListener('pointerdown',e=>{e.preventDefault();clearAll();});
window.addEventListener('blur',clearAll);

// battery-friendly idle: when stopped, ping only 1/s (keeps connection status
// fresh); the server deadman keeps motors stopped. Active driving still streams
// commands at the fast 90ms rate above.
setInterval(()=>{ const [l,r]=driveVals(); if(!l&&!r) send(0,0); },1000);
window.addEventListener('gamepadconnected',()=>{document.getElementById('gp').textContent='ready';});
window.addEventListener('gamepaddisconnected',()=>{document.getElementById('gp').textContent='none';});

// audio toggle (needs a user gesture to start playback)
let audioOn=false;
const snd=document.getElementById('snd'), audBtn=document.getElementById('aud');
function startAudio(){ snd.src='/audio?ts='+Date.now(); snd.play().catch(()=>{}); }
function stopAudio(){ snd.pause(); snd.removeAttribute('src'); snd.load(); }
audBtn.addEventListener('click',()=>{
  audioOn=!audioOn;
  if(audioOn){ startAudio(); audBtn.classList.add('on'); audBtn.innerHTML='&#128266; audio on'; }
  else { stopAudio(); audBtn.classList.remove('on'); audBtn.innerHTML='&#128264; audio off'; }
});

// pause the video/audio when the tab is hidden (saves CPU/bandwidth/power), resume on return
const cam=document.getElementById('cam');
document.addEventListener('visibilitychange',()=>{
  if(document.hidden){ clearAll(); cam.removeAttribute('src'); if(audioOn)stopAudio(); }
  else { cam.src='/stream?ts='+Date.now(); if(audioOn)startAudio(); }
});

// recording: captures A/V + motor log + Pi system stats on the Pi (recorder owns the mic)
let recording=false;
const recBtn=document.getElementById('rec'), recInfo=document.getElementById('recinfo');
const clipsBtn=document.getElementById('clips'), clipbox=document.getElementById('clipbox');
function fmtTime(s){const m=Math.floor(s/60),ss=Math.floor(s%60);
  return String(m).padStart(2,'0')+':'+String(ss).padStart(2,'0');}
recBtn.addEventListener('click',async()=>{
  recBtn.disabled=true;
  try{
    if(!recording){
      if(audioOn){audioOn=false;stopAudio();audBtn.classList.remove('on');audBtn.innerHTML='&#128264; audio off';}
      audBtn.disabled=true;
      await fetch('/rec/start',{method:'POST'});
      recording=true; recBtn.classList.add('on'); recInfo.textContent=' 00:00';
    }else{
      await fetch('/rec/stop',{method:'POST'});
      recording=false; recBtn.classList.remove('on'); recInfo.textContent=''; audBtn.disabled=false;
      loadClips();
    }
  }catch(e){}
  recBtn.disabled=false;
});
setInterval(async()=>{
  if(!recording)return;
  try{const s=await(await fetch('/rec/status')).json();
    if(s.recording)recInfo.textContent=' '+fmtTime(s.secs)+' \u00b7 '+s.size_mb+'MB';
  }catch(e){}
},1000);
async function loadClips(){
  try{const list=await(await fetch('/recordings')).json();
    clipbox.innerHTML = list.length ? list.map(s=>{
      const links=s.files.map(f=>'<a href="/rec/file?s='+encodeURIComponent(s.name)+'&f='+encodeURIComponent(f.name)+'" target=_blank>'+f.name+' ('+f.mb+'MB)</a>').join(' \u00b7 ');
      return '<div class=clip><b>'+s.name+'</b><br>'+links+'</div>';
    }).join('') : '<i>no recordings yet</i>';
  }catch(e){ clipbox.innerHTML='<i>error loading</i>'; }
}
clipsBtn.addEventListener('click',()=>{
  const show=clipbox.style.display!=='block';
  clipbox.style.display=show?'block':'none';
  if(show)loadClips();
});
</script></body></html>"""


_CT = {".mkv": "video/x-matroska", ".csv": "text/csv",
       ".json": "application/json", ".webm": "audio/webm"}


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

    def _list_recordings(self):
        out = []
        try:
            for name in sorted(os.listdir(REC_DIR), reverse=True):
                d = os.path.join(REC_DIR, name)
                if not os.path.isdir(d):
                    continue
                files = []
                for fn in sorted(os.listdir(d)):
                    fp = os.path.join(d, fn)
                    files.append({"name": fn,
                                  "mb": round(os.path.getsize(fp) / 1e6, 2)})
                out.append({"name": name, "files": files})
        except Exception:
            pass
        return out

    def _serve_recording_file(self, u):
        q = parse_qs(u.query)
        s = os.path.basename(q.get("s", [""])[0])
        f = os.path.basename(q.get("f", [""])[0])
        path = os.path.join(REC_DIR, s, f)
        if not s or not f or not os.path.isfile(path) \
                or not os.path.abspath(path).startswith(os.path.abspath(REC_DIR)):
            self.send_response(404)
            self.end_headers()
            return
        ct = _CT.get(os.path.splitext(f)[1], "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.send_header("Content-Disposition", f'attachment; filename="{s}_{f}"')
        self.end_headers()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/rec/start":
            name = recorder.start()
            self._json({"recording": True, "name": name})
            return
        if u.path == "/rec/stop":
            name = recorder.stop()
            self._json({"recording": False, "name": name})
            return
        self.send_response(404)
        self.end_headers()

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

        if u.path == "/cmd":
            q = parse_qs(u.query)
            try:
                l = float(q.get("l", ["0"])[0])
                r = float(q.get("r", ["0"])[0])
            except ValueError:
                l = r = 0.0
            with _mlock:
                _target["l"], _target["r"], _target["ts"] = l, r, time.time()
            recorder.log_motor(l, r)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if u.path == "/snapshot":
            with _cond:
                jpg = _latest["jpg"]
            if jpg is None:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            return

        if u.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = 0
            try:
                while True:
                    with _cond:
                        while _latest["n"] == last:
                            _cond.wait(timeout=5)
                        jpg = _latest["jpg"]
                        last = _latest["n"]
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if u.path == "/stats":
            self._json(dict(_sysstats))
            return

        if u.path == "/odom":
            with _olock:
                od = dict(_odom)
            od["applied_l"], od["applied_r"] = _applied["l"], _applied["r"]
            self._json(od)
            return

        if u.path == "/range":
            if ranger is None:
                self._json({"error": "no range sensors (set FOLLOWME_SONAR=1)"}, 404)
                return
            self._json(ranger.read())
            return

        if u.path == "/rec/status":
            self._json(recorder.status())
            return

        if u.path == "/recordings":
            self._json(self._list_recordings())
            return

        if u.path == "/rec/file":
            self._serve_recording_file(u)
            return

        if u.path == "/audio":
            if recorder.active:   # mic is owned by the recorder
                self.send_response(409)
                self.end_headers()
                return
            p = start_audio()
            self.send_response(200)
            self.send_header("Content-Type", "audio/webm")
            self.send_header("Cache-Control", "no-cache, private")
            self.end_headers()
            try:
                while True:
                    chunk = p.stdout.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                try:
                    p.terminate()
                except Exception:
                    pass
            return

        self.send_response(404)
        self.end_headers()


if __name__ == "__main__":
    print(f"FPV server on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
