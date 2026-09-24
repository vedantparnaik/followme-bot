# Hardware

The reference rover. Nothing in `followme/` depends on it: any drivetrain
that takes left/right duty and any camera that gives 640x480 frames will do,
after adjusting `followme/config.py`.

## Parts

| Item | Notes |
|------|-------|
| Chassis | 4-wheel skid-steer, ~0.62 x 0.40 m |
| Computer | Raspberry Pi 4B on the rover (motors, camera stream, sensors) |
| Autonomy | Any laptop on the same Wi-Fi (YOLO + the brain) |
| Motor drivers | 4x BTS7960 (IBT-2), one per motor |
| Motors | 4x DC gear motors |
| Motor power | 3S LiPo (~11.1 V) to all four drivers' B+/B- in parallel |
| Pi power | USB power bank to the Pi's USB-C |
| Camera | USB webcam (Logitech BRIO), front centre, ~0.45 m above the ground |
| Range (optional) | 3x HC-SR04 ultrasonic, front-left / front / front-right |

Do not feed the LiPo into the Pi. Do not put driver VCC on 5 V.

## Motors

```
FL ----- FR
|        |
|        |
RL ----- RR
```

| Corner | RPWM (BCM / pin) | LPWM (BCM / pin) | Inverted |
|--------|------------------|------------------|----------|
| FR | GPIO 18 / 12 | GPIO 19 / 35 | yes |
| FL | GPIO 12 / 32 | GPIO 13 / 33 | no |
| RL | GPIO 5 / 29 | GPIO 6 / 31 | no |
| RR | GPIO 16 / 36 | GPIO 20 / 38 | yes |

Right-side motors are mounted mirrored, so they are inverted in software.
Every driver: VCC, R_EN, L_EN to the Pi's 3.3 V; GND common with the LiPo -.

When one corner only turns one way, check the pin number before blaming the
motor: an RPWM landed on physical pin 27 (GPIO 0, the ID EEPROM pin) instead
of 29 looks exactly like a dead motor that still reverses.

## Ultrasonic sensors (optional)

| Sensor | Angle | TRIG (BCM) | ECHO (BCM) |
|--------|-------|------------|------------|
| front-left | +30 deg | 23 | 24 |
| front | 0 deg | 17 | 27 |
| front-right | -30 deg | 22 | 25 |

HC-SR04 ECHO is 5 V and the Pi's GPIO is not 5 V tolerant. Put a divider on
every ECHO line: 1 kOhm from ECHO to the GPIO pin, 2 kOhm from the GPIO pin to
ground. VCC to 5 V, GND common. Enable with `PI_SONAR=1 tools/deploy_pi.sh`.

Mount them at bumper height (~0.15-0.25 m). They cover what the camera
cannot classify: boxes, pallets, bollards, glass. They do not cover the sides
or the back.

## Drive layer (`robot/pi/fpv_server.py`)

Between a duty command and the motors:

| Constant | Value | Why |
|----------|-------|-----|
| `MIN_DUTY` | 18 % | Rolling floor. Any nonzero request is remapped onto 18-100 %. |
| `TURN_MIN_DUTY` | 26 % | A skid-steer pivot scrubs all four tyres sideways. |
| `KICK_DUTY`, `KICK_S` | 55 %, 0.18 s | Breaks stiction when a side starts from rest. Without it the floor would have to sit at the breakaway duty and the rover could never crawl. |
| `SLEW` | 400 %/s | Four motors stepping to full duty at once sag the LiPo enough to brown out the driver logic. Stops are immediate. |
| deadman | 0.25 s | No command for 0.25 s: motors stop. |

These are why the `hardware` profile exists. The smallest possible turn from
rest is a 55 % kick, so the brain steers by curving while rolling and only
pivots in short nudge-and-settle pulses. `simkit.plants.HardwarePlant`
reproduces this layer (plus ~0.25 s camera-to-motor latency), which is what
the scenarios are tuned against.

## Camera

640x480 MJPEG at 30 fps. Pinhole defaults: `fx = fy = 461 px`, which is what a
1.7 m person 3.0 m away (261 px tall) gives. Recalibrate from the laptop UI:
stand a measured distance from the camera, fully in frame, and click
Calibrate.

The BRIO needs a USB-C to USB-C cable plus a passive C-to-A adapter. A
C-to-A cable without the CC pull-up leaves it unpowered and invisible to
`lsusb`.
