# followme bot

A robot that follows one person: it keeps ~2.6 m behind them, goes around
things in the way, and does not get confused when someone else walks in
between.

This repo is the brain, a simulator that tries hard to break it, and the code
that runs it on a real 4-wheel rover (Raspberry Pi + a laptop running YOLO).

| Stranger in the line of sight | Around furniture | Target behind a pillar |
|:---:|:---:|:---:|
| ![crossing](docs/media/crossing.gif) | ![obstacles](docs/media/obstacles.gif) | ![pillar](docs/media/pillar.gif) |
| The tracker hands the target's ID to a look-alike. The brain notices (the box teleported, the colours are off), drops it, and re-identifies the right person when they reappear. | The direct line cuts through a pallet and a chair. The pallet has no camera label, so only the ultrasonics see it. | The person vanishes mid-frame. The rover drives around the pillar to where they were last seen, then picks them up again by appearance. |

## What it does

- **Locks one person** and follows at a set distance, stopping when they stop.
- **Knows who it is following.** Appearance gallery (torso and legs colour
  histograms), ID-swap detection, and re-identification after occlusion.
  Never switches to a stranger on its own.
- **Avoids obstacles** from ultrasonics (or a lidar) and from anything the
  camera detector knows (chairs, bags, other people), with a few seconds of
  memory so corners do not vanish when they leave the sensor cone.
- **Handles losing them.** Drives to where they were last seen if they
  disappeared behind something, turns toward where they went if they left
  the frame, then waits. Bounded, always.
- **Two drive profiles, same brain.** `hardware` for a real skid-steer that
  cannot turn gently (stiction kick, duty floors, 250 ms latency); `ideal`
  for perfect motors.

## Quick start (simulator, no robot needed)

```bash
git clone https://github.com/vedantparnaik/followme-bot.git && cd followme-bot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m simkit.run                          # every scenario, headless, ~2 s
python -m simkit.run -s crossing -v           # one scenario with its event log
python -m simkit.run --sensor lidar --seeds 4
python -m simkit.run -s pillar --gif pillar.gif
pytest                                        # unit tests + all scenarios
```

```
scenario   seed  pass  collisions  person_contacts  follow_s  wrong_s  reid  range_err_m  final
loop          0  True           0                0      45.6      0.0     0         0.41  FOLLOW@2.9m
obstacles     0  True           0                0      33.6      0.0     0         0.84  FOLLOW@2.5m
crossing      0  True           0                0      26.3      0.0     1         0.71  FOLLOW@2.6m
pillar        0  True           0                0      24.0      0.0     1         0.65  FOLLOW@2.8m
```

A run passes with no collisions, no contact with any person, under 0.5 s
spent following the wrong person, and ending locked on the right one.
Current results: hardware plant with sonar 32/32 (8 seeds), with lidar 16/16,
ideal plant 16/16.

### With ROS 2 and RViz

```bash
tools/run_ros_sim.sh                                  # obstacles, hardware, sonar
tools/run_ros_sim.sh scenario:=crossing sensor:=lidar
ros2 topic pub --once /follow/cmd std_msgs/String '{data: estop}'
ros2 topic echo /follow/state
```

Tested with ROS 2 Humble (RoboStack on macOS). The `world` node stands in for
the robot (detections, range, odometry); the `brain` node is the same
`followme.Brain`.

## On a real robot

```
 Raspberry Pi (rover)                      laptop
 ─────────────────────                     ──────────────────────────────
 camera ── MJPEG /stream ───────────────▶  YOLO + ByteTrack, colour features
 HC-SR04 x3 ── /range (optional) ───────▶  followme.Brain
 duty dead reckoning ── /odom ──────────▶      │
 motors ◀── /cmd?l=&r= (0.25 s deadman) ◀──────┘   UI: http://127.0.0.1:8080/
```

```bash
# once: ssh-copy-id pi@raspberrypi.local
PI_HOST=raspberrypi.local PI_USER=pi tools/deploy_pi.sh      # PI_SONAR=1 for ultrasonics
pip install -e ".[robot]"
FOLLOWME_PI=raspberrypi.local:8000 python robot/mac/follow_server.py
```

Open http://127.0.0.1:8080/, stand in front of the camera, press ARM. It
starts disarmed, E-STOP latches, and the Pi stops the motors if the laptop
or the Wi-Fi goes away. Start with the default 15 % speed cap.

No robot yet? `python tools/fake_pi.py --video some_walk.mov` serves a video
as the camera so you can run the laptop side against recorded footage.

Wiring, pins and the drive layer: [docs/hardware.md](docs/hardware.md).

## Layout

```
followme/            the brain: perception, re-ID, tracker, obstacles, planner, state machine
simkit/              2-D world, virtual camera + range sensors, drivetrain models, scenarios, runner
tests/               unit tests and every scenario as a test
ros2_ws/src/followme_ros/   world + brain nodes, launch, RViz config
robot/pi/            Pi server: MJPEG, motors (kick, floors, deadman), /odom, HC-SR04 driver
robot/mac/           laptop app: YOLO + ByteTrack, colour features, web UI
tools/               deploy, fake Pi, ROS launcher
docs/                architecture, hardware
```

How it works: [docs/architecture.md](docs/architecture.md).

## Limitations

Stated plainly, because a follow-me robot that "just works" in a demo is easy
and one that works in a car park is not:

- **No map, no SLAM.** A few seconds of obstacle memory. It follows a person
  along a path a person can walk; it does not plan through mazes.
- **Camera-only obstacle sensing is limited to what the detector knows.**
  In camera-only mode the scenarios with a pallet, a bollard and a pillar
  collide, by design: those have no camera label. Use the ultrasonics.
- **Colour re-ID** tells a red jacket from a grey one, not two identical
  uniforms. The gallery takes any feature vector, so a learned re-ID model
  drops in.
- **Odometry is dead reckoning from motor duty**, no encoders. Good for a few
  seconds of memory, not for navigation.
- **The sim is 2-D and kind.** Flat ground, cylinder people, clean geometry.
  The drivetrain model is the measured one, which is where most of the real
  difficulty is, but passing the sim is necessary, not sufficient.

## License

AGPL-3.0 ([LICENSE](LICENSE)), with a commercial license available for
closed-source use: see [COMMERCIAL.md](COMMERCIAL.md). The laptop app uses
Ultralytics YOLO, which is separately AGPL-3.0 licensed.
