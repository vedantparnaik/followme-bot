# Commercial licensing

followme bot is dual-licensed.

**Open source: AGPL-3.0.** Free to use, study, modify and share, including
commercially, as long as you follow the AGPL. In short: if you distribute this
software, or modified versions of it, or let people interact with it over a
network, you must make the complete corresponding source (including your
changes) available under the AGPL too. The full terms are in [LICENSE](LICENSE).

**Commercial license.** If you want to ship followme bot, or anything built on
it, inside a closed-source product or service without the AGPL's obligations,
you can buy a commercial license. It covers the code in this repository
(`followme/`, `simkit/`, `robot/`, `ros2_ws/`, `tools/`) and can include
support and custom development.

Contact: vedantparnaik@gmail.com, subject "followme bot commercial license".

## Third-party components

The license above covers this repository's code only. Dependencies keep their
own licenses. The important one:

- **Ultralytics YOLO** (`ultralytics`, used by `robot/mac/follow_server.py`
  for person detection and tracking) is AGPL-3.0 as well. A commercial
  followme license does not cover it: closed-source use needs an
  [Ultralytics Enterprise License](https://www.ultralytics.com/license), or a
  different detector. The brain (`followme/`) and the simulator (`simkit/`) do
  not depend on it; the detector is only plumbing in the laptop app.

This file is a summary, not legal advice.
