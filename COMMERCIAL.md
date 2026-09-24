# Commercial licensing

followme-bot is dual-licensed.

## Open source (AGPL-3.0)

Free to use, study, modify and share, including commercially, as long as you
follow the AGPL. In short: if you distribute this software or a modified
version of it, or let people interact with it over a network, you have to
make the complete source (including your changes) available under the AGPL
too. The full terms are in [LICENSE](LICENSE).

## Commercial license

If you want to ship followme-bot, or something built on it, in a
closed-source product or service without the AGPL's obligations, you can buy
a commercial license. It covers the code in this repository (`followme/`,
`simkit/`, `robot/`, `ros2_ws/`, `tools/`) and can include support and custom
development.

Contact: vedantparnaik@gmail.com, subject "followme-bot commercial license".

## Third-party components

The licenses above only cover this repository's code. Dependencies keep
their own licenses, and one of them matters here: Ultralytics YOLO
(`ultralytics`, used by `robot/mac/follow_server.py` for detection and
tracking) is also AGPL-3.0. A commercial followme-bot license doesn't cover
it, so closed-source use needs an
[Ultralytics Enterprise License](https://www.ultralytics.com/license) or a
different detector. The brain (`followme/`) and the simulator (`simkit/`)
don't depend on it; the detector only matters in the laptop app.

This file is a summary, not legal advice.
