# Contributing

Issues and pull requests are welcome.

## Before you open a PR

```bash
pip install -e ".[dev]"
pytest                     # unit tests + every scenario, headless (~10 s)
python -m simkit.run       # the scenario table, if you changed the brain
```

A change to `followme/` should keep every scenario passing for the
`hardware` plant with both `sonar3` and `lidar`, and for the `ideal` plant.
If a scenario has to change, say why in the PR.

## Licensing of contributions

followme bot is dual-licensed (AGPL-3.0 and a commercial license, see
[COMMERCIAL.md](COMMERCIAL.md)). To keep that possible, contributions need a
contributor license agreement: by opening a pull request you agree that your
contribution is licensed to the project under the AGPL-3.0, and that the
maintainer may also distribute it under the commercial license. You keep the
copyright to your work.

If you cannot agree to that, open an issue describing the change instead.

## Style

- Plain Python, numpy, no framework in the core. `followme/` must not import
  ROS, OpenCV or anything hardware-specific.
- All tunables live in `followme/config.py`, with a comment saying why.
- Units in names or comments: metres, radians (+ = left), duty in percent.
