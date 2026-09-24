#!/usr/bin/env bash
# World + brain + RViz on ROS 2 (tested with RoboStack Humble in micromamba).
# Usage:  tools/run_ros_sim.sh
#         tools/run_ros_sim.sh scenario:=crossing sensor:=lidar
#         FORCE_REBUILD=1 tools/run_ros_sim.sh
# Commands:  ros2 topic pub --once /follow/cmd std_msgs/String '{data: estop}'
#            ros2 topic echo /follow/state
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WS="$ROOT/ros2_ws"
ENV_NAME="${ROS_ENV:-ros_env}"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
  if ! command -v micromamba >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$MAMBA_ROOT_PREFIX/etc/profile.d/micromamba.sh"
  fi
  set +u   # conda activate scripts reference unset vars
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate "$ENV_NAME"
  # shellcheck disable=SC1091
  source "$CONDA_PREFIX/setup.bash"
  set -u
fi

# Everything runs on this machine. Set FOLLOWME_ROS_NETWORK=1 to expose topics
# on the LAN (discovery over the network does not work on every Mac setup).
if [[ "${FOLLOWME_ROS_NETWORK:-}" == "1" ]]; then
  export ROS_LOCALHOST_ONLY=0
else
  export ROS_LOCALHOST_ONLY=1
fi
# The nodes import the shared core straight from the repo.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$WS"
if [[ "${FORCE_REBUILD:-}" == "1" ]] || [[ ! -f "$WS/install/setup.bash" ]]; then
  rm -rf "$WS/build/followme_ros" "$WS/install/followme_ros"
  colcon build --packages-select followme_ros
fi
set +u
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u
exec ros2 launch followme_ros sim.launch.py "$@"
