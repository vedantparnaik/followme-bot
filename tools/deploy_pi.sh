#!/usr/bin/env bash
# Copy the Pi-side server to the rover and restart it.
#
#   PI_HOST=raspberrypi.local PI_USER=pi tools/deploy_pi.sh
#   PI_SONAR=1 tools/deploy_pi.sh        # also start the HC-SR04 driver
#   tools/deploy_pi.sh --no-restart
#
# Uses SSH keys. One-time setup:  ssh-copy-id "$PI_USER@$PI_HOST"
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${PI_HOST:-raspberrypi.local}"
USER_="${PI_USER:-pi}"
DEST="${PI_DIR:-followme}"
T="$USER_@$HOST"
SSH=(ssh -o ConnectTimeout=6 -o BatchMode=yes)

"${SSH[@]}" "$T" "mkdir -p ~/$DEST"
scp -o ConnectTimeout=6 -o BatchMode=yes robot/pi/fpv_server.py robot/pi/range_sensors.py "$T:~/$DEST/"
if [[ "${1:-}" == "--no-restart" ]]; then
  echo "copied to $T:~/$DEST (not restarted)"
  exit 0
fi

ENVS=""
[[ "${PI_SONAR:-}" == "1" ]] && ENVS="FOLLOWME_SONAR=1"
"${SSH[@]}" "$T" "pkill -f '[f]pv_server.py' || true; sleep 1; cd ~/$DEST && \
  $ENVS setsid nohup python3 fpv_server.py > fpv.log 2>&1 < /dev/null & sleep 2; echo launched"
sleep 1
curl -s -o /dev/null -w "teleop page http %{http_code}\n" --max-time 8 "http://$HOST:8000/" || true
echo "teleop: http://$HOST:8000/"
