#!/bin/sh
# Install the boot and auto-stop units on the EC2 host.
#   sudo deploy/aws/install.sh [HOURS]    (default: power off 3 h after boot)
set -eu

HOURS="${1:-3}"
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"
UNITS=/etc/systemd/system

sed "s|@REPO@|$REPO|" "$DIR/traffic-monitoring.service" > "$UNITS/traffic-monitoring.service"
cp "$DIR/traffic-autostop.service" "$UNITS/traffic-autostop.service"
sed "s|@HOURS@|$HOURS|g" "$DIR/traffic-autostop.timer" > "$UNITS/traffic-autostop.timer"

systemctl daemon-reload
systemctl enable traffic-monitoring.service traffic-autostop.timer
# Restart so the countdown starts now rather than from the original boot
systemctl restart traffic-autostop.timer

echo "Stack starts on boot from $REPO; host powers off ${HOURS} h after each start."
systemctl list-timers traffic-autostop.timer --no-pager
