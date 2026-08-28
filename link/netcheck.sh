#!/usr/bin/env bash
# Check whether the robot can actually reach the server's stream port.
#
# Run this ON THE ROBOT (the Orin) before debugging anything in Python — most
# "the stream does not work" problems are a firewall or a wrong address, and
# this tells the two apart in a few seconds.
#
#   ./netcheck.sh                       # defaults to nipg36 on the LAN
#   ./netcheck.sh 10.128.17.196 5555

set -uo pipefail

HOST="${1:-10.128.17.196}"
PORT="${2:-5555}"

echo "=== target: ${HOST}:${PORT} ==="
echo

echo "--- this machine ---"
hostname
ip -4 addr show scope global 2>/dev/null | awk '/inet /{print "  " $2 "  (" $NF ")"}' \
    || ifconfig 2>/dev/null | awk '/inet /{print "  " $2}'
echo

echo "--- route to ${HOST} ---"
ip route get "$HOST" 2>/dev/null || echo "  (no route information)"
echo

echo "--- ICMP ---"
if ping -c 2 -W 2 "$HOST" >/dev/null 2>&1; then
    echo "  ping OK"
else
    echo "  ping FAILED (not conclusive: ICMP is often blocked)"
fi
echo

echo "--- TCP ${PORT} ---"
if command -v nc >/dev/null 2>&1; then
    if nc -z -w 3 "$HOST" "$PORT" 2>/dev/null; then
        echo "  TCP connect OK — the server is listening and reachable"
        RC=0
    else
        echo "  TCP connect FAILED"
        RC=1
    fi
else
    # No netcat on a minimal JetPack image; bash's /dev/tcp works everywhere.
    if timeout 3 bash -c "exec 3<>/dev/tcp/${HOST}/${PORT}" 2>/dev/null; then
        echo "  TCP connect OK — the server is listening and reachable"
        RC=0
    else
        echo "  TCP connect FAILED"
        RC=1
    fi
fi
echo

if [[ $RC -ne 0 ]]; then
    cat <<EOF
Things to check, in order:
  1. Is the server actually running?   ssh into nipg36 and look for the
     "listening on tcp://0.0.0.0:${PORT}" line.
  2. Is ${HOST} the right address? Run 'hostname -I' on nipg36.
  3. Are you off the university network? Then the LAN address is unreachable
     and you need the VPN, or the SSH reverse tunnel described in the README.
  4. Host firewall on the server (needs an admin: sudo ufw status).
EOF
fi

exit $RC
