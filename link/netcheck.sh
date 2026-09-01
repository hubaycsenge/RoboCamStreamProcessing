#!/usr/bin/env bash
# Check whether the robot can actually reach the server's stream port.
#
# Run this ON THE ROBOT (the Orin) before debugging anything in Python — most
# "the stream does not work" problems are a firewall or a wrong address, and
# this tells the two apart in a few seconds.
#
#   ./netcheck.sh                       # the local end of the tunnel (normal)
#   ./netcheck.sh nipg1.inf.elte.hu 22  # is the rendezvous host reachable?
#
# The default is 127.0.0.1 on purpose. The robot never dials a cluster address:
# it is behind the lab router's NAT, so mecanumbot-deep3r-tunnel.service dials
# out to nipg1 and forwards the server's port back to the robot's loopback. So
# the question that matters is "is the tunnel up and is something behind it",
# and 127.0.0.1:5555 is where that gets answered.
#
# Testing 10.128.17.196 (nipg36's LAN address) was the old default and is a
# false lead: it is unreachable from the robot and always was.

set -uo pipefail

HOST="${1:-127.0.0.1}"
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
Things to check, in order — the path is
robot:${PORT} -> nipg1:${PORT} -> compute node:${PORT} (server), so work along it:

  1. Is the robot's tunnel up?
       systemctl status mecanumbot-deep3r-tunnel     (or --user)
     It restarts every 10 s when nothing is bridged on nipg1, which is the
     normal look of "no server allocated right now", not a fault.

  2. Is a job bridged to nipg1? On nipg1:
       squeue -u \$USER
       ss -ltn | grep ${PORT}      # the rendezvous port should be LISTEN
     If squeue is empty, allocate one and run scripts/run_deep3r_bridged.sh.

  3. Is the server itself up inside that job? Its output has
     "listening on tcp://0.0.0.0:${PORT}" once the model has loaded. The 512
     DPT checkpoint takes tens of seconds, so a fresh job is briefly normal
     to fail this check.

  4. Stale port: if a previous job's forward still holds ${PORT} on nipg1,
     the new job's tunnel cannot bind and exits. 'ss -ltnp | grep ${PORT}'
     on nipg1 finds it; kill it, or use DEEP3R_BRIDGE_PORT on both ends.

Note none of this involves reaching a cluster address from the robot. If you
are testing 10.128.17.196 or any other 10.128.17.x address, that is the old
(and always mistaken) idea of how this works — see link/README.md.
EOF
fi

exit $RC
