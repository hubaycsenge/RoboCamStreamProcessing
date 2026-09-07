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
# It also reports which of the lab's two routers the robot is on. The lab hands
# out two subnets and the robot's last octet is reserved on both, so it is
# 192.168.0.240 behind the 192.168.0.1 router and 192.168.1.240 behind the
# 192.168.1.1 one. Nothing on the data path depends on which -- both tunnels are
# opened by the robot, so its own address is never dialled from outside -- but
# it is the first thing worth knowing when something on the LAN cannot find it.
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

echo "--- which lab router ---"
# Which subnet this machine landed on, and whether its address is the one the
# DHCP reservation should have given it. A robot that came up on the right
# router with the wrong address is a reservation that did not apply, which looks
# like a network fault from every other angle.
LAN_ADDR="$(ip -4 addr show scope global 2>/dev/null \
    | awk '/inet 192\.168\.[01]\./{split($2, a, "/"); print a[1]; exit}')"
if [[ -n "${LAN_ADDR:-}" ]]; then
    SUBNET="${LAN_ADDR%.*}"
    echo "  on ${SUBNET}.0/24 (gateway ${SUBNET}.1) as ${LAN_ADDR}"
    if [[ "$LAN_ADDR" == "${SUBNET}.240" ]]; then
        echo "  this is the robot's reserved address on this router -- as expected"
    else
        echo "  NOTE: expected ${SUBNET}.240 here. If this is the robot, its DHCP"
        echo "        reservation did not apply; anything on the LAN looking for"
        echo "        it at .240 will not find it."
    fi
else
    echo "  not on 192.168.0.0/24 or 192.168.1.0/24 -- the two lab subnets."
    echo "  If this is the robot, it has not associated with either lab router."
fi
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

Nor does any of it depend on which lab router the robot is on. Both tunnels are
opened *by* the robot, so 192.168.0.240 and 192.168.1.240 work identically and
neither address is ever dialled from outside. The one command that does need it
is the laptop ad-hoc tunnel, and link/robot-addr.sh finds it for you.
EOF
fi

exit $RC
