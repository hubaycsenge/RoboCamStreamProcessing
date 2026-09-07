#!/usr/bin/env bash
# Print the Mecanumbot's LAN address, whichever of the lab's two routers it is on.
#
# The lab has two routers and they hand out different subnets. The robot's last
# octet is reserved, so only the third moves:
#
#     router 192.168.0.1  ->  robot 192.168.0.240
#     router 192.168.1.1  ->  robot 192.168.1.240
#
# Run this ON A MACHINE ON THE SAME WIFI (a laptop). It is not for the robot and
# not for nipg1 -- nipg1 has no route to either subnet, which is the entire
# reason the tunnels in this directory exist.
#
# The use it exists for is the ad-hoc tunnel in link/README.md step 3:
#
#     ssh -R 8222:"$(./robot-addr.sh)":22 csengehubay@nipg1.inf.elte.hu
#
# Hard-coding one address there is the mistake this prevents. The failure it
# produces is `channel 2: open failed: administratively prohibited`, which reads
# like a permissions problem on nipg1 and is really a laptop on the other
# router; nothing in the message says so.
#
#   ./robot-addr.sh          # print the address, or fail with a diagnosis
#   ./robot-addr.sh --all    # print every candidate and whether it answered
#
# Exit 0 with one address on stdout, or exit 1 with an explanation on stderr.
# Everything explanatory goes to stderr so that $(...) captures only the address.

set -uo pipefail

CANDIDATES=(192.168.0.240 192.168.1.240)
PORT="${ROBOT_SSH_PORT:-22}"
TIMEOUT="${ROBOT_PROBE_TIMEOUT:-2}"

probe() {
    # SSH rather than ping: ICMP is often filtered and, more to the point, a
    # robot that answers ping but not 22 is no use for a tunnel. This tests the
    # thing that is actually going to be forwarded.
    if command -v nc >/dev/null 2>&1; then
        nc -z -w "$TIMEOUT" "$1" "$PORT" 2>/dev/null
    else
        timeout "$TIMEOUT" bash -c "exec 3<>/dev/tcp/$1/$PORT" 2>/dev/null
    fi
}

if [[ "${1:-}" == "--all" ]]; then
    for addr in "${CANDIDATES[@]}"; do
        if probe "$addr"; then
            echo "$addr  ssh open"
        else
            echo "$addr  no answer"
        fi
    done
    exit 0
fi

for addr in "${CANDIDATES[@]}"; do
    if probe "$addr"; then
        echo "$addr"
        exit 0
    fi
done

{
    echo "robot-addr.sh: neither ${CANDIDATES[*]} answered on port ${PORT}."
    echo
    echo "This machine's addresses:"
    ip -4 addr show scope global 2>/dev/null | awk '/inet /{print "  " $2 "  (" $NF ")"}' \
        || ifconfig 2>/dev/null | awk '/inet /{print "  " $2}'
    echo
    echo "Check, in order:"
    echo "  1. Is this machine on the lab WiFi at all? If its own address is not"
    echo "     192.168.0.x or 192.168.1.x, it is on a different network and no"
    echo "     amount of probing will reach the robot from here."
    echo "  2. Is the robot powered and associated? Its address is reserved, so"
    echo "     it is .240 on whichever subnet the router it joined hands out."
    echo "  3. If you already have the permanent reverse tunnel up, you do not"
    echo "     need this at all: 'ssh -F ssh_config mecanumbot' goes through"
    echo "     nipg1:8200 and does not care which router the robot is on."
} >&2
exit 1
