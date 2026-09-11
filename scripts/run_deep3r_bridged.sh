#!/usr/bin/env bash
# Run the deep3r server on this Slurm compute node and bridge it to nipg1, where
# the robot's forward tunnel is waiting. This is the cluster-side half of the
# path that link/mecanumbot-deep3r-tunnel.service is the robot-side half of:
#
#   robot:5555 --ssh -L--> nipg1:5555 <--ssh -R-- this node :5555 (the server)
#
# Use it instead of run_deep3r.sh whenever a real robot is connected.
# run_deep3r.sh on its own is still the right thing for local benchmarking,
# where nothing has to cross the cluster boundary.
#
#   salloc --no-shell --gres=gpu:1 -c 8 --mem=24G -t 08:00:00
#   srun --jobid=<id> --overlap ./scripts/run_deep3r_bridged.sh
#
# Note the absence of `-w nipg36`. That constraint is gone, and removing it is
# the whole point: the robot now meets the server at a fixed rendezvous on
# nipg1, so the *compute* is free to land wherever a GPU is. Prefer a node with
# Ampere or newer -- nipg10's 3090s are sm_86 and much faster than nipg36's
# TITAN RTX (sm_75, Turing, no bf16), which is what the old pin forced you onto.
#
# Why the job dials out rather than nipg1 dialling in: you cannot ssh *into* a
# compute node. nipg36:22 is connection-refused and Slurm nodes generally do not
# accept inbound ssh, so a `-L` from nipg1 has nothing to connect to. Outbound
# from the node works (confirmed from nipg3 to nipg1:22), so the job reaches out
# and sshd on nipg1 binds the listening end.
#
# Prerequisite, one time, on nipg1: nipg1's own key must be authorised on nipg1,
# or this ssh cannot authenticate. Home is shared over the NAS so the private
# half is already on every compute node; only the public half is missing:
#
#   sed 's/^/restrict,port-forwarding /' ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
#
# `restrict,port-forwarding` because forwarding is all it ever needs to do --
# no shell, no pty, same treatment the robot's key already gets.
#
# Environment overrides:
#   DEEP3R_BRIDGE_HOST   rendezvous host          (default nipg1.inf.elte.hu)
#   DEEP3R_BRIDGE_PORT   port to bind there       (default 5555)
#   DEEP3R_LOCAL_PORT    port the server binds    (default 5555)
#   DEEP3R_BIND_WAIT_S   how long to wait for it  (default 240)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRIDGE_HOST="${DEEP3R_BRIDGE_HOST:-nipg1.inf.elte.hu}"
BRIDGE_PORT="${DEEP3R_BRIDGE_PORT:-5555}"
LOCAL_PORT="${DEEP3R_LOCAL_PORT:-5555}"
BIND_WAIT_S="${DEEP3R_BIND_WAIT_S:-240}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cat >&2 <<MSG
Not inside a Slurm job. This script is meant to run on a compute node, under:

  salloc --no-shell --gres=gpu:1 -c 8 --mem=24G -t 08:00:00
  srun --jobid=<id> --overlap ./scripts/run_deep3r_bridged.sh

Running it on nipg1 would fail anyway: nipg1 is the login node and has no GPU
(it is not in \`sinfo -N\`). Set SLURM_JOB_ID=local to override for a dry run.
MSG
    exit 1
fi

port_open() {
    timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null
}

TUNNEL_PID=""
SERVER_PID=""

cleanup() {
    # Kill the tunnel first: while it lives, nipg1:5555 accepts connections that
    # now lead nowhere, which looks to the robot like a wedged server rather
    # than an absent one.
    [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== deep3r bridged: $(hostname) -> ${BRIDGE_HOST}:${BRIDGE_PORT} (job ${SLURM_JOB_ID}) ==="

if port_open "$LOCAL_PORT"; then
    echo "Something is already listening on 127.0.0.1:${LOCAL_PORT} on $(hostname)." >&2
    echo "Another server from an earlier step, most likely. Stop it first." >&2
    exit 1
fi

./scripts/run_deep3r.sh "$@" &
SERVER_PID=$!

# Wait for the server to actually bind before opening the tunnel. Order matters:
# if the tunnel came up first, a robot connecting during model load would be
# refused at the far end, and the same reasoning that put ExitOnForwardFailure
# in the robot's unit applies here -- a live tunnel should mean a live server,
# never a live socket in front of nothing. Loading the 512 DPT checkpoint is
# tens of seconds, hence the generous default.
echo "waiting up to ${BIND_WAIT_S}s for the server to bind :${LOCAL_PORT} ..."
deadline=$(( SECONDS + BIND_WAIT_S ))
until port_open "$LOCAL_PORT"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server exited before it bound :${LOCAL_PORT}. Its output is above." >&2
        wait "$SERVER_PID"          # surface the real exit status
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Server still not listening after ${BIND_WAIT_S}s; giving up." >&2
        exit 1
    fi
    sleep 2
done
echo "server is listening on :${LOCAL_PORT}"

# Reverse forward, kept up for as long as the server runs. ExitOnForwardFailure
# makes a refused bind fatal rather than silent -- see the stale-port note below.
# The two ways this fails need two different fixes, and ssh already says which
# it is -- so read its stderr and print the one that applies rather than a menu
# of both. Guessing wrong here costs real time: the port advice sends you to
# nipg1 to hunt a stale forward that is not there, while the actual problem is
# one command long.
TUNNEL_ERR="$(mktemp)"
trap 'rm -f "$TUNNEL_ERR"' EXIT

diagnose_tunnel() {
    local rc="$1" err
    err="$(cat "$TUNNEL_ERR" 2>/dev/null || true)"
    [[ -n "$err" ]] && printf '%s\n' "$err" >&2

    echo >&2
    echo "--- reverse tunnel to ${BRIDGE_HOST}:${BRIDGE_PORT} dropped (exit ${rc}) ---" >&2

    if grep -qi 'permission denied' <<<"$err"; then
        cat >&2 <<MSG
AUTHENTICATION. ${BRIDGE_HOST} refused this node's key, so the forward never
opened. Home is shared over the NAS, so the private half of the key is already
here -- what is missing is the public half in nipg1's own authorized_keys.

Fix it once, ON NIPG1:

  sed 's/^/restrict,port-forwarding /' ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys

(\`restrict,port-forwarding\` because forwarding is all this key ever needs to
do -- no shell, no pty. If ~/.ssh/id_ed25519.pub does not exist, make the pair
first with \`ssh-keygen -t ed25519 -N ""\`.)

Then re-run the srun; the allocation is still yours and does not need redoing.
The job id is interpolated below on purpose -- you will be pasting this into a
login shell, where \$SLURM_JOB_ID is empty because it is only set inside a job
step, and \`--jobid=\` with nothing after it becomes job 0:

  srun --jobid=${SLURM_JOB_ID} --overlap ./scripts/run_deep3r_bridged.sh
MSG
    elif grep -qiE 'remote port forwarding failed|cannot listen to port' <<<"$err"; then
        cat >&2 <<MSG
PORT ALREADY HELD. Port ${BRIDGE_PORT} on ${BRIDGE_HOST} is still bound by an
earlier job's forward. sshd will not rebind it, and ExitOnForwardFailure makes
that fatal here rather than leaving a tunnel that goes nowhere. On nipg1:

  ss -ltnp | grep ${BRIDGE_PORT}      # find the stale sshd/ssh holding it
  kill <pid>

Or use a different port at both ends:
  DEEP3R_BRIDGE_PORT=5556 ./scripts/run_deep3r_bridged.sh
(and point the robot's tunnel at 5556 to match).
MSG
    else
        cat >&2 <<MSG
Neither authentication nor a held port, by ssh's own account. The two usual
causes and their fixes are in the header of this script; the ssh output above is
the thing to read first.
MSG
    fi
}

tunnel_loop() {
    local backoff=5
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        if ssh -NT \
            -o ExitOnForwardFailure=yes \
            -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -o StrictHostKeyChecking=accept-new \
            -o BatchMode=yes \
            -R "${BRIDGE_PORT}:127.0.0.1:${LOCAL_PORT}" "$BRIDGE_HOST" \
            2>"$TUNNEL_ERR"; then
            # A clean exit means the far end went away; retry promptly.
            backoff=5
        else
            diagnose_tunnel "$?"
            # Back off. A misconfigured key fails instantly and identically for
            # ever, and hammering it four times a second buries the diagnosis
            # under copies of itself -- which is exactly how this looked the
            # first time it happened.
            echo "retrying in ${backoff}s (Ctrl-C to stop)" >&2
            sleep "$backoff"
            (( backoff = backoff < 60 ? backoff * 2 : 60 ))
        fi
    done
}
tunnel_loop &
TUNNEL_PID=$!

echo "bridged: robot -> ${BRIDGE_HOST}:${BRIDGE_PORT} -> $(hostname):${LOCAL_PORT}"
wait "$SERVER_PID"
