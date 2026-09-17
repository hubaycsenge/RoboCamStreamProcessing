#!/usr/bin/env bash
# Cancel the running server job, pull the latest code, and start it again.
#
#   ./scripts/restart_server.sh                      # T1, fresh allocation
#   ./scripts/restart_server.sh --reuse-alloc        # keep the GPU you already hold
#   ./scripts/restart_server.sh --detach             # return once it is bridged
#   ./scripts/restart_server.sh -- --phase t2 --detector owl --target 'the red mug'
#
# Everything after `--` is passed through to run_deep3r_bridged.sh.
#
# Run this from nipg1. Slurm only works there (the nipg36 container cannot
# resolve the compute-node names and srun fails with `can't find address for
# host nipg3`), and nipg1 is also where the rendezvous port lives, so the port
# check below is only meaningful from here.
#
# Order matters and is deliberate: pull first, cancel second, allocate third.
# A pull that fails -- dirty tree, no key for github -- then costs you nothing,
# whereas cancelling first would leave you with no server and no new code. The
# running server does not mind the working tree changing under it; its modules
# are already imported.
#
# The wait for nipg1:5555 between cancel and allocate is the other half of that
# care. The old job's reverse tunnel holds that port, scancel is asynchronous,
# and sshd will not rebind a port it still holds -- so a new job started too
# eagerly dies on `remote port forwarding failed` with a GPU allocated. Better
# to find that out before spending the allocation.
#
# Environment overrides:
#   SALLOC_ARGS          allocation request   (default --no-shell --gres=gpu:1 ...)
#   DEEP3R_BRIDGE_PORT   rendezvous port      (default 5555)
#   PORT_WAIT_S          wait for it to free  (default 90)
#   BRIDGE_WAIT_S        --detach: wait for the bridge line (default 300)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SERVER_SCRIPT="./scripts/run_deep3r_bridged.sh"
STEP_PATTERN='run_deep3r'
BRIDGE_PORT="${DEEP3R_BRIDGE_PORT:-5555}"
PORT_WAIT_S="${PORT_WAIT_S:-90}"
BRIDGE_WAIT_S="${BRIDGE_WAIT_S:-300}"
read -ra SALLOC_ARGV <<<"${SALLOC_ARGS:---no-shell --gres=gpu:1 -c 8 --mem=24G -t 08:00:00}"

JOB=""
REUSE_ALLOC=0
DO_PULL=1
DETACH=0
EXTRA=()

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'MSG'
Options:
  --job ID        act on this job instead of auto-detecting it
  --reuse-alloc   cancel only the server step and keep the allocation
  --no-pull       skip the git pull (restart on the code already here)
  --detach        run the server under nohup, return once it is bridged
  -h, --help      this text
MSG
}

while (( $# )); do
    case "$1" in
        --job)          JOB="${2:?--job needs a job id}"; shift 2 ;;
        --reuse-alloc)  REUSE_ALLOC=1; shift ;;
        --no-pull)      DO_PULL=0; shift ;;
        --detach)       DETACH=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        --)             shift; EXTRA=("$@"); break ;;
        *)              echo "Unknown option: $1 (pass server arguments after --)" >&2
                        exit 2 ;;
    esac
done

say() { printf '>>> %s\n' "$*"; }

if ! command -v squeue >/dev/null 2>&1; then
    echo "No Slurm here. Run this from nipg1." >&2
    exit 1
fi
if ! squeue -u "$USER" -h >/dev/null 2>&1; then
    echo "squeue does not work from $(hostname). Run this from nipg1." >&2
    exit 1
fi

# --- 1. new code -------------------------------------------------------------

if (( DO_PULL )); then
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "Working tree is dirty; refusing to pull over it:" >&2
        git status --short >&2
        echo >&2
        echo "Commit or stash first, or restart the current code with --no-pull." >&2
        exit 1
    fi
    before="$(git rev-parse --short HEAD)"
    say "pulling $(git rev-parse --abbrev-ref HEAD)"
    git pull --ff-only
    after="$(git rev-parse --short HEAD)"
    if [[ "$before" == "$after" ]]; then
        say "already up to date at $after"
    else
        say "$before -> $after"
        git --no-pager log --oneline "$before..$after" | sed 's/^/    /'
    fi
else
    say "skipping pull, at $(git rev-parse --short HEAD)"
fi

# --- 2. find the running server ----------------------------------------------

# Steps are the reliable signal: the allocation is called `no-shell` (salloc
# names it that, and so is every other --no-shell allocation you may hold),
# while the step carries the script's own name.
mapfile -t STEPS < <(squeue -u "$USER" -h -s -o '%i|%j' 2>/dev/null \
    | awk -F'|' -v pat="$STEP_PATTERN" '$2 ~ pat { print $1 }')

if [[ -z "$JOB" ]]; then
    mapfile -t CANDIDATES < <(printf '%s\n' "${STEPS[@]:-}" \
        | awk -F. 'NF { print $1 }' | sort -u)
    case "${#CANDIDATES[@]}" in
        0)  # No server step. Maybe an allocation is sitting there idle -- take
            # it if it is unambiguous, otherwise just allocate a new one.
            mapfile -t JOBS < <(squeue -u "$USER" -h -o '%i')
            if (( ${#JOBS[@]} == 1 )); then
                JOB="${JOBS[0]}"
                say "no server step running; job $JOB is allocated but idle"
            elif (( ${#JOBS[@]} > 1 )); then
                echo "No server step is running, and you hold ${#JOBS[@]} jobs." >&2
                squeue -u "$USER" >&2
                echo "Name the one to replace with --job ID." >&2
                exit 1
            else
                say "nothing running"
            fi
            ;;
        1)  JOB="${CANDIDATES[0]}" ;;
        *)  echo "The server appears to be running in several jobs:" >&2
            printf '  %s\n' "${CANDIDATES[@]}" >&2
            echo "Name the one to replace with --job ID." >&2
            exit 1 ;;
    esac
fi

# --- 3. stop it --------------------------------------------------------------

# With an explicit --job, the step list gathered above may name steps of other
# jobs too; --reuse-alloc must only cancel steps of the job we are acting on.
if [[ -n "$JOB" ]]; then
    mapfile -t STEPS < <(printf '%s\n' "${STEPS[@]:-}" | grep -x "${JOB}\.[0-9]*" || true)
fi

wait_gone() {   # wait_gone <description> <squeue args...>
    local what="$1"; shift
    local deadline=$(( SECONDS + 60 ))
    while [[ -n "$(squeue -h "$@" 2>/dev/null)" ]]; do
        if (( SECONDS >= deadline )); then
            echo "$what is still in the queue after 60s; giving up." >&2
            exit 1
        fi
        sleep 2
    done
}

if [[ -n "$JOB" ]]; then
    squeue -j "$JOB" 2>/dev/null || true
    if (( REUSE_ALLOC )); then
        if (( ${#STEPS[@]} )); then
            say "cancelling server step(s) ${STEPS[*]}, keeping allocation $JOB"
            scancel "${STEPS[@]}"
            wait_gone "step ${STEPS[*]}" -s -j "$JOB" -o '%i'
        else
            say "reusing allocation $JOB (no step to cancel)"
        fi
    else
        say "cancelling job $JOB"
        scancel "$JOB"
        wait_gone "job $JOB" -j "$JOB" -o '%i'
        JOB=""
    fi
fi

# --- 4. wait for the rendezvous port to come free ----------------------------

port_held() { timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null; }

if port_held "$BRIDGE_PORT"; then
    say "waiting up to ${PORT_WAIT_S}s for :${BRIDGE_PORT} on $(hostname) to free"
    deadline=$(( SECONDS + PORT_WAIT_S ))
    while port_held "$BRIDGE_PORT"; do
        if (( SECONDS >= deadline )); then
            cat >&2 <<MSG

Port ${BRIDGE_PORT} on $(hostname) is still held after ${PORT_WAIT_S}s -- a stale
forward from an earlier job, most likely. sshd will not rebind it, so starting
the server now would only burn an allocation on ExitOnForwardFailure. Find and
kill the holder:

$(ss -ltnp 2>/dev/null | grep ":${BRIDGE_PORT}" | sed 's/^/  /' || echo "  ss -ltnp | grep ${BRIDGE_PORT}")

  kill <pid>

Then re-run this script with --no-pull (the code is already pulled).
MSG
            exit 1
        fi
        sleep 2
    done
fi
say "rendezvous :${BRIDGE_PORT} is free"

# --- 5. allocate -------------------------------------------------------------

if [[ -z "$JOB" ]]; then
    say "salloc ${SALLOC_ARGV[*]}"
    alloc_out="$(salloc "${SALLOC_ARGV[@]}" 2>&1)" || {
        printf '%s\n' "$alloc_out" >&2
        echo "salloc failed." >&2
        exit 1
    }
    printf '%s\n' "$alloc_out" | sed 's/^/    /'
    JOB="$(grep -oE 'Granted job allocation [0-9]+' <<<"$alloc_out" | grep -oE '[0-9]+$' || true)"
    if [[ -z "$JOB" ]]; then
        echo "Could not read a job id out of salloc's output (above)." >&2
        exit 1
    fi
fi

NODE="$(squeue -h -j "$JOB" -o '%R' 2>/dev/null | tr -d ' ')"
say "job $JOB on ${NODE:-?}"

# --- 6. start ----------------------------------------------------------------

if (( DETACH )); then
    mkdir -p logs
    LOG="logs/deep3r-${JOB}.log"
    : >"$LOG"
    nohup srun --jobid="$JOB" --overlap "$SERVER_SCRIPT" "${EXTRA[@]}" >>"$LOG" 2>&1 &
    SRUN_PID=$!
    say "started in the background, logging to $LOG"
    say "waiting up to ${BRIDGE_WAIT_S}s for the bridge"
    deadline=$(( SECONDS + BRIDGE_WAIT_S ))
    until grep -q '^bridged:' "$LOG" 2>/dev/null; do
        if ! kill -0 "$SRUN_PID" 2>/dev/null; then
            echo "The job exited before it bridged. Tail of $LOG:" >&2
            tail -n 30 "$LOG" >&2
            exit 1
        fi
        if (( SECONDS >= deadline )); then
            echo "No bridge line after ${BRIDGE_WAIT_S}s. It may still be loading; watch:" >&2
            echo "  tail -f $LOG" >&2
            exit 1
        fi
        sleep 2
    done
    grep '^bridged:' "$LOG"
    echo
    echo "Follow it with:  tail -f $LOG"
    echo "Stop it with:    scancel $JOB"
else
    echo
    say "srun --jobid=$JOB --overlap $SERVER_SCRIPT ${EXTRA[*]}"
    echo "    (Ctrl-C stops the server; allocation $JOB stays yours)"
    echo
    # The allocation outlives the step, so say how to get back into it rather
    # than leaving you to reconstruct the command with the job id in it.
    trap 'echo; echo "Server stopped. Allocation $JOB is still yours:"; 
          echo "  srun --jobid=$JOB --overlap $SERVER_SCRIPT"; 
          echo "  scancel $JOB     # to give it back"' EXIT
    srun --jobid="$JOB" --overlap "$SERVER_SCRIPT" "${EXTRA[@]}"
fi
