#!/usr/bin/env bash
# Start the stream server. Any extra arguments are passed through, e.g.
#
#   ./scripts/run_server.sh --processor noop --log-level DEBUG
#
# GPU note: nipg36 has two TITAN RTX cards and GPU 0 is usually occupied by
# another user, so we default to GPU 1. Override with CUDA_VISIBLE_DEVICES.
# This has no effect today (no models loaded) but keeps the habit in place.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "No .venv found. Run ./scripts/setup_server.sh first." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec ./.venv/bin/python -m robocam.server --config config/server.yaml "$@"
