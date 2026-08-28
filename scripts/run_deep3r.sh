#!/usr/bin/env bash
# Start the stream server with the CUT3R reconstruction processor.
#
#   ./scripts/run_deep3r.sh                        # 512 DPT checkpoint
#   ./scripts/run_deep3r.sh --log-level DEBUG      # extra args pass through
#
# Why this is separate from run_server.sh: deep3r cannot run in the default
# venv. CUT3R pins numpy==1.26.4 and the server venv is on numpy 2.x, so the
# two cannot share an interpreter. .venv stays the known-good transport
# baseline; .venv-cut3r is the one with torch in it.
#
# GPU note: nipg36 has two TITAN RTX cards and GPU 0 is usually occupied by
# another user, so we default to GPU 1, same as run_server.sh.
#
# Under Slurm, allocate on nipg36 specifically:
#
#   salloc --no-shell -w nipg36 --gres=gpu:1 -c 8 --mem=24G -t 08:00:00
#   srun --jobid=<id> --overlap ./scripts/run_deep3r.sh
#
# It has to be nipg36: that is the only node with a 10.128.17.x address, which
# is the network the robot is on. A job that lands on any other node cannot be
# reached by the robot at all.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV=".venv-cut3r"
if [[ ! -x "$VENV/bin/python" ]]; then
    cat >&2 <<MSG
No $VENV found. Build it with:

  uv venv $VENV --python 3.11
  uv pip install --python $VENV/bin/python torch torchvision \\
      --index-url https://download.pytorch.org/whl/cu121
  uv pip install --python $VENV/bin/python numpy==1.26.4 \\
      opencv-python-headless==4.10.0.84 einops roma huggingface-hub scipy \\
      tqdm pillow safetensors pyzmq pyyaml pytest transformers accelerate \\
      omegaconf setuptools

and clone CUT3R plus its checkpoints (see the README).
MSG
    exit 1
fi

CKPT="${DEEP3R_WEIGHTS:-$HOME/CUT3R/checkpoints/cut3r_512_dpt_4_64.pth}"
if [[ ! -f "$CKPT" ]]; then
    echo "Checkpoint not found: $CKPT" >&2
    echo "Download it from the links in the CUT3R README, or set DEEP3R_WEIGHTS." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# workers is pinned to 1 rather than left to the config: the recurrent state is
# the map, and a second worker stepping it corrupts the reconstruction with no
# error anywhere. The processor refuses to run concurrently, so getting this
# wrong is loud rather than silent -- but it should not come up at all.
exec ./$VENV/bin/python -m robocam.server \
    --config config/server.yaml \
    --processor deep3r \
    --workers 1 \
    "$@"
