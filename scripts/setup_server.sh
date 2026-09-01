#!/usr/bin/env bash
# Set up the server-side environment on a cluster node.
#
# No root required: uv installs into ~/.local/bin and manages its own Python.
# The system Python on these nodes is 3.8, which is too old for current torch,
# so we pin our own 3.11 rather than fighting the distribution.
#
# Home is shared over the NAS, so the venvs this builds are *visible* from every
# node -- but visible is not the same as usable, and there is a trap here:
#
# A venv hardcodes the absolute path of the interpreter it was built against.
# The venvs currently in this checkout point at
#   /home/csengehubay/.local/share/uv/python/cpython-3.11-.../bin/python3.11
# because they were built inside the nipg36 container, where home is mounted at
# /home/csengehubay. That path does not exist on nipg1 or on any compute node
# (HOME is /nas/home/csengehubay-1000257 there), so .venv/bin/python is a
# dangling symlink everywhere except nipg36 and the venvs are dead on arrival.
#
# So before running the server on any node other than nipg36, rebuild both venvs
# by running this script from a node that sees the NAS home path -- nipg1 will
# do. uv then records /nas/home/... and the result works everywhere.
#
# Re-running this is safe, so if in doubt, rerun.

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo ">>> installing uv into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo ">>> uv $(uv --version)"

if [[ ! -d .venv ]]; then
    echo ">>> creating .venv with Python ${PYTHON_VERSION}"
    uv venv --python "${PYTHON_VERSION}" .venv
fi

echo ">>> installing dependencies"
VIRTUAL_ENV="$ROOT/.venv" uv pip install pyzmq numpy opencv-python-headless pyyaml pytest

echo ">>> verifying"
./.venv/bin/python - <<'PY'
import cv2, numpy, yaml, zmq
print(f"  pyzmq  {zmq.__version__}")
print(f"  numpy  {numpy.__version__}")
print(f"  opencv {cv2.__version__}")
print(f"  pyyaml {yaml.__version__}")
PY

echo
echo "Done. Start the server with:  ./scripts/run_server.sh"
