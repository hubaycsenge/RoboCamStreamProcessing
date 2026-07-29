#!/usr/bin/env bash
# Set up the server-side environment on nipg36.
#
# No root required: uv installs into ~/.local/bin and manages its own Python.
# The system Python here is 3.8, which is too old for current torch, so we pin
# our own 3.11 rather than fighting the distribution.
#
# Re-running this is safe.

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
