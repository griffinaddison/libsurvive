#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="/home/griffin/miniforge3/envs/arx-py310"

if [[ ! -d "$ENV_PATH" ]]; then
    echo "ARX virtual environment not found at $ENV_PATH" >&2
    exit 1
fi

# shellcheck source=/dev/null
set +u
source "/home/griffin/miniforge3/bin/activate" arx-py310
set -u
cd "$REPO_ROOT"

exec python bindings/python/examples/teleop_textual.py "$@"
