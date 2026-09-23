#!/usr/bin/env bash
# Launch the evaluated dar0911 generated motion by sample index.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec .venv/bin/python tools/run_generated.py "$@"
