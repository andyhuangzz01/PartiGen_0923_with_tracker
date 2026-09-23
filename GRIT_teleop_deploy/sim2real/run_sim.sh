#!/usr/bin/env bash
# Terminal 1: MuJoCo simulator (with viewer)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec uv run src/sim2sim.py --robot g1 "$@"
