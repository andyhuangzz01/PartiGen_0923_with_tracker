#!/usr/bin/env bash
# Explicit GT IDs; never aliases generated indices or the previous baseline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec .venv/bin/python tools/run_generated.py --gt "$@"
