#!/usr/bin/env bash
# Terminal 2: GRIT control with the trimmed half-speed TextOp motion.
# Press 's' to enter the controlled standing pose, then 'a' to play.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec uv run src/deploy.py --robot g1 \
  --motion-file config/g1/motions/qpos_fixed_trimmed_slow_0p5x.npz \
  --policy-path checkpoints/policy.onnx "$@"
