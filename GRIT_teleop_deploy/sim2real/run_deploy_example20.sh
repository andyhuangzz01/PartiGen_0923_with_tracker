#!/usr/bin/env bash
# Run one converted motion from motion_examples_20.pkl.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 INDEX [extra deploy.py arguments...]" >&2
  echo "Example: $0 8" >&2
  exit 2
fi

index=$1
shift
if ! [[ ${index} =~ ^[0-9]+$ ]] || (( 10#${index} < 0 || 10#${index} > 19 )); then
  echo "INDEX must be an integer from 0 to 19" >&2
  exit 2
fi
printf -v padded_index "%02d" "$((10#${index}))"

matches=(config/g1/motions/examples20/"${padded_index}"_*.npz)
if [[ ${#matches[@]} -ne 1 || ! -f ${matches[0]} ]]; then
  echo "Expected exactly one motion for index ${padded_index}" >&2
  exit 1
fi

echo "[run_deploy_example20] motion=${matches[0]}"
exec uv run src/deploy.py --robot g1 \
  --motion-file "${matches[0]}" \
  --policy-path checkpoints/policy.onnx "$@"
