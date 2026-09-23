#!/usr/bin/env bash
# Run one converted motion from motion_examples_100.pkl in sim or hardware.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 INDEX [--hardware] [extra deploy.py arguments...]" >&2
  echo "Examples: $0 96    $0 96 --hardware" >&2
  exit 2
fi

index=$1
shift
if ! [[ ${index} =~ ^[0-9]+$ ]] || (( 10#${index} < 0 || 10#${index} > 99 )); then
  echo "INDEX must be an integer from 0 to 99" >&2
  exit 2
fi
printf -v padded_index "%02d" "$((10#${index}))"

hardware_mode=false
if [[ ${1:-} == "--hardware" ]]; then
  hardware_mode=true
  shift
fi

matches=(config/g1/motions/examples100/"${padded_index}"_*.npz)
if [[ ${#matches[@]} -ne 1 || ! -f ${matches[0]} ]]; then
  echo "Expected exactly one motion for index ${padded_index}" >&2
  exit 1
fi

if ${hardware_mode}; then
  candidate_file=config/g1/motions/examples100/hardware_candidates.txt
  if ! grep -Eq "^(tier1|tier2)[[:space:]]+${padded_index}([[:space:]]|$)" "${candidate_file}"; then
    echo "Motion ${padded_index} is not approved for initial hardware trials." >&2
    echo "See ${candidate_file}; validate it in simulation first." >&2
    exit 1
  fi
  echo "[run_deploy_example100] HARDWARE candidate=${padded_index} motion=${matches[0]}"
else
  echo "[run_deploy_example100] SIM motion=${matches[0]}"
fi

exec uv run src/deploy.py --robot g1 \
  --motion-file "${matches[0]}" \
  --policy-path checkpoints/policy.onnx "$@"
