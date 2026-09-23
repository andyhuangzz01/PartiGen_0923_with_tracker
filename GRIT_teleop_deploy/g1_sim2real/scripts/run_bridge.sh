#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${G1_BRIDGE_BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/g1_udp_bridge"
CONFIG="${G1_BRIDGE_CONFIG:-${ROOT_DIR}/config/g1_bridge.yaml}"
NET="${G1_NET:-lo}"
LOCK_FILE="${G1_BRIDGE_LOCK_FILE:-/tmp/g1_udp_bridge.lock}"

# The Unitree SDK receives the interface explicitly through
# ChannelFactory::Init(0, NET).  A CYCLONEDDS_URI inherited from a ROS/SDK
# shell can override that selection (often pointing at lo or an old NIC),
# leaving the bridge waiting forever for rt/lowstate.  For a non-loopback G1
# run, let the SDK's explicit interface selection be authoritative.
if [[ "${NET}" != "lo" && -n "${CYCLONEDDS_URI:-}" ]]; then
  echo "[run_bridge] ignoring inherited CYCLONEDDS_URI; using SDK interface=${NET}"
  unset CYCLONEDDS_URI
fi

# CycloneDDS may select an automatically-added 169.254/16 address before the
# intended G1 192.168.123.x address when both exist on the robot NIC. Fail
# early with an actionable message instead of waiting forever for LowState.
if [[ "${NET}" != "lo" ]] && command -v ip >/dev/null 2>&1; then
  has_link_local=$(ip -4 -o addr show dev "${NET}" 2>/dev/null | awk '$4 ~ /^169\.254\./ { found = 1 } END { print found + 0 }')
  has_g1_address=$(ip -4 -o addr show dev "${NET}" 2>/dev/null | awk '$4 ~ /^192\.168\.123\./ { found = 1 } END { print found + 0 }')
  if [[ "${has_link_local}" == "1" && "${has_g1_address}" == "1" ]]; then
    echo "[run_bridge] ERROR: ${NET} has a 169.254.x.x address; CycloneDDS may bind the wrong source address." >&2
    echo "[run_bridge] Remove it, then retry: sudo ip addr del <169.254-address>/<prefix> dev ${NET}" >&2
    exit 1
  fi
fi

case "$(uname -m)" in
  x86_64)
    SDK_ARCH="x86_64"
    ;;
  aarch64|arm64)
    SDK_ARCH="aarch64"
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

SDK_LIB_DIR="${ROOT_DIR}/third_party/unitree_sdk2/thirdparty/lib/${SDK_ARCH}"

if [[ ! -x "${BIN}" ]]; then
  echo "Bridge binary not found: ${BIN}" >&2
  echo "Run: bash ${ROOT_DIR}/scripts/build.sh" >&2
  exit 1
fi

if [[ ! -d "${SDK_LIB_DIR}" ]]; then
  echo "Unitree SDK library directory not found: ${SDK_LIB_DIR}" >&2
  exit 1
fi

# A second bridge can split UDP commands and initialize DDS twice on one robot.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another G1 bridge is already running (lock: ${LOCK_FILE})" >&2
  exit 1
fi
if pgrep -x g1_udp_bridge >/dev/null; then
  echo "Another g1_udp_bridge process is already running:" >&2
  pgrep -af g1_udp_bridge >&2
  exit 1
fi

echo "[run_bridge] binary=${BIN}"
echo "[run_bridge] config=${CONFIG}"
echo "[run_bridge] network=${NET}"
echo "[run_bridge] sdk_lib=${SDK_LIB_DIR}"

# Keep the native bridge independent of Conda, ROS, CUDA, and XR service libs.
unset LD_PRELOAD
export LD_LIBRARY_PATH="${SDK_LIB_DIR}"

exec "${BIN}" --net "${NET}" --config "${CONFIG}" "$@"
