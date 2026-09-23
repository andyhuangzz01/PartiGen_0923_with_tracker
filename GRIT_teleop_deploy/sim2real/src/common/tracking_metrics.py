"""Body-point tracking metrics for closed-loop MuJoCo evaluation.

All points are MuJoCo body-frame origins (``MjData.xpos``), never inertial
centres (``xipos``).  The fixed set deliberately uses one representative
origin per anatomical landmark instead of every mechanical axis, so coincident
waist/body origins do not receive accidental extra weight.
"""
from __future__ import annotations

from typing import Mapping

import mujoco
import numpy as np


BODY_POINT_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "left_toe_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "right_toe_link",
    "torso_link",
    "head_mimic",
    "left_shoulder_pitch_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_hand_mimic",
    "right_shoulder_pitch_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_hand_mimic",
)
ROOT_BODY_NAME = "pelvis"
BODY_GROUPS = {
    "left_leg": (
        "left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link", "left_toe_link",
    ),
    "right_leg": (
        "right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link", "right_toe_link",
    ),
    "torso": ("pelvis", "torso_link", "head_mimic"),
    "both_arms": (
        "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_roll_link", "left_hand_mimic",
        "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_roll_link", "right_hand_mimic",
    ),
}


def resolve_body_points(model: mujoco.MjModel) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Resolve and validate the fixed body-origin contract for one MJCF."""
    ids = []
    for name in BODY_POINT_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id <= 0:
            raise ValueError(f"required non-world body is missing: {name!r}")
        ids.append(body_id)
    if len(set(ids)) != len(ids):
        raise ValueError("tracking body set contains duplicate body IDs")
    by_name = {name: i for i, name in enumerate(BODY_POINT_NAMES)}
    groups = {name: [by_name[body] for body in bodies] for name, bodies in BODY_GROUPS.items()}
    if ROOT_BODY_NAME not in by_name:
        raise ValueError("root body must be included in the tracking point set")
    return np.asarray(ids, dtype=np.int32), groups


def compute_mpjpe(
    actual_pos_w: np.ndarray,
    reference_pos_w: np.ndarray,
    actual_root_w: np.ndarray,
    reference_root_w: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Compute global and translation-only root-relative MPJPE in millimetres."""
    actual = np.asarray(actual_pos_w, dtype=np.float64)
    reference = np.asarray(reference_pos_w, dtype=np.float64)
    actual_root = np.asarray(actual_root_w, dtype=np.float64)
    reference_root = np.asarray(reference_root_w, dtype=np.float64)
    if actual.shape != reference.shape or actual.ndim != 3 or actual.shape[-1] != 3:
        raise ValueError(f"body positions must share shape [T,K,3], got {actual.shape} and {reference.shape}")
    if actual_root.shape != (actual.shape[0], 3) or reference_root.shape != actual_root.shape:
        raise ValueError("root positions must share shape [T,3]")
    if actual.shape[0] == 0 or actual.shape[1] == 0:
        raise ValueError("MPJPE requires at least one frame and one body point")
    if not all(np.isfinite(x).all() for x in (actual, reference, actual_root, reference_root)):
        raise ValueError("MPJPE inputs contain NaN or Inf")

    global_error_mm = np.linalg.norm(actual - reference, axis=-1) * 1000.0
    actual_rel = actual - actual_root[:, None, :]
    reference_rel = reference - reference_root[:, None, :]
    root_relative_error_mm = np.linalg.norm(actual_rel - reference_rel, axis=-1) * 1000.0
    global_frame = global_error_mm.mean(axis=1)
    root_relative_frame = root_relative_error_mm.mean(axis=1)
    return {
        "global_error_mm": global_error_mm,
        "root_relative_error_mm": root_relative_error_mm,
        "global_frame_mpjpe_mm": global_frame,
        "root_relative_frame_mpjpe_mm": root_relative_frame,
        "global_mpjpe_mm": float(global_frame.mean()),
        "root_relative_mpjpe_mm": float(root_relative_frame.mean()),
    }


def summarize_mpjpe(
    metrics: Mapping[str, np.ndarray | float],
    groups: Mapping[str, list[int]],
) -> dict[str, object]:
    """Summarize per-motion curves; P95 is over per-frame mean point error."""
    global_frame = np.asarray(metrics["global_frame_mpjpe_mm"], dtype=np.float64)
    local_frame = np.asarray(metrics["root_relative_frame_mpjpe_mm"], dtype=np.float64)
    local_points = np.asarray(metrics["root_relative_error_mm"], dtype=np.float64)
    result: dict[str, object] = {
        "global_mpjpe_mm": float(metrics["global_mpjpe_mm"]),
        "root_relative_mpjpe_mm": float(metrics["root_relative_mpjpe_mm"]),
        "global_frame_mpjpe_p95_mm": float(np.percentile(global_frame, 95)),
        "global_frame_mpjpe_max_mm": float(global_frame.max()),
        "root_relative_frame_mpjpe_p95_mm": float(np.percentile(local_frame, 95)),
        "root_relative_frame_mpjpe_max_mm": float(local_frame.max()),
    }
    result["root_relative_group_mpjpe_mm"] = {
        name: float(local_points[:, indices].mean()) for name, indices in groups.items()
    }
    return result
