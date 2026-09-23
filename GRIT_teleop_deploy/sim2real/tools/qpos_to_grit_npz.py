"""Convert a RobotMDAR G1 23-DOF qpos trajectory to a GRIT reference NPZ.

Input format (RobotMDAR output, e.g. qpos.npy):
    shape (T, 30) float32 at 30 fps, MuJoCo qpos layout for
    description/robots/g1/g1_23dof_lock_wrist.xml:
        [0:3]  pelvis world position
        [3:7]  pelvis world quaternion (xyzw, RobotMDAR native convention)
        [7:30] 23 joint angles (23DOF model order = 29DOF order minus wrists)

Output format (GRIT motion contract, consumed by runtime/motion_sources.py):
    fps, joint_pos (T,29), joint_vel (T,29),
    body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w (T,B,...)
    in the 29DOF g1.xml body order (index 0 = pelvis, quat wxyz).

The GRIT runtime consumes the reference stream at reference_fps=50 Hz, so the
30 fps input is resampled to 50 fps (linear for positions/joints, slerp for
the root quaternion). Wrist joints do not exist in the 23DOF source and are
filled with 0.0 (their default keyframe value).

Usage:
    cd sim2real
    uv run python tools/qpos_to_grit_npz.py \
        --input ../qpos.npy \
        --output config/g1/motions/robotmdar_qpos.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[2]  # GRIT_teleop_deploy/
GRIT_XML = REPO / "sim2real/config/g1/assets/g1.xml"
ROBOTMDAR_XML = REPO / "../description/robots/g1/g1_23dof_lock_wrist.xml"

SRC_FPS = 30.0
DST_FPS = 50.0  # must match reference_fps in config/g1/tracking.yaml


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]


def input_quaternions_to_wxyz(quaternions: np.ndarray, order: str) -> np.ndarray:
    """Normalize input quaternions and return GRIT/MuJoCo wxyz layout."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"quaternions must have shape [T, 4], got {quaternions.shape}")
    if order == "xyzw":
        quaternions = np.roll(quaternions, 1, axis=-1)
    elif order == "wxyz":
        quaternions = quaternions.copy()
    else:
        raise ValueError(f"unsupported quaternion order: {order}")
    quat_norm = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(~np.isfinite(quat_norm)) or np.any(quat_norm < 1e-6):
        raise ValueError("qpos contains an invalid root quaternion")
    return quaternions / quat_norm


def resample(times, root_pos, root_quat_wxyz, joints, dst_fps):
    n_out = int(np.floor(times[-1] * dst_fps)) + 1
    t_out = np.arange(n_out, dtype=np.float64) / dst_fps
    pos_out = np.stack(
        [np.interp(t_out, times, root_pos[:, d]) for d in range(3)], axis=-1
    )
    slerp = Slerp(
        times,
        R.from_quat(np.roll(root_quat_wxyz, -1, axis=-1)),
    )  # wxyz -> xyzw for scipy
    quat_out = np.roll(slerp(t_out).as_quat(), 1, axis=-1)  # xyzw->wxyz
    joints_out = np.stack(
        [np.interp(t_out, times, joints[:, j]) for j in range(joints.shape[1])],
        axis=-1,
    )
    return t_out, pos_out, quat_out, joints_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="RobotMDAR qpos .npy (T,30)")
    ap.add_argument("--output", type=Path, required=True, help="GRIT .npz output")
    ap.add_argument("--src-fps", type=float, default=SRC_FPS)
    ap.add_argument("--dst-fps", type=float, default=DST_FPS)
    ap.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help=(
            "Quaternion layout in input qpos. RobotMDAR emits xyzw; use "
            "wxyz only for an already-converted MuJoCo trajectory."
        ),
    )
    ap.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help=(
            "Optional odd Savitzky-Golay window (source frames) for root "
            "position and joints; 0 disables smoothing"
        ),
    )
    args = ap.parse_args()

    if args.src_fps <= 0.0 or args.dst_fps <= 0.0:
        raise ValueError("--src-fps and --dst-fps must be positive")

    qpos = np.load(args.input, allow_pickle=False).astype(np.float64)
    m23 = mujoco.MjModel.from_xml_path(str(ROBOTMDAR_XML.resolve()))
    if qpos.ndim != 2:
        raise ValueError(f"qpos must have shape [T, nq], got {qpos.shape}")
    if qpos.shape[1] != m23.nq:
        raise ValueError(f"qpos dim {qpos.shape[1]} != 23DOF model nq {m23.nq}")
    n_src = qpos.shape[0]
    if n_src < 2:
        raise ValueError("qpos must contain at least two frames")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains NaN or Inf")
    times = np.arange(n_src, dtype=np.float64) / args.src_fps

    root_pos = qpos[:, 0:3]
    root_quat = input_quaternions_to_wxyz(qpos[:, 3:7], args.quat_order)
    joints23 = qpos[:, 7:]

    if args.smooth_window:
        if args.smooth_window < 5 or args.smooth_window % 2 == 0:
            raise ValueError("--smooth-window must be 0 or an odd integer >= 5")
        if args.smooth_window > n_src:
            raise ValueError("--smooth-window cannot exceed the source frame count")
        root_pos = savgol_filter(
            root_pos, args.smooth_window, polyorder=2, axis=0, mode="interp"
        )
        joints23 = savgol_filter(
            joints23, args.smooth_window, polyorder=2, axis=0, mode="interp"
        )
        print(
            f"[convert] smoothed root position and joints with "
            f"window={args.smooth_window}, polyorder=2"
        )

    t_out, root_pos, root_quat, joints23 = resample(
        times, root_pos, root_quat, joints23, args.dst_fps
    )
    n = t_out.shape[0]
    print(
        f"[convert] {n_src} frames @ {args.src_fps} Hz -> "
        f"{n} frames @ {args.dst_fps} Hz | input quaternion={args.quat_order}"
    )

    # Map 23DOF joints into the 29DOF reference order; wrists -> 0.
    m29 = mujoco.MjModel.from_xml_path(str(GRIT_XML.resolve()))
    names23 = joint_names(m23)[1:]  # drop floating_base_joint
    names29 = joint_names(m29)[1:]
    if len(names23) != 23 or len(names29) != 29:
        raise ValueError(f"unexpected joint counts: {len(names23)}, {len(names29)}")
    idx23 = {name: i for i, name in enumerate(names23)}
    joints29 = np.zeros((n, 29), dtype=np.float64)
    filled = []
    for j, name in enumerate(names29):
        if name in idx23:
            joints29[:, j] = joints23[:, idx23[name]]
        else:
            filled.append(name)
    print(f"[convert] joints filled with 0 (not in 23DOF source): {filled}")

    # FK with the GRIT 29DOF model. Joint/body rates come from finite
    # differences of the 50 Hz stream; the root angular velocity uses
    # quaternion differences (linear qvel rows are meaningless).
    data = mujoco.MjData(m29)
    qpos29 = np.concatenate([root_pos, root_quat, joints29], axis=1)  # (n,36)
    # qvel layout is nv = 3 + 3 + 29: root linear, root angular, joints.
    qvel = np.zeros((n, m29.nv), dtype=np.float64)
    if n > 1:
        qvel[0, 0:3] = (root_pos[1] - root_pos[0]) * args.dst_fps
        qvel[-1, 0:3] = (root_pos[-1] - root_pos[-2]) * args.dst_fps
        qvel[0, 6:] = (joints29[1] - joints29[0]) * args.dst_fps
        qvel[-1, 6:] = (joints29[-1] - joints29[-2]) * args.dst_fps
    if n > 2:
        qvel[1:-1, 0:3] = (root_pos[2:] - root_pos[:-2]) * (0.5 * args.dst_fps)
        qvel[1:-1, 6:] = (joints29[2:] - joints29[:-2]) * (0.5 * args.dst_fps)
    rots = R.from_quat(np.roll(root_quat, -1, axis=-1))
    if n > 1:
        qvel[0, 3:6] = (rots[0].inv() * rots[1]).as_rotvec() * args.dst_fps
        qvel[-1, 3:6] = (rots[-2].inv() * rots[-1]).as_rotvec() * args.dst_fps
    for i in range(1, n - 1):
        fwd = (rots[i].inv() * rots[i + 1]).as_rotvec()
        bwd = (rots[i].inv() * rots[i - 1]).as_rotvec()
        qvel[i, 3:6] = (fwd - bwd) * (0.5 * args.dst_fps)

    nbody = m29.nbody - 1  # exclude world
    body_pos = np.zeros((n, nbody, 3), dtype=np.float64)
    body_quat = np.zeros((n, nbody, 4), dtype=np.float64)
    body_lin = np.zeros((n, nbody, 3), dtype=np.float64)
    body_ang = np.zeros((n, nbody, 3), dtype=np.float64)
    vel = np.zeros(6, dtype=np.float64)
    for i in range(n):
        data.qpos[:] = qpos29[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(m29, data)
        body_pos[i] = data.xpos[1:]
        body_quat[i] = data.xquat[1:]
        for b in range(1, m29.nbody):
            mujoco.mj_objectVelocity(m29, data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
            body_ang[i, b - 1] = vel[0:3]
            body_lin[i, b - 1] = vel[3:6]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        fps=np.array([args.dst_fps], dtype=np.float32),
        joint_pos=joints29.astype(np.float32),
        joint_vel=qvel[:, 6:].astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin.astype(np.float32),
        body_ang_vel_w=body_ang.astype(np.float32),
    )
    print(f"[convert] saved {args.output}  frames={n} bodies={nbody}")
    print(f"[convert] pelvis z range: {body_pos[:, 0, 2].min():.3f} .. {body_pos[:, 0, 2].max():.3f} m")


if __name__ == "__main__":
    main()
