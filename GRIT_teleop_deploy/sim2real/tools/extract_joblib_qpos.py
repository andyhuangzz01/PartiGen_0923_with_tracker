"""Extract dataset motion dictionaries from a joblib file as G1 qpos arrays.

The input is expected to be a list of dictionaries whose ``motion`` entry
contains ``root_trans_offset`` (T, 3), ``root_rot`` (T, 4, xyzw), and
``dof`` (T, 23).  Each output is a standard TextOp-compatible (T, 30) NPY.

Joblib/pickle input must be trusted because loading it may execute code.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "motion"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = joblib.load(args.input)
    if not isinstance(records, list):
        raise TypeError("expected a list of motion dictionaries")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, record in enumerate(records):
        motion = record["motion"]
        fps = float(motion["fps"])
        root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)
        root_quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
        joints = np.asarray(motion["dof"], dtype=np.float64)
        frame_count = root_pos.shape[0]
        if root_pos.shape != (frame_count, 3):
            raise ValueError(f"record {index}: invalid root position shape")
        if root_quat_xyzw.shape != (frame_count, 4):
            raise ValueError(f"record {index}: invalid root quaternion shape")
        if joints.shape != (frame_count, 23):
            raise ValueError(f"record {index}: invalid joint shape")
        if not np.all(np.isfinite(root_pos)) or not np.all(np.isfinite(root_quat_xyzw)) or not np.all(np.isfinite(joints)):
            raise ValueError(f"record {index}: non-finite motion values")

        quat_norm = np.linalg.norm(root_quat_xyzw, axis=1, keepdims=True)
        if np.any(quat_norm < 1e-8):
            raise ValueError(f"record {index}: zero root quaternion")
        root_quat_xyzw = root_quat_xyzw / quat_norm
        root_quat = np.roll(root_quat_xyzw, 1, axis=1)  # xyzw -> wxyz
        qpos = np.concatenate((root_pos, root_quat, joints), axis=1).astype(np.float32)

        source_id = str(record.get("babel_sid", f"motion_{index:02d}"))
        filename = f"{index:02d}_{safe_name(source_id)}.npy"
        output_path = args.output_dir / filename
        np.save(output_path, qpos)

        annotations = record.get("frame_ann", [])
        description = str(annotations[0][2]) if annotations else ""
        joint_velocity = np.diff(joints, axis=0) * fps
        manifest.append(
            {
                "index": index,
                "source": str(record.get("feat_p", source_id)),
                "description": description,
                "fps": fps,
                "frames": frame_count,
                "duration_s": frame_count / fps,
                "qpos_file": filename,
                "root_z_min": float(root_pos[:, 2].min()),
                "root_z_max": float(root_pos[:, 2].max()),
                "max_joint_speed_rad_s": float(np.abs(joint_velocity).max()),
            }
        )
        print(
            f"[{index:02d}] {source_id}: frames={frame_count}, "
            f"z={root_pos[:, 2].min():.3f}..{root_pos[:, 2].max():.3f}, "
            f"max_joint_speed={np.abs(joint_velocity).max():.2f} rad/s"
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(manifest)} motions and {manifest_path}")


if __name__ == "__main__":
    main()
