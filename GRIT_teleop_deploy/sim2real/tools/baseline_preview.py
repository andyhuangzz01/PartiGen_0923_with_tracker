"""Prepare a simulation-only preview of one dar0908 PKL entry."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import tempfile

import joblib
import numpy as np
import yaml

from screen_generated_motions import extract

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "gen_all2576_dar0908_512.pkl"
RESULTS = ROOT.parent / "baseline_dar0908_mpjpe_0000_0499"
PREVIEWS = ROOT.parent / "baseline_sim_previews"
PYTHON = str(ROOT / ".venv/bin/python")


def prepare(index: int, dry_run: bool = False) -> dict:
    if not 0 <= index < 2576:
        raise ValueError("baseline PKL 的 index 范围为 0–2575，不是 src_idx。")
    existing = RESULTS / f"{index:04d}" / "result.json"
    if existing.is_file():
        result = json.loads(existing.read_text())
        if result.get("index") != index:
            raise ValueError("baseline 结果 index 不匹配。")
        if (Path(result.get("npz_file", "")).is_file()
                and (Path(result["attempt_dir"]) / "commands.json").is_file()):
            return result

    records = joblib.load(SOURCE)
    if len(records) != 2576:
        raise ValueError("baseline PKL 样本数量发生变化，请重新核对 index。")
    record = records[index]
    qpos, fps = extract(record, trim=0, order="xyzw")
    tracking = yaml.safe_load((ROOT / "config/g1/tracking.yaml").read_text())
    controller = yaml.safe_load((ROOT / "config/g1/controller.yaml").read_text())
    bridge = yaml.safe_load((ROOT / "config/g1/bridge.yaml").read_text())
    reference_fps = float(tracking["reference_fps"])
    frames = int(math.floor((len(qpos) - 1) / fps * reference_fps)) + 1
    seconds = (float(tracking["motion_source"]["npz"]["first_frame_transition_s"])
               + frames / reference_fps + tracking["transition_steps"] / reference_fps + 4.0)
    if dry_run:
        # Placeholder path is shown but never created in dry-run mode.
        attempt = PREVIEWS / f"{index:04d}_NEW_PREVIEW"
    else:
        PREVIEWS.mkdir(parents=True, exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix=f"{index:04d}_", dir=PREVIEWS))

    state_port = 56000 + 2 * index
    bridge["udp"].update(state_host="127.0.0.1", cmd_bind_host="127.0.0.1",
                         state_port=state_port, cmd_port=state_port + 1)
    controller["udp"].update(state_bind_host="127.0.0.1", cmd_host="127.0.0.1",
                             state_port=state_port, cmd_port=state_port + 1)
    motion = attempt / "motion.npz"
    result = dict(index=index, src_idx=int(record["src_idx"]), text=str(record["text"]),
                  rating="NOT_EVALUATED", hardware_screen="SIM_ONLY", error=None,
                  source_file=str(SOURCE), source_frames=len(qpos), source_fps=fps,
                  trimmed_frames=0, attempt_dir=str(attempt), npz_file=str(motion),
                  preview_not_prepared=dry_run)
    recorded = {
        "deploy": ["--max-policy-steps", str(math.ceil(seconds * controller["control_freq"]))],
        "sim": ["--max-control-seconds", str(seconds + 10)],
    }
    result["preview_commands"] = recorded
    if not dry_run:
        qpath = attempt / "qpos_wxyz.npy"
        np.save(qpath, qpos)
        subprocess.run([PYTHON, str(ROOT / "tools/qpos_to_grit_npz.py"),
                        "--input", str(qpath), "--output", str(motion),
                        "--src-fps", str(fps), "--dst-fps", str(reference_fps),
                        "--quat-order", "wxyz"], cwd=ROOT, check=True, timeout=120)
        with np.load(motion) as clip:
            if len(clip["joint_pos"]) != frames:
                raise ValueError("转换帧数与预期不一致。")
        for name, config in (("bridge", bridge), ("controller", controller)):
            (attempt / f"{name}.yaml").write_text(yaml.safe_dump(config))
        (attempt / "commands.json").write_text(json.dumps(recorded, indent=2))
        (attempt / "preview.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
