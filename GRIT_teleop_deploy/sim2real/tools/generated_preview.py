"""Cached, simulation-only preparation of dar0911 sample directories."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import yaml

from generated_dataset import digest, parse_prompt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1] / "TextOpRobotMDAR/gen_test2576_dar0911_512"
PREVIEWS = ROOT.parent / "generated_sim_previews"
PYTHON = str(ROOT / ".venv/bin/python")


def prepare(index: int, dry_run: bool = False) -> dict:
    if not 0 <= index < 2576:
        raise ValueError("生成组 sample ID 范围为 0–2575。")
    sample = SOURCE / f"sample_{index:04d}"
    metadata = parse_prompt(sample / "prompt.txt")
    qpath = sample / "qpos.npy"
    qpos = np.load(qpath, allow_pickle=False)
    fps = float(metadata["fps"])
    if (int(metadata["sample_idx"]) != index or qpos.ndim != 2
            or qpos.shape[1] != 30 or len(qpos) < 2
            or len(qpos) != int(metadata["gen_frames"])
            or not np.isfinite(qpos).all() or not math.isfinite(fps) or fps <= 0):
        raise ValueError("生成动作 metadata / qpos / fps 校验失败。")
    configs = {name: ROOT / f"config/g1/{name}.yaml"
               for name in ("tracking", "controller", "bridge")}
    converter = ROOT / "tools/qpos_to_grit_npz.py"
    inputs = [qpath, sample / "prompt.txt", converter, Path(__file__),
              ROOT / "config/g1/assets/g1.xml",
              ROOT.parents[1] / "description/robots/g1/g1_23dof_lock_wrist.xml",
              *configs.values()]
    signature = {str(p): digest(p) for p in inputs}
    for cached in sorted(PREVIEWS.glob(f"sample_{index:04d}_*/preview.json"), reverse=True):
        try:
            result = json.loads(cached.read_text())
            if (result["index"] == index and result["source_signature"] == signature
                    and result["hardware_screen"] == "SIM_ONLY"
                    and all(Path(p).is_file() and digest(Path(p)) == sha
                            for p, sha in result["artifact_sha256"].items())):
                return result
        except (ValueError, KeyError, OSError):
            continue
    tracking, controller, bridge = (yaml.safe_load(configs[n].read_text())
                                    for n in ("tracking", "controller", "bridge"))
    reference_fps = float(tracking["reference_fps"])
    frames = math.floor((len(qpos) - 1) / fps * reference_fps) + 1
    seconds = (float(tracking["motion_source"]["npz"]["first_frame_transition_s"])
               + frames / reference_fps + tracking["transition_steps"] / reference_fps + 4)
    if dry_run:
        attempt = PREVIEWS / f"sample_{index:04d}_NEW_PREVIEW"
    else:
        PREVIEWS.mkdir(parents=True, exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix=f"sample_{index:04d}_", dir=PREVIEWS))
    # Separate from hardware (55001/55002) and baseline previews (56000+).
    state_port = 62000 + 2 * (index % 1000)
    bridge["udp"].update(state_host="127.0.0.1", cmd_bind_host="127.0.0.1",
                         state_port=state_port, cmd_port=state_port + 1)
    controller["udp"].update(state_bind_host="127.0.0.1", cmd_host="127.0.0.1",
                             state_port=state_port, cmd_port=state_port + 1)
    motion = attempt / "motion.npz"
    commands = {
        "deploy": ["--max-policy-steps", str(math.ceil(seconds * controller["control_freq"]))],
        "sim": ["--max-control-seconds", str(seconds + 10)],
    }
    result = dict(index=index, src_idx=int(metadata["src_idx"]), text=metadata["prompt"],
                  rating="NOT_EVALUATED", hardware_screen="SIM_ONLY", error=None,
                  source_file=str(qpath), source_frames=len(qpos), source_fps=fps,
                  trimmed_frames=0, reference_frames=frames, source_signature=signature,
                  attempt_dir=str(attempt), npz_file=str(motion),
                  preview_not_prepared=dry_run, preview_commands=commands)
    if not dry_run:
        subprocess.run([PYTHON, str(converter), "--input", str(qpath),
                        "--output", str(motion), "--src-fps", str(fps),
                        "--dst-fps", str(reference_fps), "--quat-order", "xyzw"],
                       cwd=ROOT, check=True, timeout=120)
        with np.load(motion, allow_pickle=False) as clip:
            if clip["joint_pos"].shape != (frames, 29):
                raise ValueError("转换帧数或关节数量与预期不一致。")
            if any(not np.isfinite(clip[key]).all() for key in clip.files):
                raise ValueError("转换文件包含非有限值。")
        for name, config in (("bridge", bridge), ("controller", controller)):
            (attempt / f"{name}.yaml").write_text(yaml.safe_dump(config))
        (attempt / "commands.json").write_text(json.dumps(commands, indent=2))
        result["artifact_sha256"] = {str(p): digest(p) for p in
                                    (motion, attempt / "bridge.yaml",
                                     attempt / "controller.yaml", attempt / "commands.json")}
        (attempt / "preview.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
