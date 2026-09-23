"""Prepare and repeatedly screen bounded low-impact variants of generated 103/255.

This never talks to DDS or hardware.  It creates immutable NPZ variants and runs
the existing policy against loopback-only MuJoCo using the unchanged screening
thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import gt_arm_leg
import gt_slow
from generated_dataset import digest
from screen_generated_motions import atomic_json, classify


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = ROOT.parent / "gen_test2576_dar0911_mpjpe_0001_0500"
DATA = ROOT.parent / "generated_dynamic_refinement"
XML = ROOT / "config/g1/assets/g1.xml"
PYTHON = str(ROOT / ".venv/bin/python")

# These are new, deliberately lower-impact references.  The originals are never
# overwritten.  255 remains a jump-like squat/rise motion, not a full-height jump.
SPECS = {
    103: dict(name="run_gentle", slowdown_factor=8, lower_scale=.35,
              all_joint_scale=.40, tilt_scale=.35, xy_scale=.35, z_scale=.55),
    255: dict(name="jump_low_impact", slowdown_factor=8, lower_scale=.40,
              all_joint_scale=.50, tilt_scale=.35, xy_scale=.35, z_scale=.20),
}


def source_result(index: int) -> dict:
    if index not in SPECS:
        raise ValueError("仅支持生成组 103 和 255。")
    path = SOURCE_RESULTS / f"{index:04d}/result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("index") != index or result.get("error"):
        raise ValueError(f"原始评估结果无效：{path}")
    motion = Path(result["npz_file"])
    if not motion.is_file():
        raise ValueError(f"缺少原始转换动作：{motion}")
    return result


def transform(index: int, model: mujoco.MjModel, source: dict, default: np.ndarray):
    spec = SPECS[index]
    with np.load(source["npz_file"]) as clip:
        fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
        pos = np.asarray(clip["body_pos_w"][:, 0], dtype=float)
        quat = np.asarray(clip["body_quat_w"][:, 0], dtype=float)
        joints = np.asarray(clip["joint_pos"], dtype=float)
    if joints.shape[1] != 29 or pos.shape != (len(joints), 3) or quat.shape != (len(joints), 4):
        raise ValueError("原始 NPZ 维度不符合 G1 参考动作契约。")

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(1, model.njnt)]
    lower = [j for j, name in enumerate(names)
             if any(part in name for part in ("hip_", "knee_", "ankle_", "waist_"))]
    out_j = default + spec["all_joint_scale"] * (joints - default)
    out_j[:, lower] = default[lower] + spec["lower_scale"] * (joints[:, lower] - default[lower])

    # Keep a real margin inside the model ranges; record any clipping explicitly.
    ranges = np.asarray(model.jnt_range[1:], dtype=float)
    limited = model.jnt_limited[1:].astype(bool)
    before = out_j.copy()
    out_j[:, limited] = np.clip(out_j[:, limited],
                                ranges[limited, 0] + .01, ranges[limited, 1] - .01)
    clipped = float(np.max(np.abs(out_j - before)))

    euler = Rotation.from_quat(np.roll(quat, -1, axis=1)).as_euler("xyz")
    euler[:, :2] *= spec["tilt_scale"]
    out_q = np.roll(Rotation.from_euler("xyz", euler).as_quat(), 1, axis=1)
    out_p = pos.copy()
    out_p[:, :2] = pos[0, :2] + spec["xy_scale"] * (pos[:, :2] - pos[0, :2])
    out_p[:, 2] = pos[0, 2] + spec["z_scale"] * (pos[:, 2] - pos[0, 2])

    # Prevent the transformed kinematic reference itself from penetrating ground.
    toes = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_toe_link", "right_toe_link")]
    data = mujoco.MjData(model)
    root_raise = 0.0
    for frame in range(len(out_p)):
        data.qpos[:] = np.r_[out_p[frame], out_q[frame], out_j[frame]]
        mujoco.mj_forward(model, data)
        correction = max(0.0, .01 - float(data.xpos[toes, 2].min()))
        out_p[frame, 2] += correction
        root_raise = max(root_raise, correction)

    factor = spec["slowdown_factor"]
    _, p, q, j = gt_slow.stretch(out_p, out_q, out_j, fps, factor, 50.0)
    return p, q, j, dict(max_joint_clip_rad=clipped,
                         max_ground_correction_m=root_raise,
                         original_frames=len(joints), original_fps=fps)


def prepare(index: int) -> dict:
    source = source_result(index)
    spec = SPECS[index]
    files = [Path(__file__), Path(source["npz_file"]), XML,
             ROOT / "checkpoints/policy.onnx", ROOT / "config/g1/tracking.yaml",
             ROOT / "config/g1/controller.yaml", ROOT / "config/g1/bridge.yaml"]
    files += sorted((ROOT / "src").rglob("*.py"))
    signature = dict(index=index, source_result_sha256=digest(SOURCE_RESULTS / f"{index:04d}/result.json"),
                     spec=spec, files={str(path): digest(path) for path in files})
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    folder = DATA / f"{index:04d}_{spec['name']}_{fingerprint[:12]}"
    metadata = folder / "prepared.json"
    if metadata.is_file():
        result = json.loads(metadata.read_text())
        if result["signature"] != signature or any(
                not Path(path).is_file() or digest(Path(path)) != sha
                for path, sha in result["artifacts"].items()):
            raise ValueError("调整版文件或输入已变化；拒绝复用。")
        return result

    model = mujoco.MjModel.from_xml_path(str(XML))
    tracking = yaml.safe_load((ROOT / "config/g1/tracking.yaml").read_text())
    default = np.asarray(next(c for c in tracking["motion_clips"]
                              if c["name"] == "default")["joint_pos"], dtype=float)
    p, q, j, changes = transform(index, model, source, default)
    folder.mkdir(parents=True, exist_ok=False)
    motion = folder / "motion.npz"
    gt_arm_leg.save_reference(motion, model, p, q, j, 50.0)

    port = 65300 + 2 * list(SPECS).index(index)
    for name in ("bridge", "controller"):
        cfg = yaml.safe_load((ROOT / f"config/g1/{name}.yaml").read_text())
        cfg["udp"].update(state_host="127.0.0.1", cmd_bind_host="127.0.0.1",
                          state_bind_host="127.0.0.1", cmd_host="127.0.0.1",
                          state_port=port, cmd_port=port + 1)
        # Drop keys that do not belong to this config's UDP schema.
        allowed = ({"state_host", "cmd_bind_host", "state_port", "cmd_port"} if name == "bridge"
                   else {"state_bind_host", "cmd_host", "state_port", "cmd_port", "timeout_s"})
        cfg["udp"] = {key: value for key, value in cfg["udp"].items() if key in allowed}
        (folder / f"{name}.yaml").write_text(yaml.safe_dump(cfg))

    speed = float(np.abs(np.diff(j, axis=0)).max() * 50.0)
    excess = np.maximum(model.jnt_range[1:, 0] - j, j - model.jnt_range[1:, 1])
    excess[:, ~model.jnt_limited[1:].astype(bool)] = 0
    first = float(tracking["motion_source"]["npz"]["first_frame_transition_s"])
    seconds = first + len(j) / 50.0 + tracking["transition_steps"] / 50.0 + 4.0
    result = dict(source)
    result.update(variant=spec["name"], adjustment=spec, changes=changes,
                  signature=signature, fingerprint=fingerprint, attempt_dir=str(folder),
                  npz_file=str(motion), expected_motion_frames=len(j),
                  expected_policy_seconds=seconds, source_joint_speed_max_rad_s=speed,
                  source_limit_excess_rad=float(max(0.0, excess.max())),
                  source_xy_path_m=float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()),
                  source_leg_excursion_rad=float(np.ptp(j[:, :12], axis=0).max()),
                  rating="NOT_EVALUATED", hardware_screen="SIM_ONLY", error=None)
    result["artifacts"] = {str(path): digest(path) for path in
                           (motion, folder / "bridge.yaml", folder / "controller.yaml")}
    atomic_json(metadata, result)
    print(f"Prepared {index}: {spec['name']}, frames={len(j)}, speed={speed:.3f} rad/s", flush=True)
    return result


def evaluate(prepared: dict, repeat: int) -> dict:
    from run_generated import stop_child
    result = dict(prepared)
    folder = Path(tempfile.mkdtemp(prefix=f"eval_{repeat}_", dir=prepared["attempt_dir"]))
    seconds = result["expected_policy_seconds"]
    base = Path(prepared["attempt_dir"])
    # Use the same simulator entry point as the original 1--500 screening run.
    # The diagnostic wrapper adds enough overhead to drop reference snapshots on
    # some machines; any dropped snapshot remains a hard failure below.
    sim = [PYTHON, str(ROOT / "src/sim2sim.py"), "--robot", "g1",
           "--bridge-config", str(base / "bridge.yaml"), "--xml_path", str(XML),
           "--headless", "--auto-start", "--max-control-seconds", str(seconds + 10),
           "--metrics-out", str(folder / "metrics.json"),
           "--tracking-out", str(folder / "tracking.npz")]
    steps = math.ceil(seconds * 50)
    deploy = [PYTHON, str(ROOT / "src/deploy.py"), "--robot", "g1",
              "--controller-config", str(base / "controller.yaml"),
              "--tracking-config", str(ROOT / "config/g1/tracking.yaml"),
              "--policy-path", str(ROOT / "checkpoints/policy.onnx"),
              "--motion-file", result["npz_file"], "--publish-reference", "--auto-start",
              "--max-policy-steps", str(steps)]
    atomic_json(folder / "commands.json", dict(sim=sim, deploy=deploy))
    children = []
    print(f"Evaluating {result['index']} {result['variant']} repeat={repeat}", flush=True)
    try:
        cfg = yaml.safe_load((base / "bridge.yaml").read_text())
        for port in (cfg["udp"]["state_port"], cfg["udp"]["cmd_port"]):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind(("127.0.0.1", port))
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (folder / "sim.log").open("w") as sl, (folder / "deploy.log").open("w") as dl:
            children.append(subprocess.Popen(sim, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=sl, stderr=subprocess.STDOUT))
            time.sleep(.5)
            if children[0].poll() is not None:
                raise RuntimeError("MuJoCo 启动失败")
            children.append(subprocess.Popen(deploy, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=dl, stderr=subprocess.STDOUT))
            children[1].wait(timeout=seconds + 65)
            children[0].wait(timeout=10)
        if any(child.returncode for child in children):
            raise RuntimeError("仿真子进程异常退出")
        logs = (folder / "deploy.log").read_text()
        for marker in ("Playing 'cli_motion' from start", "Returning to default pose",
                       f"reached --max-policy-steps={steps}"):
            if marker not in logs:
                raise RuntimeError(f"动作生命周期不完整：{marker}")
        metrics = json.loads((folder / "metrics.json").read_text())
        result.update(metrics)
        if (not metrics.get("tracking_motion_frames_complete")
                or metrics.get("tracking_recorded_frames") != result["expected_motion_frames"]
                or metrics.get("tracking_missing_state_snapshots") != 0
                or metrics.get("tracking_sim_dt_max_abs_error_s", math.inf) > 1e-6):
            raise RuntimeError("跟踪帧不完整或时钟未对齐")
        if not .95 * seconds <= metrics["simulated_control_seconds"] <= 1.1 * seconds:
            raise RuntimeError("仿真时钟与控制时长不一致")
        skipped = sum(int(n) for n in re.findall(
            r"skipped (\d+) bridge state packet", logs.split("Running high level...")[-1]))
        result["skipped_policy_state_packets"] = skipped
        if skipped > max(2, steps * .01):
            raise RuntimeError("策略状态包丢失过多")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for child in reversed(children):
            stop_child(child)
    result["rating"], result["hardware_screen"], result["reasons"] = classify(result)
    result["evaluation_dir"] = str(folder)
    result["evaluation_repeat"] = repeat
    atomic_json(folder / "result.json", result)
    print(f"{result['index']}: {result['rating']} / {result['hardware_screen']}: {result['reasons']}", flush=True)
    return result


def write_selection(prepared: dict) -> None:
    folder = Path(prepared["attempt_dir"])
    runs = []
    for path in sorted(folder.glob("eval_*/result.json"), key=lambda p: p.stat().st_mtime_ns):
        result = json.loads(path.read_text())
        if result.get("signature") == prepared["signature"] and result.get("artifacts") == prepared["artifacts"]:
            runs.append((path, result))
    latest = runs[-3:]
    qualified = len(latest) == 3 and all(
        result.get("rating") == "PASS"
        and result.get("hardware_screen") in {"CANDIDATE_TIER1", "CANDIDATE_TIER2"}
        and result.get("tracking_motion_frames_complete")
        for _, result in latest)
    selection = dict(index=prepared["index"], variant=prepared["variant"], qualified=qualified,
                     required_consecutive_runs=3,
                     runs=[{"path": str(path), "sha256": digest(path),
                            "rating": result.get("rating"),
                            "hardware_screen": result.get("hardware_screen")}
                           for path, result in latest])
    atomic_json(folder / "selection.json", selection)
    print(f"Selection {prepared['index']}: qualified={qualified}, runs={len(latest)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, choices=sorted(SPECS))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("--repeats 必须是 1–3")
    prepared = prepare(args.index)
    for repeat in range(1, args.repeats + 1):
        evaluate(prepared, repeat)
    write_selection(prepared)


if __name__ == "__main__":
    main()
