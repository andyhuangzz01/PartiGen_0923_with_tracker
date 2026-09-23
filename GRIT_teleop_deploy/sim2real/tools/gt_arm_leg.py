"""Prepare and screen six selected HML3D-G1 motions, preserving all 29 DOFs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

import joblib
import mujoco
import numpy as np
import yaml

from extract_gt_arm_leg import SELECTED
from generated_dataset import digest
from qpos_to_grit_npz import input_quaternions_to_wxyz, resample
from screen_generated_motions import classify, atomic_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "gt_hml3d_arm_leg"
XML = ROOT / "config/g1/assets/g1.xml"
PYTHON = str(ROOT / ".venv/bin/python")


def gt_id(index):
    mid = f"{int(index):06d}"
    if mid not in SELECTED:
        raise ValueError(f"本次 GT 编号仅支持：{', '.join(SELECTED)}；不是生成组 index。")
    return mid


def signature(mid):
    files = [DATA / f"source/motion/{mid}.pkl", DATA / f"source/texts/{mid}.txt",
             XML, Path(__file__), ROOT / "tools/qpos_to_grit_npz.py",
             ROOT / "tools/screen_generated_motions.py", ROOT / "checkpoints/policy.onnx"]
    files += sorted((ROOT / "src").rglob("*.py"))
    files += [ROOT / f"config/g1/{n}.yaml" for n in ("tracking", "controller", "bridge")]
    return {str(p): digest(p) for p in files}


def validate_source(record, model):
    """Verify order against archive local FK, not just an assumed DOF count."""
    pos = np.asarray(record["root_pos"], dtype=float)
    quat = input_quaternions_to_wxyz(record["root_rot"], "xyzw")
    joints = np.asarray(record["dof_pos"], dtype=float)
    local = np.asarray(record["local_body_pos"], dtype=float)
    names = record["link_body_list"]
    fps = float(record["fps"])
    n = len(pos)
    if (n < 2 or pos.shape != (n, 3) or quat.shape != (n, 4)
            or joints.shape != (n, 29) or local.shape != (n, len(names), 3)
            or not math.isfinite(fps) or not 0 < fps <= 1000
            or any(not np.isfinite(a).all() for a in (pos, quat, joints, local))):
        raise ValueError("Invalid GT dimensions or nonfinite values")
    pairs = [(j, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
             for j, name in enumerate(names)]
    pairs = [(j, b) for j, b in pairs if b >= 0]
    shared = {b for _, b in pairs}
    if not set(model.jnt_bodyid[1:]).issubset(shared):
        raise ValueError("GT local FK does not cover all 29 joint bodies")
    data = mujoco.MjData(model)
    worst = 0.0
    for i in range(n):
        data.qpos[:] = np.r_[0, 0, 0, 1, 0, 0, 0, joints[i]]
        mujoco.mj_forward(model, data)
        worst = max(worst, float(np.linalg.norm(
            data.xpos[[b for _, b in pairs]] - local[i, [j for j, _ in pairs]], axis=1).max()))
    if worst > 1e-4:
        raise ValueError(f"GT joint order/model FK mismatch: {worst:.6f} m")
    return pos, quat, joints, fps, worst


def save_reference(path, model, pos, quat, joints, fps):
    q = np.concatenate((pos, quat, joints), axis=1)
    n = len(q)
    vel = np.zeros((n, model.nv))
    for i in range(n):
        left, right = max(0, i - 1), min(n - 1, i + 1)
        mujoco.mj_differentiatePos(model, vel[i], (right - left) / fps, q[left], q[right])
    data = mujoco.MjData(model)
    body_pos = np.empty((n, model.nbody - 1, 3))
    body_quat = np.empty((n, model.nbody - 1, 4))
    lin, ang = np.empty_like(body_pos), np.empty_like(body_pos)
    speed = np.zeros(6)
    for i in range(n):
        data.qpos[:], data.qvel[:] = q[i], vel[i]
        mujoco.mj_forward(model, data)
        body_pos[i], body_quat[i] = data.xpos[1:], data.xquat[1:]
        for b in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, speed, 0)
            ang[i, b - 1], lin[i, b - 1] = speed[:3], speed[3:]
    arrays = dict(fps=np.array([fps]), joint_pos=joints, joint_vel=vel[:, 6:],
                  body_pos_w=body_pos, body_quat_w=body_quat,
                  body_lin_vel_w=lin, body_ang_vel_w=ang)
    np.savez(path, **{k: v.astype(np.float32) for k, v in arrays.items()})


def prepare(index):
    mid = gt_id(index)
    sig = signature(mid)
    folder = DATA / "prepared" / mid
    metadata = folder / "prepared.json"
    if metadata.is_file():
        result = json.loads(metadata.read_text())
        if result["signature"] != sig or any(
                not Path(p).is_file() or digest(Path(p)) != sha
                for p, sha in result["artifacts"].items()):
            raise ValueError(f"GT {mid} 输入/配置/转换文件已变化；拒绝复用旧结果，请重新准备独立目录。")
        return result
    model = mujoco.MjModel.from_xml_path(str(XML))
    record = joblib.load(DATA / f"source/motion/{mid}.pkl")
    pos, quat, joints, fps, fk_error = validate_source(record, model)
    text = (DATA / f"source/texts/{mid}.txt").read_text()
    generated = ROOT.parents[1] / f"TextOpRobotMDAR/gen_test2576_dar0911_512/sample_{SELECTED[mid]:04d}/prompt.txt"
    prompt = generated.read_text().split("prompt: ", 1)[1].strip()
    if prompt not in [line.split("#")[0].strip() for line in text.splitlines()]:
        raise ValueError("Selected GT caption no longer matches generated caption")
    tracking, controller, bridge = (yaml.safe_load((ROOT / f"config/g1/{n}.yaml").read_text())
                                    for n in ("tracking", "controller", "bridge"))
    ref_fps = float(tracking["reference_fps"])
    _, out_pos, out_quat, out_joints = resample(np.arange(len(pos)) / fps, pos, quat, joints, ref_fps)
    folder.mkdir(parents=True, exist_ok=True)
    motion = folder / "motion.npz"
    if motion.exists():
        raise ValueError(f"Incomplete previous conversion exists: {motion}")
    save_reference(motion, model, out_pos, out_quat, out_joints, ref_fps)
    port = 65000 + 2 * list(SELECTED).index(mid)
    bridge["udp"].update(state_host="127.0.0.1", cmd_bind_host="127.0.0.1", state_port=port, cmd_port=port+1)
    controller["udp"].update(state_bind_host="127.0.0.1", cmd_host="127.0.0.1", state_port=port, cmd_port=port+1)
    for name, cfg in (("bridge", bridge), ("controller", controller)):
        (folder / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
    seconds = (tracking["motion_source"]["npz"]["first_frame_transition_s"]
               + len(out_pos) / ref_fps + tracking["transition_steps"] / ref_fps + 4)
    commands = dict(deploy=["--max-policy-steps", str(math.ceil(seconds * controller["control_freq"]))],
                    sim=["--max-control-seconds", str(seconds + 10)])
    excess = np.maximum(model.jnt_range[1:, 0] - joints, joints - model.jnt_range[1:, 1])
    excess[:, ~model.jnt_limited[1:].astype(bool)] = 0
    result = dict(index=int(mid), gt_id=mid, generated_index=SELECTED[mid], text=prompt,
                  rating="NOT_EVALUATED", hardware_screen="SIM_ONLY", error=None,
                  attempt_dir=str(folder), npz_file=str(motion), preview_commands=commands,
                  signature=sig, source_frames=len(pos), source_fps=fps,
                  expected_motion_frames=len(out_pos), expected_policy_seconds=seconds,
                  source_fk_max_error_m=fk_error, source_joint_order="GRIT XML; validated against GT local FK",
                  preserved_dofs=29, trimmed_frames=0,
                  source_joint_speed_max_rad_s=float(np.abs(np.diff(joints, axis=0)).max() * fps),
                  source_limit_excess_rad=float(max(0, excess.max())),
                  source_xy_path_m=float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum()),
                  source_leg_excursion_rad=float(np.ptp(joints[:, :12], axis=0).max()))
    result["artifacts"] = {str(p): digest(p) for p in (motion, folder / "bridge.yaml", folder / "controller.yaml")}
    atomic_json(metadata, result)
    print(f"GT {mid}: {len(pos)} -> {len(out_pos)} frames, preserved 29 DOFs; FK error={fk_error:.3g} m", flush=True)
    return result


def get_result(index):
    result = prepare(index)
    latest = Path(result["attempt_dir"]) / "latest_evaluation.json"
    if latest.is_file():
        evaluated = json.loads(latest.read_text())
        if evaluated["signature"] == result["signature"] and evaluated["artifacts"] == result["artifacts"]:
            return evaluated
    return result


def evaluate(index):
    from run_generated import build_commands, stop_child
    result, sim, deploy = build_commands(index, False, gt=True)
    result = prepare(index).copy()
    folder = Path(tempfile.mkdtemp(prefix="eval_", dir=result["attempt_dir"]))
    sim += ["--headless", "--metrics-out", str(folder / "metrics.json"),
            "--tracking-out", str(folder / "tracking.npz")]
    atomic_json(folder / "commands.json", dict(sim=sim, deploy=deploy))
    processes = []
    try:
        cfg = yaml.safe_load((Path(result["attempt_dir"]) / "bridge.yaml").read_text())
        for port in (cfg["udp"]["state_port"], cfg["udp"]["cmd_port"]):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind(("127.0.0.1", port))
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (folder / "sim.log").open("w") as sl, (folder / "deploy.log").open("w") as dl:
            processes.append(subprocess.Popen(sim, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=sl, stderr=subprocess.STDOUT))
            time.sleep(0.5)
            if processes[0].poll() is not None:
                raise RuntimeError("MuJoCo startup failed")
            processes.append(subprocess.Popen(deploy, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=dl, stderr=subprocess.STDOUT))
            processes[1].wait(timeout=result["expected_policy_seconds"] + 65)
            processes[0].wait(timeout=10)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Nonzero simulation process exit")
        logs = (folder / "deploy.log").read_text()
        if "Traceback" in logs or "An exception occurred" in logs:
            raise RuntimeError("Controller exception")
        for marker in ("Playing 'cli_motion' from start", "Returning to default pose", "reached --max-policy-steps="):
            if marker not in logs:
                raise RuntimeError(f"Incomplete lifecycle: {marker}")
        metrics = json.loads((folder / "metrics.json").read_text())
        frames = result["expected_motion_frames"]
        result.update(metrics)
        if (not metrics.get("tracking_motion_frames_complete") or metrics.get("tracking_recorded_frames") != frames
                or metrics.get("tracking_missing_state_snapshots") != 0
                or metrics.get("tracking_sim_dt_max_abs_error_s", math.inf) > 1e-6):
            raise RuntimeError("Incomplete or misaligned tracking")
        seconds = result["expected_policy_seconds"]
        if not 0.95 * seconds <= metrics["simulated_control_seconds"] <= 1.1 * seconds:
            raise RuntimeError("Control/physics clock mismatch")
        skipped = sum(int(n) for n in re.findall(r"skipped (\d+) bridge state packet", logs.split("Running high level...")[-1]))
        result["skipped_policy_state_packets"] = skipped
        if skipped > max(2, seconds * 50 * .01):
            raise RuntimeError("Excess policy state packet loss")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for p in reversed(processes):
            stop_child(p)
    result["rating"], result["hardware_screen"], result["reasons"] = classify(result)
    result["evaluation_dir"] = str(folder)
    def finite(v):
        if isinstance(v, dict): return {k: finite(x) for k, x in v.items()}
        if isinstance(v, list): return [finite(x) for x in v]
        return None if isinstance(v, float) and not math.isfinite(v) else v
    result = finite(result)
    atomic_json(folder / "result.json", result)
    atomic_json(Path(result["attempt_dir"]) / "latest_evaluation.json", result)
    print(f"GT {result['gt_id']}: {result['rating']} / {result['hardware_screen']} "
          f"Global={result.get('global_mpjpe_mm')} Root={result.get('root_relative_mpjpe_mm')} "
          f"reasons={result['reasons']}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    for mid in SELECTED:
        evaluate(int(mid)) if args.evaluate else prepare(int(mid))
