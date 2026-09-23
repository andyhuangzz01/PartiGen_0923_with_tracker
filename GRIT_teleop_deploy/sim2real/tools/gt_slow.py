"""Independent, uniformly time-stretched GT copies; no screening overrides."""
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

import gt_arm_leg as gt
from generated_dataset import digest
from qpos_to_grit_npz import resample
from screen_generated_motions import atomic_json, classify

DATA = gt.ROOT.parent / "gt_hml3d_arm_leg_slow"
TARGET_SPEED = 3.5  # margin below the unchanged source-speed screening limit 4.0


def slowdown_factor(speed):
    if not math.isfinite(speed) or speed < 0:
        raise ValueError("Invalid source joint speed")
    return max(2, math.ceil(speed / TARGET_SPEED))


def stretch(pos, quat, joints, fps, factor, reference_fps):
    if not math.isfinite(factor) or factor < 1:
        raise ValueError("Slowdown factor must be >= 1")
    return resample(np.arange(len(pos)) / fps * factor, pos, quat, joints, reference_fps)


def prepare(index):
    base = gt.prepare(index)
    mid = base["gt_id"]
    factor = slowdown_factor(base["source_joint_speed_max_rad_s"])
    sig = dict(base["signature"], slowdown_factor=factor, target_speed=TARGET_SPEED)
    sig[str(Path(__file__))] = digest(Path(__file__))
    folder = DATA / "prepared" / mid
    metadata = folder / "prepared.json"
    if metadata.is_file():
        result = json.loads(metadata.read_text())
        if result["signature"] != sig or any(
                not Path(p).is_file() or digest(Path(p)) != sha
                for p, sha in result["artifacts"].items()):
            raise ValueError(f"GT slow {mid}: 输入/配置/文件已变化；拒绝复用旧结果。")
        return result
    model = mujoco.MjModel.from_xml_path(str(gt.XML))
    raw = joblib.load(gt.DATA / f"source/motion/{mid}.pkl")
    pos, quat, joints, fps, fk_error = gt.validate_source(raw, model)
    tracking, controller, bridge = (yaml.safe_load((gt.ROOT / f"config/g1/{n}.yaml").read_text())
                                    for n in ("tracking", "controller", "bridge"))
    ref_fps = float(tracking["reference_fps"])
    _, out_pos, out_quat, out_joints = stretch(pos, quat, joints, fps, factor, ref_fps)
    folder.mkdir(parents=True, exist_ok=True)
    motion = folder / "motion.npz"
    if motion.exists():
        raise ValueError(f"Incomplete existing slow conversion: {motion}")
    gt.save_reference(motion, model, out_pos, out_quat, out_joints, ref_fps)
    with np.load(motion) as clip:
        if any(not np.isfinite(clip[k]).all() for k in clip.files):
            raise ValueError("Nonfinite slow reference")
        speed = float(np.abs(np.diff(clip["joint_pos"], axis=0)).max() * ref_fps)
        if speed > TARGET_SPEED + 1e-4:
            raise ValueError(f"Slow reference exceeds target speed: {speed}")
    port = 65100 + 2 * list(gt.SELECTED).index(mid)
    bridge["udp"].update(state_host="127.0.0.1", cmd_bind_host="127.0.0.1", state_port=port, cmd_port=port+1)
    controller["udp"].update(state_bind_host="127.0.0.1", cmd_host="127.0.0.1", state_port=port, cmd_port=port+1)
    for name, cfg in (("bridge", bridge), ("controller", controller)):
        (folder / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
    seconds = (tracking["motion_source"]["npz"]["first_frame_transition_s"]
               + len(out_pos) / ref_fps + tracking["transition_steps"] / ref_fps + 4)
    result = dict(base)
    result.update(variant="slow", slowdown_factor=factor, playback_speed=1/factor,
                  original_source_fps=fps, source_fps=fps/factor,
                  original_source_joint_speed_max_rad_s=base["source_joint_speed_max_rad_s"],
                  source_joint_speed_max_rad_s=max(base["source_joint_speed_max_rad_s"]/factor, speed),
                  resampled_joint_speed_max_rad_s=speed, target_reference_joint_speed_rad_s=TARGET_SPEED,
                  source_fk_max_error_m=fk_error, signature=sig, attempt_dir=str(folder),
                  npz_file=str(motion), expected_motion_frames=len(out_pos), expected_policy_seconds=seconds,
                  original_duration_s=(len(pos)-1)/fps, stretched_duration_s=(len(pos)-1)/fps*factor,
                  rating="NOT_EVALUATED", hardware_screen="SIM_ONLY", error=None,
                  preview_commands=dict(
                      deploy=["--max-policy-steps", str(math.ceil(seconds*controller["control_freq"]))],
                      sim=["--max-control-seconds", str(seconds+10)]))
    result["artifacts"] = {str(p): digest(p) for p in (motion, folder / "bridge.yaml", folder / "controller.yaml")}
    atomic_json(metadata, result)
    print(f"GT {mid}: {factor}x duration, {1/factor:.3f}x speed, "
          f"{len(out_pos)} frames, reference peak={speed:.3f} rad/s", flush=True)
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
    _, sim, deploy = build_commands(index, False, gt=True, slow=True)
    result = prepare(index).copy()
    folder = Path(tempfile.mkdtemp(prefix="eval_", dir=result["attempt_dir"]))
    sim += ["--headless", "--metrics-out", str(folder/"metrics.json"), "--tracking-out", str(folder/"tracking.npz")]
    atomic_json(folder/"commands.json", dict(sim=sim, deploy=deploy))
    processes = []
    print(f"Evaluating slow GT {result['gt_id']} ({result['stretched_duration_s']:.1f}s motion)...", flush=True)
    try:
        cfg = yaml.safe_load((Path(result["attempt_dir"])/"bridge.yaml").read_text())
        for port in (cfg["udp"]["state_port"], cfg["udp"]["cmd_port"]):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind(("127.0.0.1", port))
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (folder/"sim.log").open("w") as sl, (folder/"deploy.log").open("w") as dl:
            processes.append(subprocess.Popen(sim, cwd=gt.ROOT, env=env, stdin=subprocess.DEVNULL, stdout=sl, stderr=subprocess.STDOUT))
            time.sleep(.5)
            if processes[0].poll() is not None:
                raise RuntimeError("MuJoCo startup failed")
            processes.append(subprocess.Popen(deploy, cwd=gt.ROOT, env=env, stdin=subprocess.DEVNULL, stdout=dl, stderr=subprocess.STDOUT))
            processes[1].wait(timeout=result["expected_policy_seconds"]+65)
            processes[0].wait(timeout=10)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Nonzero simulation exit")
        logs = (folder/"deploy.log").read_text()
        if "Traceback" in logs or "An exception occurred" in logs:
            raise RuntimeError("Controller exception")
        for marker in ("Playing 'cli_motion' from start", "Returning to default pose", "reached --max-policy-steps="):
            if marker not in logs:
                raise RuntimeError(f"Incomplete lifecycle: {marker}")
        metrics = json.loads((folder/"metrics.json").read_text())
        frames = result["expected_motion_frames"]
        result.update(metrics)
        if (not metrics.get("tracking_motion_frames_complete") or metrics.get("tracking_recorded_frames") != frames
                or metrics.get("tracking_missing_state_snapshots") != 0
                or metrics.get("tracking_sim_dt_max_abs_error_s", math.inf) > 1e-6):
            raise RuntimeError("Incomplete or misaligned tracking")
        seconds = result["expected_policy_seconds"]
        if not .95*seconds <= metrics["simulated_control_seconds"] <= 1.1*seconds:
            raise RuntimeError("Control/physics clock mismatch")
        skipped = sum(int(n) for n in re.findall(r"skipped (\d+) bridge state packet", logs.split("Running high level...")[-1]))
        result["skipped_policy_state_packets"] = skipped
        if skipped > max(2, seconds*50*.01):
            raise RuntimeError("Excess policy state packet loss")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for process in reversed(processes):
            stop_child(process)
    result["rating"], result["hardware_screen"], result["reasons"] = classify(result)
    result["evaluation_dir"] = str(folder)
    def finite(v):
        if isinstance(v, dict): return {k: finite(x) for k, x in v.items()}
        if isinstance(v, list): return [finite(x) for x in v]
        return None if isinstance(v, float) and not math.isfinite(v) else v
    result = finite(result)
    atomic_json(folder/"result.json", result)
    atomic_json(Path(result["attempt_dir"])/"latest_evaluation.json", result)
    print(f"Slow GT {result['gt_id']}: {result['rating']} / {result['hardware_screen']}; "
          f"Global={result.get('global_mpjpe_mm')} Root={result.get('root_relative_mpjpe_mm')}; "
          f"{result['reasons']}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--indices", default=",".join(gt.SELECTED))
    args = parser.parse_args()
    for mid in args.indices.split(","):
        evaluate(int(mid)) if args.evaluate else prepare(int(mid))
