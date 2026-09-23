"""Bounded independent refinement experiments for GT 005225."""
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

import numpy as np
import joblib
import mujoco
import yaml
from scipy.spatial.transform import Rotation

import gt_slow
from generated_dataset import digest
from screen_generated_motions import atomic_json, classify

ROOT = gt_slow.gt.ROOT
DATA = ROOT.parent / "gt_005225_refinement"

SPECS = {
    "gentle90": dict(lower_scale=.9, tilt_scale=.8, wrist_pitch_scale=.6),
    "gentle80": dict(lower_scale=.8, tilt_scale=.65, wrist_pitch_scale=.6),
}


def transform(model, pos, quat, joints, default, spec):
    """Retain arm gesture and stepping; preserve minimum toe height per frame."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    out_j = joints.copy()
    lower = [j for j,n in enumerate(names) if any(s in n for s in ("hip_", "knee_", "ankle_", "waist_"))]
    out_j[:,lower] = default[lower] + spec["lower_scale"]*(joints[:,lower]-default[lower])
    out_j[:,names.index("left_wrist_pitch_joint")] *= spec["wrist_pitch_scale"]
    euler = Rotation.from_quat(np.roll(quat,-1,axis=1)).as_euler("xyz")
    euler[:,:2] *= spec["tilt_scale"]
    out_q = np.roll(Rotation.from_euler("xyz",euler).as_quat(),1,axis=1)
    out_p = pos.copy()
    out_p[:,:2] = pos[0,:2]+spec["lower_scale"]*(pos[:,:2]-pos[0,:2])
    toes = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ("left_toe_link","right_toe_link")]
    if min(toes)<0: raise ValueError("Missing toe bodies")
    data = mujoco.MjData(model)
    for i in range(len(pos)):
        data.qpos[:] = np.r_[pos[i],quat[i],joints[i]]; mujoco.mj_forward(model,data)
        old_z = data.xpos[toes,2].min()
        data.qpos[:] = np.r_[out_p[i],out_q[i],out_j[i]]; mujoco.mj_forward(model,data)
        out_p[i,2] += old_z-data.xpos[toes,2].min()
    correction = float(np.abs(out_p[:,2]-pos[:,2]).max())
    if correction > .05: raise ValueError(f"Root-height compensation exceeds 5 cm: {correction}")
    return out_p,out_q,out_j,correction


def prepare_variant(name):
    if name not in SPECS: raise ValueError(f"Unknown refinement {name}")
    base = gt_slow.prepare(5225)
    spec = SPECS[name]
    sig = dict(base["signature"])
    for file in (Path(__file__),ROOT/"tools/sim2sim_diagnostics.py"):
        sig[str(file)] = digest(file)
    fingerprint = hashlib.sha256(json.dumps(dict(signature=sig,spec=spec),sort_keys=True).encode()).hexdigest()
    folder = DATA/"variants"/f"{name}_{fingerprint[:12]}"
    metadata = folder/"prepared.json"
    if metadata.is_file():
        result = json.loads(metadata.read_text())
        if result["signature"] != sig or any(not Path(p).is_file() or digest(Path(p))!=sha for p,sha in result["artifacts"].items()):
            raise ValueError("Refinement artifacts changed")
        return result
    model = mujoco.MjModel.from_xml_path(str(gt_slow.gt.XML))
    raw = joblib.load(gt_slow.gt.DATA/"source/motion/005225.pkl")
    pos,quat,joints,fps,_ = gt_slow.gt.validate_source(raw,model)
    tracking = yaml.safe_load((ROOT/"config/g1/tracking.yaml").read_text())
    default = np.array(next(c for c in tracking["motion_clips"] if c["name"]=="default")["joint_pos"])
    pos,quat,joints,correction = transform(model,pos,quat,joints,default,spec)
    factor = base["slowdown_factor"]
    _,p,q,j = gt_slow.stretch(pos,quat,joints,fps,factor,50)
    folder.mkdir(parents=True,exist_ok=True)
    motion = folder/"motion.npz"
    if motion.exists(): raise ValueError("Incomplete existing refinement")
    gt_slow.gt.save_reference(motion,model,p,q,j,50)
    result = dict(base)
    port = 65210+2*list(SPECS).index(name)
    for config_name in ("bridge","controller"):
        cfg=yaml.safe_load((Path(base["attempt_dir"])/f"{config_name}.yaml").read_text())
        cfg["udp"].update(state_port=port,cmd_port=port+1)
        (folder/f"{config_name}.yaml").write_text(yaml.safe_dump(cfg))
    excess = np.maximum(model.jnt_range[1:,0]-joints,joints-model.jnt_range[1:,1])
    excess[:,~model.jnt_limited[1:].astype(bool)] = 0
    result.update(variant=name, adjustment=spec, signature=sig, fingerprint=fingerprint,
        attempt_dir=str(folder),npz_file=str(motion),max_root_height_compensation_m=correction,
        source_joint_speed_max_rad_s=float(np.abs(np.diff(joints,axis=0)).max()*fps/factor),
        source_limit_excess_rad=float(max(0,excess.max())),
        source_xy_path_m=float(np.linalg.norm(np.diff(pos[:,:2],axis=0),axis=1).sum()),
        source_leg_excursion_rad=float(np.ptp(joints[:,:12],axis=0).max()),
        rating="NOT_EVALUATED",hardware_screen="SIM_ONLY",error=None)
    result["artifacts"]={str(p):digest(p) for p in (motion,folder/"bridge.yaml",folder/"controller.yaml")}
    atomic_json(metadata,result)
    print(f"Prepared {name}: {spec}; max height compensation={correction:.4f}m",flush=True)
    return result


def evaluate(prepared, repeat=1):
    """Evaluate immutable prepared NPZ with original policy and all original gates."""
    from run_generated import stop_child
    result = dict(prepared)
    folder = Path(tempfile.mkdtemp(prefix=f"eval_{repeat}_", dir=prepared["attempt_dir"]))
    seconds = result["expected_policy_seconds"]
    sim = [gt_slow.gt.PYTHON, str(ROOT/"tools/sim2sim_diagnostics.py"), "--robot", "g1",
           "--bridge-config", str(Path(prepared["attempt_dir"])/"bridge.yaml"),
           "--xml_path", str(gt_slow.gt.XML), "--headless", "--auto-start",
           "--max-control-seconds", str(seconds+10), "--metrics-out", str(folder/"metrics.json"),
           "--tracking-out", str(folder/"tracking.npz")]
    steps = math.ceil(seconds*50)
    deploy = [gt_slow.gt.PYTHON, str(ROOT/"src/deploy.py"), "--robot", "g1",
              "--controller-config", str(Path(prepared["attempt_dir"])/"controller.yaml"),
              "--tracking-config", str(ROOT/"config/g1/tracking.yaml"),
              "--policy-path", str(ROOT/"checkpoints/policy.onnx"), "--motion-file", result["npz_file"],
              "--publish-reference", "--auto-start", "--max-policy-steps", str(steps)]
    atomic_json(folder/"commands.json", dict(sim=sim, deploy=deploy))
    children = []
    print(f"Evaluating {prepared['variant']} repeat={repeat}", flush=True)
    try:
        cfg = yaml.safe_load((Path(prepared["attempt_dir"])/"bridge.yaml").read_text())
        for port in (cfg["udp"]["state_port"], cfg["udp"]["cmd_port"]):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind(("127.0.0.1", port))
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (folder/"sim.log").open("w") as sl, (folder/"deploy.log").open("w") as dl:
            children.append(subprocess.Popen(sim, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=sl, stderr=subprocess.STDOUT))
            time.sleep(.5)
            if children[0].poll() is not None:
                raise RuntimeError("Simulator startup failed")
            children.append(subprocess.Popen(deploy, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=dl, stderr=subprocess.STDOUT))
            children[1].wait(timeout=seconds+65)
            children[0].wait(timeout=10)
        if any(p.returncode for p in children):
            raise RuntimeError("Simulation subprocess failed")
        logs = (folder/"deploy.log").read_text()
        if "An exception occurred" in logs or "Traceback" in logs:
            raise RuntimeError("Controller exception")
        for marker in ("Playing 'cli_motion' from start", "Returning to default pose", f"reached --max-policy-steps={steps}"):
            if marker not in logs: raise RuntimeError(f"Missing lifecycle marker: {marker}")
        metrics = json.loads((folder/"metrics.json").read_text())
        diagnostic = json.loads((folder/"diagnostics.json").read_text())
        if not diagnostic["matches_aggregate_peaks"]:
            raise RuntimeError("Diagnostic peaks do not match original metric implementation")
        expected = result["expected_motion_frames"]
        result.update(metrics)
        if (not metrics.get("tracking_motion_frames_complete") or metrics.get("tracking_recorded_frames") != expected
                or metrics.get("tracking_missing_state_snapshots") != 0
                or metrics.get("tracking_sim_dt_max_abs_error_s", math.inf) > 1e-6):
            raise RuntimeError("Incomplete or misaligned tracking")
        if not .95*seconds <= metrics["simulated_control_seconds"] <= 1.1*seconds:
            raise RuntimeError("Clock mismatch")
        skipped = sum(int(n) for n in re.findall(r"skipped (\d+) bridge state packet", logs.split("Running high level...")[-1]))
        result["skipped_policy_state_packets"] = skipped
        if skipped > max(2, steps*.01): raise RuntimeError("Excess policy state packet loss")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for p in reversed(children): stop_child(p)
    result["rating"], result["hardware_screen"], result["reasons"] = classify(result)
    result["evaluation_dir"] = str(folder)
    def finite(v):
        if isinstance(v,dict): return {k:finite(x) for k,x in v.items()}
        if isinstance(v,list): return [finite(x) for x in v]
        return None if isinstance(v,float) and not math.isfinite(v) else v
    result = finite(result)
    atomic_json(folder/"result.json", result)
    print(f"{result['variant']}: {result['rating']} / {result['hardware_screen']}: {result['reasons']}", flush=True)
    return result


def baseline():
    base = gt_slow.prepare(5225)
    folder = DATA/"baseline_diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("bridge", "controller"):
        config = yaml.safe_load((Path(base["attempt_dir"])/f"{name}.yaml").read_text())
        config["udp"].update(state_port=65200, cmd_port=65201)
        target = folder/f"{name}.yaml"
        if not target.exists(): target.write_text(yaml.safe_dump(config))
    base.update(variant="baseline_diagnostics", attempt_dir=str(folder), error=None)
    return base


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=["baseline",*SPECS],default="baseline")
    parser.add_argument("--repeats",type=int,default=1)
    args=parser.parse_args()
    if not 1<=args.repeats<=3: parser.error("repeats must be 1..3")
    prepared=baseline() if args.variant=="baseline" else prepare_variant(args.variant)
    for repeat in range(1,args.repeats+1):
        evaluate(prepared,repeat)
