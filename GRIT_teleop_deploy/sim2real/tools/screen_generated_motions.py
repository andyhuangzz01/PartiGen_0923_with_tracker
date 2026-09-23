"""Screen trusted generated PKL motions with the actual GRIT policy in MuJoCo.

Simulation only: isolated loopback UDP, no DDS or hardware allowlist writes.
PKL/joblib can execute code: only use files from a trusted source.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import threading
import time

import joblib
import mujoco
import numpy as np
import yaml

from batch_validate_motions import ROOT, XML, stop_process
from generated_dataset import audit_directory, records_from_manifest, write_manifest
from qpos_to_grit_npz import ROBOTMDAR_XML
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from common.tracking_metrics import BODY_POINT_NAMES, ROOT_BODY_NAME, resolve_body_points

PYTHON = str(Path(sys.executable).absolute())
STOP = threading.Event()
# Conservative screening heuristics, NOT manufacturer limits or certification.
LIMITS = dict(root_z_min=0.65, root_tilt_max_deg=20.0,
              reference_joint_rmse_rad=0.15, reference_lower_body_rmse_rad=0.12,
              reference_joint_error_max_rad=0.6, source_joint_speed_max_rad_s=4.0,
              actual_joint_speed_max_rad_s=6.0, all_joint_ctrl_saturation_fraction=0.01,
              source_limit_excess_rad=0.01)
H2O_REFERENCE = {
    "url": "https://github.com/LeCAR-Lab/human2humanoid/blob/main/phc/phc/smpllib/smpl_eval.py",
    "file_commit": "43665822657d8c856658add6ae1db4e8521d1f1b",
    "sha256": "3c062082c52f9aa46a786c7d3ecc76351421dc9926cd1de4e59b5d10fdc6985d",
    "contract": "compute_metrics_lite: mpjpe_g before alignment; mpjpe_l after independently subtracting root position",
}


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(record, trim, order):
    """The generated file's qpos is xyz + XYZW + 23 joints, not MuJoCo qpos."""
    q = np.array(record["qpos"], dtype=np.float64, copy=True)
    fps = float(record["fps"])
    if q.ndim != 2 or q.shape[1] != 30 or len(q) - trim < 2:
        raise ValueError("expected (T,30), with at least two frames after trimming")
    if not np.isfinite(q).all() or not math.isfinite(fps) or not 0 < fps <= 1000:
        raise ValueError("nonfinite data or invalid fps")
    q = q[trim:].copy()
    norms = np.linalg.norm(q[:, 3:7], axis=1)
    if np.any(norms < 1e-6) or np.max(np.abs(norms - 1)) > 0.05:
        raise ValueError("invalid/non-unit root quaternion")
    q[:, 3:7] /= norms[:, None]
    if order == "xyzw":
        q[:, 3:7] = np.roll(q[:, 3:7], 1, axis=1)
    return q, fps


def classify(result):
    if result.get("error"):
        return "ERROR", "NOT_RECOMMENDED", [result["error"]]
    for name in LIMITS:
        value = result.get(name)
        if value is None or not np.isfinite(value):
            return "ERROR", "NOT_RECOMMENDED", [f"missing/nonfinite {name}"]
    z, tilt = result["root_z_min"], result["root_tilt_max_deg"]
    if z < 0.45 or tilt > 60:
        return "FALL", "NOT_RECOMMENDED", ["low pelvis / excessive tilt (conservative fall proxy)"]
    rating = "PASS" if z >= 0.60 and tilt <= 35 else "MARGINAL"
    reasons = []
    for name, limit in LIMITS.items():
        value = result[name]
        if (value < limit if name == "root_z_min" else value > limit):
            reasons.append(f"{name}={value:.4g}, threshold={limit}")
    if rating != "PASS":
        return rating, "NOT_RECOMMENDED", reasons
    if reasons:
        return rating, "REVIEW", reasons
    # Text is an additional veto, never evidence of safety. Chinese/English prompts supported.
    if re.search(r"jump|hop|kick|run\w*|crawl|kneel|sit\w*|lie|lying|roll|cartwheel|flip|蹲|跳|跑|踢|跪|坐|躺|爬|翻", result.get("text", ""), re.I):
        return rating, "REVIEW", ["high-risk motion description requires manual review"]
    moving = result["source_xy_path_m"] > 0.25 or result["source_leg_excursion_rad"] > 0.35
    return rating, "CANDIDATE_TIER2" if moving else "CANDIDATE_TIER1", ["simulation-only candidate; manual replay and hardware checks required"]


def write_report(out, results, total):
    rows = sorted(results.values(), key=lambda x: x["index"])
    counts = {key: sum(r["hardware_screen"] == key for r in rows) for key in
              ("CANDIDATE_TIER1", "CANDIDATE_TIER2", "REVIEW", "NOT_RECOMMENDED")}
    valid = [r for r in rows if not r.get("error") and r.get("global_mpjpe_mm") is not None]
    stable = [r for r in valid if r.get("root_z_min", -math.inf) >= 0.60 and r.get("root_tilt_max_deg", math.inf) <= 35.0]

    def tracking_stats(subset):
        if not subset:
            return None
        def one(field):
            values = np.asarray([r[field] for r in subset], dtype=np.float64)
            frames = np.asarray([r["tracking_recorded_frames"] for r in subset], dtype=np.float64)
            return {
                "motion_equal_mean_mm": float(values.mean()),
                "motion_median_mm": float(np.median(values)),
                "frame_weighted_mean_mm": float(np.average(values, weights=frames)),
                "motions": len(subset),
                "frames": int(frames.sum()),
            }
        return {field: one(field) for field in ("global_mpjpe_mm", "root_relative_mpjpe_mm")}

    tracking_statistics = {
        "all_valid": tracking_stats(valid),
        "stability_pass_subset": tracking_stats(stable),
    }
    atomic_json(out / "summary.json", dict(
        total=total, completed=len(rows), valid_tracking_runs=len(valid), errors=len(rows) - len(valid),
        stability_pass_valid_runs=len(stable), counts=counts, thresholds=LIMITS,
        tracking_statistics=tracking_statistics, h2o_reference=H2O_REFERENCE, motions=rows,
    ))
    fields = ["index", "src_idx", "rating", "hardware_screen", "root_z_min", "root_tilt_max_deg",
              "reference_joint_rmse_rad", "reference_lower_body_rmse_rad",
              "global_mpjpe_mm", "root_relative_mpjpe_mm", "global_frame_mpjpe_p95_mm",
              "global_frame_mpjpe_max_mm", "root_relative_frame_mpjpe_p95_mm",
              "root_relative_frame_mpjpe_max_mm", "tracking_recorded_frames",
              "expected_motion_frames", "motion_coverage_fraction",
              "actual_joint_speed_max_rad_s", "all_joint_ctrl_saturation_fraction", "text", "reasons", "attempt_dir"]
    with (out / "summary.csv.tmp").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.csv.tmp").replace(out / "summary.csv")
    lines = ["# 生成动作 MuJoCo 初筛报告", "",
             f"已完成 {len(rows)} / {total} 条；未完成动作不作任何推荐。", "",
             "仅仿真初筛，不是安全认证。不会修改真机白名单。CANDIDATE_TIER1：优先人工复核候选；",
             "CANDIDATE_TIER2：含较多下肢/位移，次级候选；REVIEW：证据不足，暂不直接上真机；",
             "NOT_RECOMMENDED：不推荐。即使 PASS，也可能跟踪误差过大，不能直接放行。", "",
             "完整入场、动作、回默认及尾部稳定阶段均计入指标。未完成、超时、缺少参考数据均为 ERROR。",
             "高度/倾角是保守跌倒代理指标，可能把有意低姿态动作排除。力矩饱和采用仿真模型限值，不代表硬件限值。", "",
             "MPJPE 只统计完整生成动作阶段；执行前插值、回默认和尾部观察不计入位置误差，但稳定性仍覆盖完整流程。", "",
             f"统计：`{counts}`", "", "## 位置跟踪主结果", "",
             "先对每条动作内的逐帧 MPJPE 求均值，再对动作等权平均。frame-weighted 另列，不能混称主结果。", "",
             "| 范围 | 有效动作数 | Global MPJPE↓ (mm) | 中位数 | Root-relative MPJPE↓ (mm) | 中位数 |", "|---|---:|---:|---:|---:|---:|"]
    for label, key in (("全部有效运行", "all_valid"), ("稳定性通过子集", "stability_pass_subset")):
        stats = tracking_statistics[key]
        if stats is None:
            lines.append(f"| {label} | 0 | — | — | — | — |")
        else:
            g, l = stats["global_mpjpe_mm"], stats["root_relative_mpjpe_mm"]
            lines.append(f"| {label} | {g['motions']} | {g['motion_equal_mean_mm']:.3f} | {g['motion_median_mm']:.3f} | {l['motion_equal_mean_mm']:.3f} | {l['motion_median_mm']:.3f} |")
    lines += ["", "| 范围 | Global 全帧加权均值 (mm) | Root-relative 全帧加权均值 (mm) |", "|---|---:|---:|"]
    for label, key in (("全部有效运行", "all_valid"), ("稳定性通过子集", "stability_pass_subset")):
        stats = tracking_statistics[key]
        if stats is None:
            lines.append(f"| {label} | — | — |")
        else:
            lines.append(f"| {label} | {stats['global_mpjpe_mm']['frame_weighted_mean_mm']:.3f} | {stats['root_relative_mpjpe_mm']['frame_weighted_mean_mm']:.3f} |")
    lines += ["", "阈值（经验初筛，非宇树官方限值；未给 MPJPE 发明通过阈值）：", "", "```json",
             json.dumps(LIMITS, indent=2), "```", ""]
    for group in counts:
        lines += [f"## {group}", "", " ".join(str(r["index"]) for r in rows if r["hardware_screen"] == group) or "无", ""]
    lines += ["## 明细", "", "编号为 PKL 列表的零基索引；src_idx 是原测试集索引，与旧 100 条编号无关。", "",
              "| index / src_idx | 仿真 | 真机初筛 | 原因 | 描述 |", "|---|---|---|---|---|"]
    for r in rows:
        clean = lambda s: str(s).replace("|", "/").replace("\n", " ")
        lines.append(f"| {r['index']} / {r.get('src_idx')} | {r['rating']} | {r['hardware_screen']} | {clean('; '.join(r['reasons']))} | {clean(r.get('text', ''))} |")
    (out / "report.md.tmp").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "report.md.tmp").replace(out / "report.md")


def run_one(index, record, args, tracking, model):
    attempt = args.output / f"{index:04d}" / f"attempt_{time.time_ns()}"
    attempt.mkdir(parents=True)
    result = dict(index=index, src_idx=int(record.get("src_idx", index)), text=str(record.get("text", "")),
                  attempt_dir=str(attempt), error=None)
    started = time.monotonic()
    processes = []
    try:
        if STOP.is_set():
            raise RuntimeError("batch interrupted")
        q, fps = extract(record, args.trim_start, args.quat_order)
        joints = q[:, 7:]
        ranges = model.jnt_range[1:]
        excess = np.maximum(ranges[:, 0] - joints, joints - ranges[:, 1])
        excess[:, ~model.jnt_limited[1:].astype(bool)] = 0
        result.update(source_frames=len(q), source_fps=fps, trimmed_frames=args.trim_start,
                      sample_id=int(record.get("sample_idx", index)),
                      source_paths=record.get("source_paths"), source_sha256=record.get("source_sha256"),
                      source_joint_speed_max_rad_s=float(np.max(np.abs(np.diff(joints, axis=0))) * fps),
                      source_limit_excess_rad=float(max(0, excess.max())),
                      source_xy_path_m=float(np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1).sum()),
                      source_leg_excursion_rad=float(np.ptp(joints[:, :12], axis=0).max()))
        qpath, motion = attempt / "qpos_wxyz.npy", attempt / "motion.npz"
        np.save(qpath, q)
        ref_fps = float(tracking["reference_fps"])
        with (attempt / "convert.log").open("w") as log:
            subprocess.run([PYTHON, str(ROOT / "tools/qpos_to_grit_npz.py"), "--input", str(qpath),
                            "--output", str(motion), "--src-fps", str(fps), "--dst-fps", str(ref_fps),
                            "--quat-order", "wxyz"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        with np.load(motion) as clip:
            motion_frames = len(clip["joint_pos"])
            duration = motion_frames / ref_fps
        first = float(tracking["motion_source"]["npz"].get("first_frame_transition_s", tracking["transition_steps"] / ref_fps))
        # Tail keep/context margin + full return interpolation + 3 seconds settling.
        seconds = first + duration + tracking["transition_steps"] / ref_fps + args.settle_seconds + 1.0
        controller = yaml.safe_load((ROOT / "config/g1/controller.yaml").read_text())
        bridge = yaml.safe_load((ROOT / "config/g1/bridge.yaml").read_text())
        state_port = args.port_base + 2 * index
        for port in (state_port, state_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind(("127.0.0.1", port))  # fail rather than share an active port
        bridge["udp"].update(state_host="127.0.0.1", cmd_bind_host="127.0.0.1", state_port=state_port, cmd_port=state_port + 1)
        controller["udp"].update(state_bind_host="127.0.0.1", cmd_host="127.0.0.1", state_port=state_port, cmd_port=state_port + 1)
        for name, cfg in (("bridge", bridge), ("controller", controller)):
            (attempt / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
        steps = math.ceil(seconds * float(controller["control_freq"]))
        result.update(expected_policy_seconds=seconds, npz_file=str(motion))
        # Deploy controls termination. Simulator upper bound includes margin so it
        # cannot quit early and leave deploy waiting for state during tail evaluation.
        sim_cmd = [PYTHON, str(ROOT / "src/sim2sim.py"), "--robot", "g1", "--headless", "--auto-start",
                   "--bridge-config", str(attempt / "bridge.yaml"), "--xml_path", str(XML),
                   "--max-control-seconds", str(seconds + 10), "--metrics-out", str(attempt / "metrics.json"),
                   "--tracking-out", str(attempt / "tracking.npz")]
        deploy_cmd = [PYTHON, str(ROOT / "src/deploy.py"), "--robot", "g1", "--auto-start",
                      "--controller-config", str(attempt / "controller.yaml"),
                      "--tracking-config", str(ROOT / "config/g1/tracking.yaml"),
                      "--policy-path", str(args.policy), "--motion-file", str(motion),
                      "--publish-reference", "--max-policy-steps", str(steps)]
        atomic_json(attempt / "commands.json", dict(sim=sim_cmd, deploy=deploy_cmd))
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (attempt / "sim.log").open("w") as slog, (attempt / "deploy.log").open("w") as dlog:
            processes.append(subprocess.Popen(sim_cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=slog, stderr=subprocess.STDOUT))
            time.sleep(0.5)
            if processes[0].poll() is not None:
                raise RuntimeError("simulator failed to start; see sim.log")
            processes.append(subprocess.Popen(deploy_cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=dlog, stderr=subprocess.STDOUT))
            deadline = time.monotonic() + seconds + float(controller.get("startup_interpolation_s", 5)) + 60
            while any(p.poll() is None for p in processes):
                if STOP.is_set():
                    raise RuntimeError("batch interrupted")
                if time.monotonic() > deadline:
                    raise RuntimeError("simulation/deploy timeout")
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError("subprocess failed; see logs")
                time.sleep(0.1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("subprocess failed; see logs")
        logs = (attempt / "deploy.log").read_text()
        simlogs = (attempt / "sim.log").read_text()
        if "Traceback" in logs + simlogs or "An exception occurred" in logs:
            raise RuntimeError("exception in subprocess log")
        for marker in ("Playing 'cli_motion' from start", "Returning to default pose", f"reached --max-policy-steps={steps}"):
            if marker not in logs:
                raise RuntimeError(f"incomplete playback lifecycle: missing {marker}")
        metrics = json.loads((attempt / "metrics.json").read_text())
        result.update(metrics)
        if not (metrics.get("reference_samples", 0) > 0 and metrics.get("physics_samples", 0) > 0):
            raise RuntimeError("no reference/physics samples")
        if not metrics.get("tracking_motion_frames_complete", False):
            raise RuntimeError(
                "incomplete action-phase tracking frames: "
                f"recorded={metrics.get('tracking_recorded_frames')}, "
                f"expected={metrics.get('expected_motion_frames')}"
            )
        if metrics.get("expected_motion_frames") != motion_frames:
            raise RuntimeError("tracking/reference motion frame count mismatch")
        if metrics.get("tracking_sim_dt_max_abs_error_s", math.inf) > 1e-6:
            raise RuntimeError("tracking state/reference clock drift or duplicated timestamp")
        simulated = metrics["simulated_control_seconds"]
        if not 0.95 * seconds <= simulated <= 1.1 * seconds:
            raise RuntimeError(f"physics/control clock mismatch: simulated={simulated:.2f}, expected={seconds:.2f}")
        skipped = sum(int(n) for n in re.findall(r"skipped (\d+) bridge state packet", logs.split("Running high level...")[-1]))
        result["skipped_policy_state_packets"] = skipped
        if skipped > max(2, steps * 0.01):
            raise RuntimeError("excess state packet loss; reduce parallelism and retest")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for p in reversed(processes):
            stop_process(p)
    result["wall_seconds"] = time.monotonic() - started
    result["rating"], result["hardware_screen"], result["reasons"] = classify(result)
    # Sanitize failed simulation NaNs without allowing them to become candidates.
    def finite(value):
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        return None if isinstance(value, float) and not math.isfinite(value) else value
    result = finite(result)
    atomic_json(args.output / f"{index:04d}" / "result.json", result)
    print(f"[{index:04d}] {result['rating']} / {result['hardware_screen']} ({result['wall_seconds']:.1f}s)", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT.parent / "gen_all2576_dar0908_512.pkl",
                        help="trusted legacy PKL or unpacked sample_XXXX directory")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "gen_all2576_screen")
    parser.add_argument("--policy", type=Path, default=ROOT / "checkpoints/policy.onnx")
    parser.add_argument("--indices", default="", help="zero-based PKL list indices, e.g. 0,1,96")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--port-base", type=int, default=56000)
    parser.add_argument("--trim-start", type=int, default=0, help="explicitly discard this many source frames; default keeps all")
    parser.add_argument("--quat-order", choices=("xyzw", "wxyz"), default="xyzw")
    parser.add_argument("--settle-seconds", type=float, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--replay-index", type=int, help="only print two manual SIM replay commands for a completed index")
    args = parser.parse_args()
    if not 1 <= args.parallel <= 8 or args.trim_start < 0 or not math.isfinite(args.settle_seconds) or args.settle_seconds < 3:
        parser.error("parallel must be 1..8, trim >= 0, settle >= 3")
    args.input, args.output, args.policy = args.input.resolve(), args.output.resolve(), args.policy.resolve()
    if args.replay_index is not None:
        result = json.loads((args.output / f"{args.replay_index:04d}" / "result.json").read_text())
        commands = json.loads((Path(result["attempt_dir"]) / "commands.json").read_text())
        for name, command in commands.items():
            filtered = []
            it = iter(command)
            for word in it:
                if word in ("--headless", "--auto-start"):
                    continue
                if word in ("--max-control-seconds", "--metrics-out", "--max-policy-steps"):
                    next(it)
                    continue
                filtered.append(word)
            print(f"# Terminal: {name} (SIMULATION ONLY)\ncd {shlex.quote(str(ROOT))}\n{shlex.join(filtered)}\n")
        print("Wait for both terminals. In deploy: s = default pose, a = play, x = damping/exit. These commands are NOT for hardware.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, parser)


def run(args, parser):
    manifest = None
    if args.input.is_dir():
        manifest, manifest_rows = audit_directory(args.input)
        audit = manifest["summary"]
        if audit["invalid_samples"] or not audit["contiguous_sample_ids_from_zero"]:
            raise ValueError(f"input directory audit failed: {audit}")
        write_manifest(args.output, manifest)
        records = records_from_manifest(manifest_rows)
    else:
        records = joblib.load(args.input)
    if not isinstance(records, list) or not records:
        raise ValueError("expected a nonempty list of generated motion records")
    indices = sorted({int(v) for v in args.indices.split(",")}) if args.indices else list(range(len(records)))
    if not indices or indices[0] < 0 or indices[-1] >= len(records):
        parser.error("indices outside PKL list")
    if args.port_base < 56000 or args.port_base + 2 * indices[-1] + 1 > 65535:
        parser.error("isolated ports must stay within 56000..65535")
    source_files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "config/g1").glob("*.yaml"))
    source_files += sorted((ROOT / "config/g1/assets").rglob("*.xml"))
    source_files += [Path(__file__), ROOT / "tools/generated_dataset.py", ROOT / "tools/qpos_to_grit_npz.py",
                     ROOT / "tools/batch_validate_motions.py", ROBOTMDAR_XML.resolve(), args.policy]
    input_signature = (
        {"input_manifest_sha256": digest(args.output / "input_manifest.json"),
         "sample_count": len(records), "kind": "sample_directory"}
        if manifest is not None else
        {"input_sha256": digest(args.input), "kind": "trusted_joblib"}
    )
    signature = dict(input=str(args.input), **input_signature, trim_start=args.trim_start,
                     quat_order=args.quat_order, settle_seconds=args.settle_seconds,
                     runtime_versions={p: importlib.metadata.version(p) for p in ("numpy", "mujoco", "onnxruntime", "scipy", "joblib")},
                     files={str(p): digest(p) for p in source_files}, thresholds=LIMITS,
                     h2o_reference=H2O_REFERENCE,
                     metric_scope="all resampled generated frames only; excludes transition-in/return/settle")
    run_path = args.output / "run.json"
    if run_path.exists():
        if not args.resume:
            raise RuntimeError("output already has a run; use --resume or a new --output")
        if json.loads(run_path.read_text()) != signature:
            raise RuntimeError("source/model/config/script changed; use a NEW output directory to avoid stale recommendations")
    else:
        atomic_json(run_path, signature)
    results = {}
    for path in args.output.glob("[0-9]*/result.json"):
        r = json.loads(path.read_text())
        results[r["index"]] = r
    pending = [i for i in indices if i not in results or (args.retry_errors and results[i]["rating"] == "ERROR")]
    tracking = yaml.safe_load((ROOT / "config/g1/tracking.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(ROBOTMDAR_XML.resolve()))
    tracking_model = mujoco.MjModel.from_xml_path(str(XML.resolve()))
    body_ids, groups = resolve_body_points(tracking_model)
    atomic_json(args.output / "body_points.json", {
        "point_origin": "MjData.xpos body/link origin; not xipos centre of mass",
        "includes_root": True,
        "root_name": ROOT_BODY_NAME,
        "points": [
            {"order": i, "name": name, "mujoco_body_id": int(body_ids[i]),
             "reference_body_pos_w_index": int(body_ids[i] - 1)}
            for i, name in enumerate(BODY_POINT_NAMES)
        ],
        "groups": {name: [BODY_POINT_NAMES[i] for i in indices] for name, indices in groups.items()},
    })
    print(f"total={len(records)}, selected={len(indices)}, pending={len(pending)}, parallel={args.parallel}; SIMULATION ONLY", flush=True)
    write_report(args.output, results, len(records))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel)
    futures = {}
    try:
        # Bounded submission makes Ctrl+C cancel without queuing thousands of jobs.
        todo = iter(pending)
        for _ in range(args.parallel):
            i = next(todo, None)
            if i is not None:
                futures[pool.submit(run_one, i, records[i], args, tracking, model)] = i
        while futures:
            done, _ = concurrent.futures.wait(futures, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                r = future.result()
                results[r["index"]] = r
                write_report(args.output, results, len(records))
                i = next(todo, None)
                if i is not None:
                    futures[pool.submit(run_one, i, records[i], args, tracking, model)] = i
    except KeyboardInterrupt:
        print("Stopping simulator/controller pairs; completed results preserved for --resume.", flush=True)
        STOP.set()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for path in args.output.glob("[0-9]*/result.json"):
            r = json.loads(path.read_text())
            results[r["index"]] = r
        write_report(args.output, results, len(records))
    print(f"Report: {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
