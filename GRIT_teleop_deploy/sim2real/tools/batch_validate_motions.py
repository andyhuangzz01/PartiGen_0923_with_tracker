"""Run local GRIT NPZ motions through isolated headless MuJoCo pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
# Keep the virtual-environment executable path intact.  Resolving its symlink
# selects the base uv interpreter and loses the venv's installed packages.
PYTHON = Path(sys.executable).absolute()
SIM = ROOT / "src/sim2sim.py"
DEPLOY = ROOT / "src/deploy.py"
BRIDGE_CONFIG = ROOT / "config/g1/bridge.yaml"
CONTROLLER_CONFIG = ROOT / "config/g1/controller.yaml"
XML = ROOT / "config/g1/assets/g1.xml"


def write_isolated_configs(output_dir: Path, index: int, port_base: int) -> tuple[Path, Path]:
    state_port = port_base + 2 * index
    cmd_port = state_port + 1
    bridge = yaml.safe_load(BRIDGE_CONFIG.read_text(encoding="utf-8"))
    controller = yaml.safe_load(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
    bridge["udp"]["state_port"] = state_port
    bridge["udp"]["cmd_port"] = cmd_port
    controller["udp"]["state_port"] = state_port
    controller["udp"]["cmd_port"] = cmd_port
    bridge_path = output_dir / f"bridge_{index:02d}.yaml"
    controller_path = output_dir / f"controller_{index:02d}.yaml"
    bridge_path.write_text(yaml.safe_dump(bridge, sort_keys=False), encoding="utf-8")
    controller_path.write_text(yaml.safe_dump(controller, sort_keys=False), encoding="utf-8")
    return bridge_path, controller_path


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def validate_one(task: tuple[int, Path, Path, Path, Path]) -> dict:
    index, motion_path, output_dir, bridge_path, controller_path = task
    with np.load(motion_path, allow_pickle=False) as data:
        frames = int(data["joint_pos"].shape[0])
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    duration_s = frames / fps
    # Two seconds into the clip and two seconds back toward default, plus a
    # small margin, exercise the complete one-shot playback lifecycle.
    control_seconds = duration_s + 4.5
    policy_steps = int(math.ceil(control_seconds * 50.0))
    metrics_path = output_dir / f"{motion_path.stem}.metrics.json"
    sim_log_path = output_dir / f"{motion_path.stem}.sim.log"
    deploy_log_path = output_dir / f"{motion_path.stem}.deploy.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    sim_command = [
        str(PYTHON), str(SIM), "--robot", "g1",
        "--bridge-config", str(bridge_path), "--xml_path", str(XML),
        "--headless", "--auto-start", "--max-control-seconds", str(control_seconds),
        "--metrics-out", str(metrics_path),
    ]
    deploy_command = [
        str(PYTHON), str(DEPLOY), "--robot", "g1",
        "--controller-config", str(controller_path), "--auto-start",
        "--motion-file", str(motion_path), "--publish-reference",
        "--max-policy-steps", str(policy_steps),
    ]

    sim_process = None
    deploy_process = None
    started = time.monotonic()
    error = None
    with sim_log_path.open("w", encoding="utf-8") as sim_log, deploy_log_path.open(
        "w", encoding="utf-8"
    ) as deploy_log:
        try:
            sim_process = subprocess.Popen(
                sim_command, cwd=ROOT, env=env, stdout=sim_log,
                stderr=subprocess.STDOUT, text=True,
            )
            time.sleep(0.5)
            deploy_process = subprocess.Popen(
                deploy_command, cwd=ROOT, env=env, stdout=deploy_log,
                stderr=subprocess.STDOUT, text=True,
            )
            deploy_process.wait(timeout=control_seconds + 25.0)
            sim_process.wait(timeout=10.0)
            if deploy_process.returncode != 0 or sim_process.returncode != 0:
                error = (
                    f"return codes: sim={sim_process.returncode}, "
                    f"deploy={deploy_process.returncode}"
                )
        except Exception as exc:
            error = str(exc)
        finally:
            if deploy_process is not None:
                stop_process(deploy_process)
            if sim_process is not None:
                stop_process(sim_process)

    result = {
        "index": index,
        "motion": motion_path.name,
        "frames": frames,
        "duration_s": duration_s,
        "wall_seconds": time.monotonic() - started,
        "error": error,
    }
    if metrics_path.is_file():
        result.update(json.loads(metrics_path.read_text(encoding="utf-8")))
        root_min = float(result["root_z_min"])
        tilt_max = float(result["root_tilt_max_deg"])
        if root_min < 0.45 or tilt_max > 60.0:
            result["rating"] = "FALL"
        elif root_min >= 0.60 and tilt_max <= 35.0:
            result["rating"] = "PASS"
        else:
            result["rating"] = "MARGINAL"
    else:
        result["rating"] = "ERROR"
    print(
        f"[{index:02d}] {motion_path.stem}: {result['rating']} "
        f"z_min={result.get('root_z_min')} "
        f"tilt={result.get('root_tilt_max_deg')} "
        f"rmse={result.get('reference_joint_rmse_rad')}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--port-base", type=int, default=56100)
    parser.add_argument(
        "--indices",
        default="",
        help="Optional comma-separated filename indices, for example 2,8,10",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motions = sorted(args.motion_dir.resolve().glob("*.npz"))
    if args.indices.strip():
        selected = {int(value) for value in args.indices.split(",")}
        motions = [
            path for path in motions
            if int(path.stem.split("_", 1)[0]) in selected
        ]
    if not motions:
        raise FileNotFoundError(f"no NPZ motions in {args.motion_dir}")

    tasks = []
    for index, motion_path in enumerate(motions):
        bridge_path, controller_path = write_isolated_configs(
            args.output_dir, index, args.port_base
        )
        tasks.append((index, motion_path, args.output_dir, bridge_path, controller_path))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(validate_one, tasks))
    results.sort(key=lambda item: item["index"])
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
