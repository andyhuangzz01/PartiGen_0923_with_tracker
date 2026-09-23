"""Record examples100 manual/candidate-list motions with a front MuJoCo camera."""
from __future__ import annotations

import concurrent.futures
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import yaml

from record_generated_sim import ROOT, PYTHON, SCRIPT, XML, run

MANUAL = ROOT.parent / "MOTION_EXAMPLES_100_PIPELINE_ZH.md"
MOTIONS = ROOT / "config/g1/motions/examples100"
CANDIDATES = MOTIONS / "hardware_candidates.txt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entries():
    manual = {int(i): tier for i, tier in re.findall(
        r"^\|\s*(\d+)\s*\|\s*PASS\s*\|\s*(tier[12])\s*\|", MANUAL.read_text(), re.M)}
    current = {int(i): tier for tier, i in re.findall(
        r"^(tier[12])\s+(\d+)\b", CANDIDATES.read_text(), re.M)}
    for i in manual.keys() & current.keys():
        if manual[i] != current[i]:
            raise ValueError(f"Tier mismatch for {i}")
    selected = manual | current
    records = {r["index"]: r for r in json.loads(
        (ROOT.parent / "motion_examples_100_validation.json").read_text())["motions"]}
    rows = []
    for i, tier in sorted(selected.items(), key=lambda x: (x[1], x[0])):
        record = records[i]
        paths = list(MOTIONS.glob(f"{i:02d}_*.npz"))
        if len(paths) != 1 or paths[0].name != record["motion"]:
            raise ValueError(f"Motion mapping mismatch for {i}")
        rows.append(dict(index=i, tier=tier, prompt=record["description"], source=record["source"],
                         motion=str(paths[0]), motion_sha256=sha(paths[0]),
                         selection_origin="manual_and_current_list" if i in manual and i in current else
                         "manual_only" if i in manual else "current_list_only",
                         status="pending", attempts=[]))
    return rows


def commands(row):
    tracking = yaml.safe_load((ROOT / "config/g1/tracking.yaml").read_text())
    controller = yaml.safe_load((ROOT / "config/g1/controller.yaml").read_text())
    if tracking["motion_source"]["npz"].get("loop", False):
        raise ValueError("Recording requires one-shot playback")
    with np.load(row["motion"], allow_pickle=False) as z:
        fps = float(z["fps"].reshape(-1)[0])
        frames = len(z["joint_pos"])
    seconds = (float(tracking["motion_source"]["npz"]["first_frame_transition_s"])
               + frames / fps + tracking["transition_steps"] / tracking["reference_fps"] + 4)
    steps = math.ceil(seconds * controller["control_freq"])
    result = dict(index=row["index"], text=row["prompt"], npz_file=row["motion"])
    sim = [PYTHON, str(ROOT / "src/sim2sim.py"), "--robot", "g1", "--auto-start",
           "--bridge-config", str(ROOT / "config/g1/bridge.yaml"), "--xml_path", str(XML),
           "--max-control-seconds", str(seconds + 10)]
    deploy = [PYTHON, str(ROOT / "src/deploy.py"), "--robot", "g1", "--auto-start",
              "--controller-config", str(ROOT / "config/g1/controller.yaml"),
              "--tracking-config", str(ROOT / "config/g1/tracking.yaml"),
              "--policy-path", str(ROOT / "checkpoints/policy.onnx"),
              "--motion-file", row["motion"], "--publish-reference", "--max-policy-steps", str(steps)]
    return result, sim, deploy


def write_index(out, rows):
    manifest = dict(dataset="motion_examples_100", manual=str(MANUAL),
                    manual_sha256=sha(MANUAL), candidates=str(CANDIDATES),
                    candidates_sha256=sha(CANDIDATES),
                    done=sum(r["status"] == "done" for r in rows), total=len(rows), motions=rows)
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(out / "manifest.json")
    cards = []
    for row in rows:
        title = f"{row['tier'].upper()} / {row['index']:02d} / {row['source']}"
        note = "（当前名单新增，未列在手册正文 Tier 表中）" if row["selection_origin"] == "current_list_only" else ""
        video = (f'<video controls preload="none" src="{html.escape(row["video_relative"])}"></video>'
                 if row["status"] == "done" else f'<p>{html.escape(row["status"])}</p>')
        cards.append(f'<article><h2>{html.escape(title)}</h2><p>{html.escape(row["prompt"])}</p>'
                     f'<p>{note}</p>{video}</article>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Examples100 MuJoCo 正面录像</title>'
            '<style>body{font:16px sans-serif;background:#f4f5f7;color:#222;margin:30px}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}'
            'article{padding:18px;background:white;border-radius:12px}video{width:100%;background:#111}'
            'h2{font-size:18px}</style>'
            f'<h1>motion_examples_100：MuJoCo 正面录像（{manifest["done"]}/{len(rows)}）</h1>'
            '<p>按实际仿真速度播放；包含首帧插值、动作及返回默认姿态。'
            'Tier 来自手册/当前名单，录像不会改变原有评级。</p><main>' + ''.join(cards) + '</main></html>')
    (out / "index.html").write_text(page, encoding="utf-8")


def finish_video(out, row, folder):
    env = dict(os.environ, MUJOCO_GL="egl", LP_NUM_THREADS="2", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (folder / "render.log").open("w") as log:
        subprocess.run([PYTHON, SCRIPT, "--render-front", str(folder)], cwd=ROOT,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    meta = json.loads((folder / "recording_front.json").read_text())
    source = Path(meta["video"])
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(source),
                    "-f", "null", "-"], check=True, timeout=90, stdout=subprocess.DEVNULL)
    destination = out / row["tier"] / source.name
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    metrics = json.loads((folder / "metrics.json").read_text())
    with np.load(folder / "actual_trajectory.npz", allow_pickle=False) as z:
        action = z["phase"] == "motion"
        motion_range = (z["time"][action][[0, -1]] - z["time"][0]).tolist()
    return dict(video_relative=str(destination.relative_to(out)), video=str(destination),
                video_sha256=sha(destination), duration_s=meta["video_duration_s"],
                video_frames=meta["video_frames"], motion_time_range_s=motion_range,
                recorded_motion_frames=metrics["tracking_recorded_frames"],
                expected_motion_frames=metrics["expected_motion_frames"],
                tracking_complete=metrics["tracking_motion_frames_complete"],
                recording_dir=str(folder))


def main():
    out = ROOT.parent / "motion_examples_100_front_videos" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    rows = entries()
    write_index(out, rows)
    print(f"OUTPUT={out}\nSelected {len(rows)} motions", flush=True)
    # One simulator pair at a time; one bounded render worker overlaps the next
    # recording. All UDP addresses are replaced with isolated loopback ports.
    pending = []
    def collect(wait=False):
        for future, row in list(pending):
            if not wait and not future.done():
                continue
            try:
                row.update(future.result())
                row["status"] = "done"
                print(f"VIDEO DONE {row['tier']} {row['index']:02d} ({row['duration_s']:.1f}s)", flush=True)
            except Exception as exc:
                row.update(status="error", error=str(exc))
                print(f"VIDEO ERROR {row['index']:02d}: {exc}", flush=True)
            pending.remove((future, row))
            write_index(out, rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for row in rows:
            collect()
            for attempt in range(1, 4):
                try:
                    row["status"] = "recording"
                    write_index(out, rows)
                    print(f"RECORD {row['tier']} {row['index']:02d} attempt={attempt}", flush=True)
                    folder = run(row["index"], commands=commands(row), output_parent=out / "runs",
                                 extra_metadata=dict(dataset="motion_examples_100", tier=row["tier"],
                                     video_basename=f"example100_{Path(row['motion']).stem}_mujoco",
                                     source_motion_sha256=row["motion_sha256"]),
                                 render_video=False, require_complete=True)
                    row["attempts"].append(dict(attempt=attempt, status="complete", folder=str(folder)))
                    row["status"] = "rendering"
                    pending.append((pool.submit(finish_video, out, dict(row), folder), row))
                    break
                except Exception as exc:
                    row["attempts"].append(dict(attempt=attempt, status="error", error=str(exc)))
                    row.update(status="error", error=str(exc))
                    print(f"RECORD ERROR {row['index']:02d}: {exc}", flush=True)
            write_index(out, rows)
        collect(wait=True)
    write_index(out, rows)
    failed = [r["index"] for r in rows if r["status"] != "done"]
    print(f"FINISHED {len(rows)-len(failed)}/{len(rows)}; failures={failed}; {out / 'index.html'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
