"""Reproducible, simulation-only random non-mirrored GT benchmark."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import time
import tarfile

import joblib
import mujoco
import numpy as np
import yaml
import zstandard

from gt_arm_leg import validate_source, save_reference
from qpos_to_grit_npz import resample
from screen_generated_motions import atomic_json, classify, write_report
from batch_validate_motions import stop_process

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
ARCHIVE = BASE.parent / "HML3D-G1.tar.zst"
OUT = BASE / "gt_random500_seed20260918"
PYTHON = str(ROOT / ".venv/bin/python")
SEED = 20260918


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def archive_members():
    with ARCHIVE.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                yield archive, member


def prepare_sources():
    manifest_path = OUT / "selection.json"
    if manifest_path.exists():
        selected = json.loads(manifest_path.read_text())
        for item in selected["files"]:
            assert sha(OUT / item["path"]) == item["sha256"]
        return selected
    initial = ARCHIVE.stat()
    motions, texts = set(), set()
    for _, member in archive_members():
        p = Path(member.name)
        if not member.isfile() or not re.fullmatch(r"\d{6}", p.stem):
            continue
        if p.parent.name == "motion" and p.suffix == ".pkl": motions.add(p.stem)
        if p.parent.name == "texts" and p.suffix == ".txt": texts.add(p.stem)
    population = sorted(motions & texts)
    ids = random.Random(SEED).sample(population, 500)
    wanted = {f"HML3D-G1/{folder}/{mid}.{ext}": f"source/{folder}/{mid}.{ext}"
        for mid in ids for folder, ext in (("motion", "pkl"), ("texts", "txt"))}
    files = []
    for archive, member in archive_members():
        if member.name not in wanted: continue
        assert member.isfile() and member.size < 100_000_000
        content = archive.extractfile(member).read()
        assert len(content) == member.size
        path = OUT / wanted[member.name]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): assert path.read_bytes() == content
        else:
            with path.open("xb") as f: f.write(content)
        files.append(dict(path=wanted[member.name], archive_member=member.name,
            sha256=hashlib.sha256(content).hexdigest()))
    assert len(files) == 1000
    final = ARCHIVE.stat()
    assert (initial.st_size, initial.st_mtime_ns) == (final.st_size, final.st_mtime_ns)
    selected = dict(seed=SEED, population_count=len(population), population_ids=population,
        gt_ids_in_draw_order=ids, files=files, archive=str(ARCHIVE), archive_size=final.st_size,
        archive_mtime_ns=final.st_mtime_ns, sampling="random.sample without replacement over sorted non-M IDs with motion and text",
        split="whole archive, NOT verified val", paired_with_generated=False,
        exclusions="M-prefixed mirror IDs only; no quality filtering, no replacing failed samples")
    atomic_json(manifest_path, selected)
    print(f"Selected 500 / {len(population)} non-M GT motions; seed={SEED}", flush=True)
    return selected


def finite(v):
    if isinstance(v, dict): return {k: finite(x) for k, x in v.items()}
    if isinstance(v, list): return [finite(x) for x in v]
    return None if isinstance(v, float) and not math.isfinite(v) else v


def evaluate(index, mid, model, tracking):
    folder = OUT / f"{index:04d}"
    folder.mkdir(exist_ok=True)
    attempt = folder / f"attempt_{time.time_ns()}"
    attempt.mkdir()
    result = dict(index=index, gt_id=mid, src_idx=None, error=None, attempt_dir=str(attempt),
        preserved_dofs=29, trimmed_frames=0, dataset="random non-M GT, unpaired")
    started = time.monotonic()
    processes = []
    try:
        source = OUT / f"source/motion/{mid}.pkl"
        result["source_sha256"] = sha(source)
        text = (OUT / f"source/texts/{mid}.txt").read_text()
        result["text"] = text.splitlines()[0].split("#")[0]
        pos, quat, joints, fps, fk_error = validate_source(joblib.load(source), model)
        ref_fps = float(tracking["reference_fps"])
        _, rp, rq, rj = resample(np.arange(len(pos))/fps, pos, quat, joints, ref_fps)
        motion = attempt / "motion.npz"
        save_reference(motion, model, rp, rq, rj, ref_fps)
        excess = np.maximum(model.jnt_range[1:, 0]-joints, joints-model.jnt_range[1:, 1])
        excess[:, ~model.jnt_limited[1:].astype(bool)] = 0
        result.update(source_frames=len(pos), source_fps=fps, source_fk_max_error_m=fk_error,
            expected_motion_frames=len(rp), npz_file=str(motion),
            source_joint_speed_max_rad_s=float(np.abs(np.diff(joints, axis=0)).max()*fps),
            source_limit_excess_rad=float(max(0, excess.max())),
            source_xy_path_m=float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum()),
            source_leg_excursion_rad=float(np.ptp(joints[:, :12], axis=0).max()))
        controller, bridge = [yaml.safe_load((ROOT / f"config/g1/{n}.yaml").read_text()) for n in ("controller", "bridge")]
        port = 62000 + 2*index
        for candidate in [port, *range(64000, 65000, 2)]:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as state_sock, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as cmd_sock:
                    state_sock.bind(("127.0.0.1", candidate))
                    cmd_sock.bind(("127.0.0.1", candidate+1))
                port = candidate
                break
            except OSError:
                continue
        else:
            raise RuntimeError("No free isolated loopback port pair")
        result["transport_ports"] = [port, port+1]
        controller["udp"].update(state_bind_host="127.0.0.1", cmd_host="127.0.0.1", state_port=port, cmd_port=port+1)
        bridge["udp"].update(state_host="127.0.0.1", cmd_bind_host="127.0.0.1", state_port=port, cmd_port=port+1)
        for name, cfg in (("controller", controller), ("bridge", bridge)):
            (attempt / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
        first = float(tracking["motion_source"]["npz"].get("first_frame_transition_s", tracking["transition_steps"]/ref_fps))
        seconds = first + len(rp)/ref_fps + tracking["transition_steps"]/ref_fps + 4
        steps = math.ceil(seconds*float(controller["control_freq"]))
        result["expected_policy_seconds"] = seconds
        sim = [PYTHON, str(ROOT / "src/sim2sim.py"), "--robot", "g1", "--headless", "--auto-start",
            "--bridge-config", str(attempt / "bridge.yaml"), "--xml_path", str(ROOT / "config/g1/assets/g1.xml"),
            "--max-control-seconds", str(seconds+10), "--metrics-out", str(attempt / "metrics.json"),
            "--tracking-out", str(attempt / "tracking.npz")]
        deploy = [PYTHON, str(ROOT / "src/deploy.py"), "--robot", "g1", "--auto-start",
            "--controller-config", str(attempt / "controller.yaml"), "--tracking-config", str(ROOT / "config/g1/tracking.yaml"),
            "--policy-path", str(ROOT / "checkpoints/policy.onnx"), "--motion-file", str(motion),
            "--publish-reference", "--max-policy-steps", str(steps)]
        atomic_json(attempt / "commands.json", dict(sim=sim, deploy=deploy))
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MUJOCO_GL="egl")
        with (attempt / "sim.log").open("w") as sl, (attempt / "deploy.log").open("w") as dl:
            processes.append(subprocess.Popen(sim, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=sl, stderr=subprocess.STDOUT))
            time.sleep(.5)
            if processes[0].poll() is not None: raise RuntimeError("simulator startup failed")
            processes.append(subprocess.Popen(deploy, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=dl, stderr=subprocess.STDOUT))
            deadline = time.monotonic()+seconds+float(controller.get("startup_interpolation_s", 5))+60
            while any(p.poll() is None for p in processes):
                if time.monotonic() > deadline: raise RuntimeError("simulation timeout")
                if any(p.poll() not in (None, 0) for p in processes): raise RuntimeError("subprocess failed")
                time.sleep(.1)
        logs = (attempt / "deploy.log").read_text()
        if "Traceback" in logs+(attempt / "sim.log").read_text() or "An exception occurred" in logs:
            raise RuntimeError("subprocess exception")
        for marker in ("Playing 'cli_motion' from start", "Returning to default pose", f"reached --max-policy-steps={steps}"):
            if marker not in logs: raise RuntimeError(f"incomplete lifecycle: {marker}")
        metrics = json.loads((attempt / "metrics.json").read_text())
        result.update(metrics)
        if (not metrics.get("tracking_motion_frames_complete") or metrics.get("tracking_recorded_frames") != len(rp)
            or metrics.get("tracking_missing_state_snapshots") != 0
            or metrics.get("tracking_sim_dt_max_abs_error_s", math.inf) > 1e-6):
            raise RuntimeError("incomplete/misaligned tracking")
        if not .95*seconds <= metrics["simulated_control_seconds"] <= 1.1*seconds:
            raise RuntimeError("physics/control clock mismatch")
        skipped = sum(int(n) for n in re.findall(r"skipped (\d+) bridge state packet", logs.split("Running high level...")[-1]))
        result["skipped_policy_state_packets"] = skipped
        if skipped > max(2, steps*.01): raise RuntimeError("excess state packet loss")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for p in reversed(processes): stop_process(p)
    result["rating"], screen, result["reasons"] = classify(result)
    result["pipeline_screen_only"] = screen
    result["hardware_screen"] = "SIM_ONLY"
    result["wall_seconds"] = time.monotonic()-started
    result = finite(result)
    atomic_json(folder / "result.json", result)
    print(f"[{index+1}/500] GT {mid}: {result['rating']} Global={result.get('global_mpjpe_mm')} Root={result.get('root_relative_mpjpe_mm')} error={result['error']} ({result['wall_seconds']:.1f}s)", flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    # Prevent duplicate background runners from sharing output/ports.
    import fcntl
    lock = (OUT / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = [Path(__file__), ROOT / "tools/gt_arm_leg.py", ROOT / "tools/screen_generated_motions.py",
        ROOT / "tools/qpos_to_grit_npz.py", ROOT / "checkpoints/policy.onnx"]
    paths += sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "config/g1").glob("*.yaml"))
    paths += sorted((ROOT / "config/g1/assets").rglob("*.xml"))
    signature = {str(p):sha(p) for p in paths}
    sigpath = OUT / "signature.json"
    if sigpath.exists():
        original_signature = json.loads(sigpath.read_text())
        revision_path = OUT / "signature_transport_resume.json"
        if revision_path.exists():
            assert json.loads(revision_path.read_text())["files"] == signature, "Changed resume code/config"
        elif original_signature != signature:
            changed = {k for k in original_signature.keys() | signature.keys()
                       if original_signature.get(k) != signature.get(k)}
            assert changed == {str(Path(__file__))}, "Metric/model/config changed; use new output"
            assert original_signature[str(Path(__file__))] == "851ed4ce71a2be133526b3693306a4c90bffa8fb72f7e0f1caf9e8787895a87a", "Unknown original runner"
            atomic_json(revision_path, dict(files=signature, original_signature_file=str(sigpath),
                changes="Loopback ports 60000->62000 with free-pair fallback in 64000..64999; retry ERROR once per invocation; preserve attempt results; consecutive-error counter reset. No metric/model/policy/reference changes."))
    else: atomic_json(sigpath, signature)
    atomic_json(OUT / "status.json", dict(state="extracting", pid=os.getpid(), seed=SEED))
    selected = prepare_sources()
    (OUT / "README_ZH.md").write_text("# 随机 GT 500 条闭环 MPJPE\n\n固定种子 20260918，从全库非镜像动作中无放回抽样；不是已确认 val，也不是生成组配对样本。"
        "保留 GT 原始 29DOF、全长、原速，30→50Hz 使用原有参考转换方式。生成组原本是 23DOF（腕部补零），两者不完全同分布。"
        "仅仿真，不修改任何真机名单。ERROR 单列；完整失稳运行仍计入 MPJPE。主均值为动作等权，位置误差只统计动作阶段。\n\n"
        "selection.json 为抽样与来源校验；summary.json/csv 为逐条与汇总；status.json 为实时进度；runner.log 为日志。\n", encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "config/g1/assets/g1.xml"))
    tracking = yaml.safe_load((ROOT / "config/g1/tracking.yaml").read_text())
    results = {int(p.parent.name): json.loads(p.read_text()) for p in OUT.glob("[0-9]*/result.json")}
    write_report(OUT, results, 500)
    consecutive_errors = 0
    for index, mid in enumerate(selected["gt_ids_in_draw_order"]):
        if index in results and not results[index].get("error"): continue
        if index in results:
            previous = results[index]
            archive_result = Path(previous["attempt_dir"]) / "result.json"
            if not archive_result.exists(): atomic_json(archive_result, previous)
            print(f"Retry transport/error sample {mid}: {previous.get('error')}", flush=True)
        atomic_json(OUT / "status.json", dict(state="running", pid=os.getpid(), completed=len(results), total=500, current_gt=mid))
        results[index] = evaluate(index, mid, model, tracking)
        atomic_json(Path(results[index]["attempt_dir"]) / "result.json", results[index])
        write_report(OUT, results, 500)
        # Stop on systemic initial failures rather than producing 500 invalid runs.
        consecutive_errors = consecutive_errors + 1 if results[index].get("error") else 0
        if consecutive_errors >= 3:
            atomic_json(OUT / "status.json", dict(state="stopped_consecutive_errors", completed=len(results), total=500))
            raise RuntimeError("Three consecutive errors; inspect logs before resuming")
    atomic_json(OUT / "status.json", dict(state="complete", completed=len(results), total=500))
    print("COMPLETE: " + str(OUT / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
