"""Run a generated motion in isolated MuJoCo and render its actual trajectory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv/bin/python")
SCRIPT = str(Path(__file__).resolve())
XML = ROOT / "config/g1/assets/g1.xml"


def capture(path, argv):
    sys.path.insert(0, str(ROOT / "src"))
    import sim2sim

    class RecordingSim(sim2sim.Sim2Sim):
        def __init__(self, *args, **kwargs):
            self.video_rows = []
            self.video_reference = {}
            self.video_next_time = 0.0
            super().__init__(*args, **kwargs)

        def _read_command_snapshot(self):
            snapshot = super()._read_command_snapshot()
            self.video_reference = snapshot[3] or {}
            return snapshot

        def _viewer_sync(self):
            t = float(self.data.time)
            if t + 1e-9 >= self.video_next_time:
                self.video_rows.append((t, self.data.qpos.copy(), self.data.qvel.copy(),
                    str(self.video_reference.get("evaluation_phase", "startup")),
                    int(self.video_reference.get("motion_frame", -1))))
                self.video_next_time = t + .02
            return super()._viewer_sync()

        def _write_summary(self):
            if self._summary_written:
                return
            super()._write_summary()
            if self.video_rows:
                np.savez_compressed(path,
                    time=np.array([r[0] for r in self.video_rows]),
                    qpos=np.stack([r[1] for r in self.video_rows]),
                    qvel=np.stack([r[2] for r in self.video_rows]),
                    phase=np.array([r[3] for r in self.video_rows]),
                    motion_frame=np.array([r[4] for r in self.video_rows]))

    sim2sim.Sim2Sim = RecordingSim
    sim2sim.main(argv)


def render(folder, front=False, side=False):
    import mujoco
    with np.load(folder / "actual_trajectory.npz", allow_pickle=False) as z:
        times, qpos, qvel = z["time"], z["qpos"], z["qvel"]
        phases = z["phase"]
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise RuntimeError("Invalid recorded simulation timestamps")
    model = mujoco.MjModel.from_xml_path(str(XML))
    width, height = (1280, 720) if side else (960, 720)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth, camera.elevation = 35.0, -18.0
    if front or side:
        w, x, y, z = qpos[0, 3:7]
        yaw = np.degrees(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
        # MuJoCo's free-camera azimuth points toward lookat, hence +180
        # places the camera in front of the robot's initial forward axis.
        camera.azimuth, camera.elevation = (yaw + 180) % 360, -10.0
        if side:
            camera.azimuth = (yaw + 90) % 360
            camera.elevation = -float(np.degrees(np.arctan2(.25, 6)))
            model.vis.global_.orthographic = 1
    camera.distance = max(3.3, float(np.ptp(qpos[:, :2], axis=0).max()) + 2.7)
    camera.lookat[:] = [*np.median(qpos[:, :2], axis=0), .85]
    if side:
        # In orthographic mode fovy is a world-space vertical extent, not degrees.
        model.vis.global_.fovy = max(2.2, camera.distance * height / width)
        # With a nearly level orthographic view, move the eye back so the lower
        # image rays start above the floor rather than underneath it.
        camera.distance = 40.0
    fps = 30
    sample_times = np.arange(times[0], times[-1] + 1e-9, 1 / fps)
    indices = np.searchsorted(times, sample_times, side="right") - 1
    # The index is supplied through metadata, rather than inferred from paths.
    meta = json.loads((folder / "recording.json").read_text())
    suffix = "_side" if side else "_front" if front else ""
    basename = meta.get("video_basename", f"sample_{meta['index']:04d}_mujoco")
    target = folder / f"{basename}{suffix}.mp4"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    with (folder / f"ffmpeg{suffix}.log").open("w") as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            with mujoco.Renderer(model, height=height, width=width) as renderer:
                for n, i in enumerate(indices):
                    data.qpos[:], data.qvel[:] = qpos[i], qvel[i]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    encoder.stdin.write(renderer.render().tobytes())
                    if n % 150 == 0:
                        print(f"Rendering {n}/{len(indices)}", flush=True)
            encoder.stdin.close()
            if encoder.wait(timeout=60):
                raise RuntimeError("Video encoding failed; see ffmpeg.log")
        finally:
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
    meta.update(video=str(target), video_fps=fps, video_frames=len(indices),
                video_duration_s=len(indices)/fps, width=width, height=height,
                camera_view="side" if side else "front" if front else "oblique",
                camera_azimuth=float(camera.azimuth), camera_elevation=float(camera.elevation),
                recording="actual closed-loop MuJoCo qpos; offline rendering at simulation speed")
    action_times = times[phases == "motion"]
    if len(action_times):
        meta["motion_time_range_s"] = [float(action_times[0]-times[0]),
                                       float(action_times[-1]-times[0]+.02)]
    (folder / f"recording{suffix}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(target, flush=True)


def run(index, *, commands=None, output_parent=None, extra_metadata=None,
        front=False, render_video=True, require_complete=False):
    from run_generated import build_commands, stop_child
    result, sim, deploy = build_commands(index, False) if commands is None else commands
    sim, deploy = list(sim), list(deploy)
    parent = Path(output_parent) if output_parent is not None else ROOT.parent / "mujoco_recordings"
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f"sample_{index:04d}_", dir=parent))
    # Pick unused loopback ports and preserve every other simulation setting.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as a, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as b:
        a.bind(("127.0.0.1", 0)); b.bind(("127.0.0.1", 0))
        ports = a.getsockname()[1], b.getsockname()[1]
    for cmd, flag, name in ((sim, "--bridge-config", "bridge"),
                            (deploy, "--controller-config", "controller")):
        at = cmd.index(flag) + 1
        cfg = yaml.safe_load(Path(cmd[at]).read_text())
        cfg["udp"].update(state_port=ports[0], cmd_port=ports[1])
        if name == "bridge":
            cfg["udp"].update(state_host="127.0.0.1", cmd_bind_host="127.0.0.1")
        else:
            cfg["udp"].update(state_bind_host="127.0.0.1", cmd_host="127.0.0.1")
        path = folder / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        cmd[at] = str(path)
    sim[1:2] = [SCRIPT, "--capture", str(folder / "actual_trajectory.npz")]
    sim += ["--headless", "--metrics-out", str(folder / "metrics.json"),
            "--tracking-out", str(folder / "tracking.npz")]
    meta = dict(index=index, prompt=result.get("text"), source_motion=result["npz_file"],
                commands=dict(sim=sim, deploy=deploy))
    meta.update(extra_metadata or {})
    (folder / "recording.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Recording run: {folder}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    children = []
    try:
        with (folder / "sim.log").open("w") as sl, (folder / "deploy.log").open("w") as dl:
            children.append(subprocess.Popen(sim, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=sl, stderr=subprocess.STDOUT))
            time.sleep(1)
            if children[0].poll() is not None:
                raise RuntimeError(f"Simulator exited; see {folder / 'sim.log'}")
            children.append(subprocess.Popen(deploy, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=dl, stderr=subprocess.STDOUT))
            deadline = time.monotonic() + float(sim[sim.index("--max-control-seconds")+1]) + 75
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None, 0) for c in children):
                    raise RuntimeError(f"Simulation subprocess failed; logs: {folder}")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Simulation timed out; logs: {folder}")
                time.sleep(.2)
        log = (folder / "deploy.log").read_text()
        if "Traceback" in log or "An exception occurred" in log:
            raise RuntimeError(f"Controller exception; logs: {folder}")
        for marker in ("Playing 'cli_motion' from start", "Returning to default pose", "reached --max-policy-steps="):
            if marker not in log:
                raise RuntimeError(f"Incomplete playback: {marker}")
        metrics = json.loads((folder / "metrics.json").read_text())
        print(f"Motion tracking frames: {metrics.get('tracking_recorded_frames')}/{metrics.get('expected_motion_frames')}", flush=True)
        meta["tracking_complete"] = metrics.get("tracking_motion_frames_complete", False)
        (folder / "recording.json").write_text(json.dumps(meta, indent=2) + "\n")
        if require_complete and (
                not meta["tracking_complete"] or metrics.get("tracking_missing_state_snapshots") != 0
                or metrics.get("tracking_sim_dt_max_abs_error_s", float("inf")) > 1e-6):
            raise RuntimeError(f"Incomplete or misaligned tracking; recording retained in {folder}")
    finally:
        for child in reversed(children):
            stop_child(child)
    if render_video:
        subprocess.run([PYTHON, SCRIPT, "--render-front" if front else "--render", str(folder)], cwd=ROOT,
                       env=dict(env, MUJOCO_GL="egl", LP_NUM_THREADS="2"), check=True, timeout=600)
    return folder


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capture":
        capture(Path(sys.argv[2]), sys.argv[3:])
    elif len(sys.argv) > 1 and sys.argv[1] == "--render":
        render(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--render-front":
        render(Path(sys.argv[2]), front=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--render-side":
        render(Path(sys.argv[2]), side=True)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("index", type=int)
        run(parser.parse_args().index)
