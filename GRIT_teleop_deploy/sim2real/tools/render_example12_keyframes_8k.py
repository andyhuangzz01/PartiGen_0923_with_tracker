"""Native tiled 8K, lossless pose screenshots; never modifies or simulates motion."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[2] / "motion_examples_100_qpos"
RECORD = BASE.parent / "mujoco_recordings/sample_0012_nbqh58ci"
OUT = BASE / "12_005600_keyframes_8k"
SIZE, TILE, EXTENT = 8192, 4096, 1.5
FRAMES = [0, 15, 60, 78, 136, 184]


def filename(order, frame):
    return f"{order:02d}_frame{frame:03d}_t{frame/30:.3f}s.png"


def offsets():
    for row in range(2):
        for col in range(2):
            yield row, col, (col-.5)*EXTENT/2, (.5-row)*EXTENT/2


def centered(points, rotation):
    projected = points @ rotation
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    assert np.max((hi-lo)[:2]) < EXTENT*.95, (lo, hi)
    return ((lo+hi)/2) @ rotation.T


def render_raw(output, scene, camera_node, camera_pose, nodes, poses, corners):
    import pyrender
    output.mkdir(parents=True, exist_ok=False)
    rotation = camera_pose[:3, :3]
    renderer = pyrender.OffscreenRenderer(TILE, TILE)
    rows = []
    try:
        for order, frame in enumerate(FRAMES, 1):
            points = np.concatenate([c.reshape(len(poses), -1, 3)[frame] for c in corners])
            center = centered(points, rotation)
            for node, body, offset in nodes:
                scene.set_pose(node, pose=poses[frame, body] @ offset)
            result = Image.new("RGB", (SIZE, SIZE))
            camera_node.camera.xmag = EXTENT/4
            camera_node.camera.ymag = EXTENT/4
            for row, col, x, y in offsets():
                cp = camera_pose.copy()
                cp[:3, 3] = center + rotation[:, 2]*40 + rotation[:, 0]*x + rotation[:, 1]*y
                scene.set_pose(camera_node, pose=cp)
                rgb, _ = renderer.render(scene)
                result.paste(Image.fromarray(rgb), (col*TILE, row*TILE))
            result.save(output / filename(order, frame), compress_level=6)
            rows.append(dict(frame=frame, time_s=frame/30, camera_center=center.tolist()))
            print(f"Original native 8K: {order}/6", flush=True)
    finally:
        renderer.delete()
    (output / "render.json").write_text(json.dumps(dict(frames=rows, extent_m=EXTENT,
        renderer="NumPy/SciPy FK + PyRender, no physics; native 4096 tiles, 4x MSAA"), indent=2))


def render_grit():
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    output = OUT / "grit"
    output.mkdir(parents=True, exist_ok=False)
    with np.load(RECORD / "actual_trajectory.npz", allow_pickle=False) as archive:
        times, qpos, qvel = archive["time"], archive["qpos"], archive["qvel"]
    # Reproduce record_generated_sim.render's original arange/searchsorted mapping
    # exactly, including the 150-frame (5 s) trim of the action-only video.
    grid = np.arange(times[0], times[-1]+1e-9, 1/30)
    indices = np.searchsorted(times, grid, side="right")-1
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / "config/g1/assets/g1.xml"))
    model.vis.global_.offwidth = model.vis.global_.offheight = TILE
    model.vis.global_.orthographic = 1
    model.vis.global_.fovy = EXTENT/2
    model.vis.quality.offsamples = 4
    model.vis.quality.shadowsize = 4096
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    metadata = json.loads((RECORD / "recording_side.json").read_text())
    camera.azimuth = metadata["camera_azimuth"]
    camera.elevation = metadata["camera_elevation"]
    camera.distance = 40
    a, e = np.radians([camera.azimuth, camera.elevation])
    back = np.array([-np.cos(a)*np.cos(e), -np.sin(a)*np.cos(e), -np.sin(e)])
    right = np.array([np.sin(a), -np.cos(a), 0])
    up = np.cross(back, right)
    rotation = np.column_stack([right, up, back])
    rows = []
    with mujoco.Renderer(model, height=TILE, width=TILE) as renderer:
        for order, frame in enumerate(FRAMES, 1):
            index = int(indices[frame+150])
            data.qpos[:], data.qvel[:] = qpos[index], qvel[index]
            mujoco.mj_forward(model, data)
            points = []
            for g in range(model.ngeom):
                if model.geom_group[g] != 2 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                m = model.geom_dataid[g]
                start, count = model.mesh_vertadr[m], model.mesh_vertnum[m]
                vertices = model.mesh_vert[start:start+count]
                points.append(vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
            center = centered(np.concatenate(points), rotation)
            result = Image.new("RGB", (SIZE, SIZE))
            for row, col, x, y in offsets():
                camera.lookat[:] = center + right*x + up*y
                renderer.update_scene(data, camera=camera)
                result.paste(Image.fromarray(renderer.render()), (col*TILE, row*TILE))
            result.save(output / filename(order, frame), compress_level=6)
            rows.append(dict(frame=frame, time_s=frame/30, trajectory_index=index,
                trajectory_time_s=float(times[index]), target_simulation_time_s=float(grid[frame+150]),
                camera_center=center.tolist()))
            print(f"GRIT native 8K: {order}/6", flush=True)
    (output / "render.json").write_text(json.dumps(dict(frames=rows, extent_m=EXTENT,
        renderer="MuJoCo recorded trajectory, no new dynamics; native 4096 tiles, 4x MSAA"), indent=2))


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def package():
    (OUT / "pairs").mkdir(exist_ok=False)
    overview = Image.new("RGB", (6*1024, 2*1100), "#eef1f5")
    rows = []
    for order, frame in enumerate(FRAMES, 1):
        pair = Image.new("RGB", (SIZE*2, SIZE+240), "#eef1f5")
        row = dict(order=order, frame=frame, time_s=frame/30)
        for r, (kind, label) in enumerate((("raw", "Original"), ("grit", "GRIT"))):
            path = OUT / kind / filename(order, frame)
            with Image.open(path) as picture:
                picture.load()
                assert picture.size == (SIZE, SIZE)
                pair.paste(picture, (r*SIZE, 240))
                overview.paste(picture.resize((1024, 1024), Image.Resampling.LANCZOS), ((order-1)*1024, r*1100+76))
            ImageDraw.Draw(pair).text((r*SIZE+90, 55), f"{label} | {frame/30:.3f} s", font=font(115), fill="#182c40")
            ImageDraw.Draw(overview).text(((order-1)*1024+20, r*1100+18), f"{label} | {frame/30:.3f} s", font=font(36), fill="#182c40")
            row[kind] = str(path.relative_to(OUT))
        pair.save(OUT / "pairs" / f"pair_{order:02d}_8k.png", compress_level=6)
        rows.append(row)
        print(f"Packaged pair: {order}/6", flush=True)
    overview.save(OUT / "comparison_6frames_8k_preview.png")
    sources = [BASE / "12_005600.npy", RECORD / "actual_trajectory.npz"]
    manifest = dict(resolution=[SIZE,SIZE], format="lossless PNG", antialiasing="4x MSAA",
        rendering="Four native 4096x4096 orthographic tiles per image; no upscaling or AI enhancement",
        view="Same right-side directions as original videos; centered full body, fixed common 1.5 m view extent",
        alignment="Exact original frame indices; GRIT uses original 30 FPS sample grid and 150-frame trim",
        pose_changes="None; no new physics simulation", keyframes=rows,
        sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (OUT / "README_ZH.md").write_text(
        "# 原生 8K 关键帧\n\n每张 8192×8192 无损 PNG，4× MSAA 抗锯齿。"
        "从原始 qpos / 已记录的 GRIT 仿真轨迹重新渲染，不是旧图放大，不使用 AI 补画。\n\n"
        "原来的六个时间点不变；统一 1.5m 正交取景范围，侧面、全身居中。只改变相机，不改姿态。"
        "GRIT 按原视频的时间采样规则取已有轨迹，不重跑仿真。\n\n"
        "raw/ 与 grit/ 各六张 8K 单图；pairs/ 是 16384×8432 的原尺寸左右对照。"
        "comparison_6frames_8k_preview.png 为缩小总览，不是最高分辨率原图。\n", encoding="utf-8")
    archive = OUT.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(OUT.rglob("*")):
            if path.is_file(): bundle.write(path, path.relative_to(OUT.parent))
    with zipfile.ZipFile(archive) as bundle: assert bundle.testzip() is None
    print(f"READY: {OUT}\nZIP: {archive}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["grit", "package"])
    args = parser.parse_args()
    render_grit() if args.mode == "grit" else package()
