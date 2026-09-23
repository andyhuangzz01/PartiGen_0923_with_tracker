"""Render 23-DOF qpos with NumPy FK and PyRender, without a physics engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import pyrender
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "description/robots/g1/g1_23dof_lock_wrist.xml"


def vector(element, key, default):
    return np.fromstring(element.get(key, default), sep=" ")


def transform(pos, quat):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
    matrix[:3, 3] = pos
    return matrix


def pose(element):
    return transform(vector(element, "pos", "0 0 0"), vector(element, "quat", "1 0 0 0"))


def look_at(eye, target):
    back = np.asarray(eye) - target
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((right, up, back))
    matrix[:3, 3] = eye
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--keyframes-out", type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    target = source.with_name(source.stem + "_side_raw.mp4")
    if target.exists() and args.keyframes_out is None:
        raise FileExistsError(target)
    qpos = np.load(source, allow_pickle=False)
    if qpos.ndim != 2 or qpos.shape[1] != 30 or not np.isfinite(qpos).all():
        raise ValueError("Expected finite (T,30) qpos with wxyz root quaternion")
    root = ET.parse(MODEL).getroot()
    root_body = root.find("worldbody/body")
    meshes = {m.get("name"): MODEL.parent / root.find("compiler").get("meshdir") / m.get("file")
              for m in root.findall("asset/mesh")}
    bodies, joints = [], []

    def visit(body, parent):
        idx = len(bodies)
        body_joints = []
        for joint in body.findall("joint"):
            if joint.get("type", "hinge") == "free":
                continue
            if joint.get("type", "hinge") != "hinge":
                raise ValueError("Only hinge joints supported")
            axis = vector(joint, "axis", "0 0 1")
            axis /= np.linalg.norm(axis)
            body_joints.append((len(joints), axis, vector(joint, "pos", "0 0 0")))
            joints.append(joint.get("name"))
        bodies.append((body, parent, pose(body), body_joints))
        for child in body.findall("body"):
            visit(child, idx)

    visit(root_body, -1)
    if len(joints) != 23:
        raise ValueError(f"Expected 23 model joints; got {len(joints)}")
    poses = np.zeros((len(qpos), len(bodies), 4, 4))
    for frame, row in enumerate(qpos):
        for i, (_, parent, origin, body_joints) in enumerate(bodies):
            matrix = transform(row[:3], row[3:7]) if parent < 0 else poses[frame, parent] @ origin
            for index, axis, pivot in body_joints:
                rotation = Rotation.from_rotvec(axis * row[7 + index]).as_matrix()
                hinge = np.eye(4)
                hinge[:3, :3] = rotation
                hinge[:3, 3] = pivot - rotation @ pivot
                matrix = matrix @ hinge
            poses[frame, i] = matrix

    scene = pyrender.Scene(bg_color=[.94, .96, .98, 1], ambient_light=[.38, .38, .38])
    nodes, corners = [], []
    for i, (body, _, _, _) in enumerate(bodies):
        for geom in body.findall("geom"):
            if geom.get("type") != "mesh" or geom.get("group") != "1":
                continue
            mesh = trimesh.load_mesh(meshes[geom.get("mesh")], process=False)
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=vector(geom, "rgba", ".7 .7 .7 1"),
                metallicFactor=.12, roughnessFactor=.65)
            node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
            offset = pose(geom)
            nodes.append((node, i, offset))
            box = np.column_stack((trimesh.bounds.corners(mesh.bounds), np.ones(8)))
            world = np.einsum("tij,kj->tki", poses[:, i] @ offset, box)[..., :3]
            corners.append(world.reshape(-1, 3))
    points = np.concatenate(corners)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    root_rotation = Rotation.from_quat(np.roll(qpos[0, 3:7], -1)).as_matrix()
    forward = root_rotation[:, 0].copy()
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    side = np.array([forward[1], -forward[0], 0.0])
    camera_pose = look_at(center + side * 6 + np.array([0, 0, .25]), center)
    local = (points - center) @ camera_pose[:3, :3]
    half_extents = np.max(np.abs(local[:, :2]), axis=0) * 1.16 + .10
    width, height = 1280, 720
    ymag = max(half_extents[1], half_extents[0] * height / width)
    xmag = ymag * width / height
    camera_node = scene.add(pyrender.OrthographicCamera(xmag=xmag, ymag=ymag, znear=.01, zfar=50), pose=camera_pose)
    ground_z = min(0.0, float(points[:, 2].min()) - .01)
    for gx in range(-6, 7):
        for gy in range(-6, 7):
            tile = trimesh.creation.box(extents=[.5, .5, .01])
            tile.apply_translation([center[0] + gx*.5, center[1] + gy*.5, ground_z - .005])
            color = [.77, .81, .85, 1] if (gx + gy) % 2 else [.88, .90, .93, 1]
            mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=color, roughnessFactor=1)
            scene.add(pyrender.Mesh.from_trimesh(tile, material=mat))
    for location, intensity in ((center + [3, -3, 5], 2.5), (center + [-3, 2, 4], 1.6)):
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                  pose=look_at(location, center))
    if args.keyframes_out is not None:
        from render_example12_keyframes_8k import render_raw
        render_raw(args.keyframes_out, scene, camera_node, camera_pose, nodes, poses, corners)
        return
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    renderer = pyrender.OffscreenRenderer(width, height)
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in range(len(qpos)):
            for node, body_index, offset in nodes:
                scene.set_pose(node, pose=poses[frame, body_index] @ offset)
            rgb, _ = renderer.render(scene)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            draw.text((24, 20), f"{source.stem} | Original qpos | SIDE | {args.fps:g} FPS", fill=(25, 35, 45))
            draw.text((24, 42), f"Frame {frame+1}/{len(qpos)}    Time {frame/args.fps:.2f} s", fill=(25, 35, 45))
            encoder.stdin.write(np.asarray(image).tobytes())
            if frame in (0, len(qpos)//2):
                image.save(target.with_name(f"{source.stem}_side_preview_{frame:03d}.png"))
            if frame % 30 == 0:
                print(f"Rendered {frame}/{len(qpos)}", flush=True)
        encoder.stdin.close()
        if encoder.wait(timeout=60):
            raise RuntimeError("Video encoding failed")
    finally:
        renderer.delete()
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    meta = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                video=str(target), model=str(MODEL), frames=len(qpos), fps=args.fps,
                duration_s=len(qpos)/args.fps, quaternion_order="wxyz", joint_names=joints,
                renderer="PyRender + NumPy/SciPy forward kinematics; no MuJoCo runtime or dynamics",
                camera="fixed right-side view relative to initial root heading",
                transformations="none; one video frame per source pose; original root translation retained")
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(target, flush=True)


if __name__ == "__main__":
    main()
