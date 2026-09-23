"""Re-render the existing GRIT trajectory with a matte light MuJoCo floor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "motion_examples_100_qpos"
RECORD = ROOT.parent / "mujoco_recordings/sample_0012_nbqh58ci"
XML = ROOT / "config/g1/assets/g1.xml"
PREFIX = "12_005600_side_grit_light_floor"
WIDTH, HEIGHT, FPS = 2560, 1440, 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--requested-stills-8k", action="store_true")
    args = parser.parse_args()
    source = RECORD / "actual_trajectory.npz"
    with np.load(source, allow_pickle=False) as z:
        times, qpos, qvel = z["time"], z["qpos"], z["qvel"]
    tree = ET.parse(XML).getroot()
    compiler = tree.find("compiler")
    compiler.set("meshdir", str((XML.parent / compiler.get("meshdir")).resolve()))
    # Change only render assets, in memory; preserve shared deployment XML.
    asset = tree.find("asset")
    for tex in asset.findall("texture"):
        if tex.get("type") == "skybox":
            tex.set("builtin", "flat")
            tex.set("rgb1", "0.94 0.95 0.96")
            tex.set("rgb2", "0.94 0.95 0.96")
    material = asset.find("material[@name='groundplane']")
    material.attrib.pop("texture", None)
    material.attrib.pop("texrepeat", None)
    material.attrib.pop("texuniform", None)
    # Existing multi-light rig brightens this to a light neutral floor;
    # a near-white albedo would clip to pure white and hide the surface.
    material.set("rgba", "0.45 0.47 0.49 1")
    material.set("reflectance", "0")
    material.set("specular", "0")
    material.set("shininess", "0")
    # Near-horizontal orthographic views amplify shadow-map floor acne.
    # Keep the original shadow-free light rig for a clean pose-only view.
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    model.vis.global_.orthographic = 1
    model.vis.quality.offsamples = 4
    model.vis.quality.shadowsize = 4096
    model.vis.map.fogstart, model.vis.map.fogend = 10000, 20000
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    original_meta = json.loads((RECORD / "recording_side.json").read_text())
    camera.azimuth = original_meta["camera_azimuth"]
    camera.elevation = original_meta["camera_elevation"]
    original_distance = max(3.3, float(np.ptp(qpos[:, :2],axis=0).max())+2.7)
    model.vis.global_.fovy = max(2.2, original_distance*HEIGHT/WIDTH)
    camera.distance = 40
    camera.lookat[:] = [*np.median(qpos[:, :2],axis=0), .85]
    sample_times = np.arange(times[0], times[-1]+1e-9, 1/FPS)
    indices = np.searchsorted(times, sample_times, side="right")-1
    assert len(indices) == original_meta["video_frames"] == 515
    if args.requested_stills_8k:
        from render_example12_requested_8k import render_stills
        render_stills(model, data, camera, qpos, qvel, times, sample_times, indices)
        return
    full = BASE / f"{PREFIX}_full.mp4"
    action = BASE / f"{PREFIX}.mp4"
    if not args.preview_only:
        if full.exists() or action.exists(): raise FileExistsError("New video output already exists")
        encoder = subprocess.Popen(["ffmpeg","-nostdin","-v","error","-n","-f","rawvideo","-pix_fmt","rgb24",
            "-s",f"{WIDTH}x{HEIGHT}","-r",str(FPS),"-i","pipe:0","-an","-c:v","libx264",
            "-threads","2","-preset","medium","-crf","16","-pix_fmt","yuv420p","-movflags","+faststart",str(full)],stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
            frame_numbers = (150,210,286) if args.preview_only else range(len(indices))
            for n in frame_numbers:
                i = indices[n]
                data.qpos[:], data.qvel[:] = qpos[i],qvel[i]
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
                rgb = renderer.render()
                if args.preview_only:
                    Image.fromarray(rgb).save(BASE / f"{PREFIX}_preview_{n:03d}.png")
                else:
                    encoder.stdin.write(rgb.tobytes())
                if args.preview_only or n%60==0: print(f"Rendered {n+1}/{len(indices)}",flush=True)
        if args.preview_only: return
        encoder.stdin.close()
        if encoder.wait(timeout=60): raise RuntimeError("Video encoding failed")
    finally:
        if not args.preview_only and encoder.poll() is None:
            encoder.kill(); encoder.wait()
    # Preserve the same five-second action offset and 186-frame action clip.
    subprocess.run(["ffmpeg","-nostdin","-v","error","-n","-i",str(full),"-ss","5","-t","6.18",
        "-an","-c:v","libx264","-threads","2","-preset","medium","-crf","16","-pix_fmt","yuv420p",
        "-movflags","+faststart",str(action)],check=True)
    videos=[]
    for path, expected in ((full,515),(action,186)):
        probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-count_frames",
            "-show_entries","stream=width,height,avg_frame_rate,nb_read_frames,duration","-of","json",str(path)]))["streams"][0]
        assert (probe["width"],probe["height"]) == (WIDTH,HEIGHT)
        assert int(probe["nb_read_frames"]) == expected and probe["avg_frame_rate"] == "30/1"
        subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(path),"-f","null","-"],check=True)
        videos.append(dict(path=str(path),**probe))
    meta=dict(source_trajectory=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_model=str(XML),model_sha256=hashlib.sha256(XML.read_bytes()).hexdigest(),
        source_motion=str(BASE/"12_005600.npy"),videos=videos,antialiasing="4x MSAA",crf=16,
        appearance="Matte light grey floor, no checker texture or floor reflections; light skybox, haze/fog disabled, original shadow-free light rig",
        camera=dict(azimuth=camera.azimuth,elevation=camera.elevation,lookat=camera.lookat.tolist(),
            distance=camera.distance,orthographic_extent=float(model.vis.global_.fovy)),
        execution="Offline MuJoCo replay of original actual qpos/qvel; no new simulation, no pose edits; original camera/sample-time mapping preserved")
    full.with_suffix(".json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")
    print("READY: "+str(full),flush=True)


if __name__ == "__main__":
    main()
