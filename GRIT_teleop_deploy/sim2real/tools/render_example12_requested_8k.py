"""Requested light-floor native 8K square stills from unchanged actual poses."""
import hashlib
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[2] / "motion_examples_100_qpos"
OUT = BASE / "12_005600_light_floor_requested_8k"
FRAMES = [1,21,66,76,107,162,147,183]
SIZE, TILE, EXTENT = 8192, 4096, 1.5


def render_stills(model, data, camera, qpos, qvel, times, sample_times, indices):
    OUT.mkdir(exist_ok=False)
    model.vis.global_.offwidth = model.vis.global_.offheight = TILE
    model.vis.global_.fovy = EXTENT/2
    model.vis.quality.offsamples = 4
    a,e=np.radians([camera.azimuth,camera.elevation])
    back=np.array([-np.cos(a)*np.cos(e),-np.sin(a)*np.cos(e),-np.sin(e)])
    right=np.array([np.sin(a),-np.cos(a),0])
    up=np.cross(back,right)
    rotation=np.column_stack([right,up,back])
    rows=[]
    with mujoco.Renderer(model,height=TILE,width=TILE) as renderer:
        for order,number in enumerate(FRAMES,1):
            video_frame=number-1
            i=int(indices[150+video_frame])
            data.qpos[:],data.qvel[:]=qpos[i],qvel[i]
            mujoco.mj_forward(model,data)
            points=[]
            for g in range(model.ngeom):
                if model.geom_group[g]!=2 or model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
                m=model.geom_dataid[g]; start=model.mesh_vertadr[m]; count=model.mesh_vertnum[m]
                points.append(model.mesh_vert[start:start+count]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])
            projected=np.concatenate(points)@rotation
            lo,hi=projected.min(axis=0),projected.max(axis=0)
            assert max((hi-lo)[:2]) < EXTENT*.95, "Body would be cropped"
            center=((lo+hi)/2)@rotation.T
            picture=Image.new("RGB",(SIZE,SIZE))
            for r in range(2):
                for c in range(2):
                    camera.lookat[:]=center+right*(c-.5)*EXTENT/2+up*(.5-r)*EXTENT/2
                    renderer.update_scene(data,camera=camera)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE]=False
                    picture.paste(Image.fromarray(renderer.render()),(c*TILE,r*TILE))
            filename=f"{order:02d}_frame_{number:03d}_8k.png"
            picture.save(OUT/filename,compress_level=6)
            rows.append(dict(order=order,action_frame_one_based=number,action_video_frame_zero_based=video_frame,
                action_time_s=video_frame/30,full_video_time_s=5+video_frame/30,trajectory_index=i,
                trajectory_time_s=float(times[i]),video_sample_time_s=float(sample_times[150+video_frame]),
                camera_center=center.tolist(),file=filename))
            print(f"Native 8K square: {order}/8 (action frame {number})",flush=True)
    sheet=Image.new("RGB",(4*768,2*828),"#eef1f5")
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",30)
    for idx,row in enumerate(rows):
        x=(idx%4)*768; y=(idx//4)*828
        draw.text((x+20,y+16),f"Frame {row['action_frame_one_based']:03d} | {row['action_time_s']:.3f} s",font=font,fill="#182c40")
        with Image.open(OUT/row['file']) as im:
            assert im.size == (8192,8192)
            sheet.paste(im.resize((768,768),Image.Resampling.LANCZOS),(x,y+60))
    sheet.save(OUT/"overview_8frames.png")
    source=BASE.parent/"mujoco_recordings/sample_0012_nbqh58ci/actual_trajectory.npz"
    metadata=dict(source_trajectory=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        requested_order=FRAMES,numbering="One-based action video frames, 30 FPS; full video offset is 150 frames / 5 seconds",
        resolution=[SIZE,SIZE],aspect_ratio="1:1",antialiasing="4x MSAA",format="lossless PNG",
        rendering="Native 4096-pixel tiles assembled into 8192 square; no enlargement of old images",
        camera="same right-side orientation; full body centered per frame; fixed common 1.5m orthographic extent",
        background="MuJoCo matte light floor; no checker, reflection, fog or shadow artifacts",
        pose_changes="None; original actual GRIT trajectory, no new dynamics simulation",frames=rows)
    (OUT/"manifest.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    (OUT/"README_ZH.md").write_text(
        "# 指定 8 帧：1:1 原生 8K\n\n每张 8192×8192 无损 PNG，4× MSAA 抗锯齿；从已记录的 GRIT 仿真轨迹原生渲染，不是放大旧视频截图。"
        "浅色地板、侧面、全身居中，头手脚不裁切。所有帧统一 1.5m 正交取景范围，弯腰不会额外放大。\n\n"
        "编号以 186 帧动作阶段视频的第 1 帧开始；顺序：1、21、66、76、107、162、147、183。"
        "未交换 162 和 147。manifest.json 保存对应原轨迹索引与时间。\n\n"
        "overview_8frames.png 仅为缩略总览，最高画质请打开各张 *_8k.png。原视频和旧截图不修改。\n",encoding="utf-8")
    with zipfile.ZipFile(OUT.with_suffix('.zip'),"x",compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(OUT.iterdir()):archive.write(path,Path(OUT.name)/path.name)
    with zipfile.ZipFile(OUT.with_suffix('.zip')) as archive: assert archive.testzip() is None
    print("READY: "+str(OUT),flush=True)
