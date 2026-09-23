"""Exact one-based action-video frame extraction with native-pixel closeups."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[2] / "motion_examples_100_qpos"
VIDEO = BASE / "12_005600_side_grit_light_floor.mp4"
OUT = BASE / "12_005600_light_floor_frames_1_21_66_76_107_162_147_183"
FRAMES = [1, 21, 66, 76, 107, 162, 147, 183]


def main():
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,nb_frames,avg_frame_rate","-of","json",str(VIDEO)]))["streams"][0]
    assert probe["avg_frame_rate"] == "30/1" and int(probe["nb_frames"]) == 186
    width,height=probe["width"],probe["height"]
    OUT.mkdir(exist_ok=False)
    for kind in ("full_frames","centered"): (OUT/kind).mkdir()
    rows=[]
    bounds=[]
    for order,number in enumerate(FRAMES,1):
        name=f"{order:02d}_frame_{number:03d}_t{(number-1)/30:.3f}s.png"
        path=OUT/"full_frames"/name
        subprocess.run(["ffmpeg","-nostdin","-v","error","-n","-i",str(VIDEO),"-vf",
            f"select=eq(n\\,{number-1})","-frames:v","1",str(path)],check=True)
        with Image.open(path) as im:
            pixels=np.asarray(im)
            # Plain light floor/sky have no dark objects; dark robot surfaces
            # locate the body. Add a generous margin for bright extremities.
            mask=np.min(pixels[:,:,:3],axis=2)<165
            y,x=np.where(mask)
            assert len(x)>100
            box=(max(0,int(x.min())-40),max(0,int(y.min())-40),
                 min(width,int(x.max())+41),min(height,int(y.max())+41))
            bounds.append(box)
        rows.append(dict(order=order,action_frame_one_based=number,video_frame_zero_based=number-1,
            action_time_s=(number-1)/30,full_video_time_s=5+(number-1)/30,
            full_frame=str(path.relative_to(OUT)),centered=f"centered/{name}"))
    crop_size=int(np.ceil(max(max(b[2]-b[0],b[3]-b[1]) for b in bounds)/16))*16
    assert crop_size<=height
    cell=512; gap=12; header=58
    sheet=Image.new("RGB",(4*cell+5*gap,2*(cell+header)+3*gap),"#eef1f5")
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",25)
    for idx,(row,b) in enumerate(zip(rows,bounds)):
        left=max(0,min(width-crop_size,round((b[0]+b[2]-crop_size)/2)))
        top=max(0,min(height-crop_size,round((b[1]+b[3]-crop_size)/2)))
        assert left<=b[0] and top<=b[1] and left+crop_size>=b[2] and top+crop_size>=b[3]
        with Image.open(OUT/row["full_frame"]) as im:
            crop=im.crop((left,top,left+crop_size,top+crop_size))
            crop.save(OUT/row["centered"])
            x=gap+(idx%4)*(cell+gap); y=gap+(idx//4)*(cell+header+gap)
            draw.text((x+10,y+15),f"Frame {row['action_frame_one_based']:03d} | {row['action_time_s']:.3f} s",font=font,fill="#182c40")
            sheet.paste(crop.resize((cell,cell),Image.Resampling.LANCZOS),(x,y+header))
        row["crop_xyxy"]=[left,top,left+crop_size,top+crop_size]
    sheet.save(OUT/"overview_8frames.png")
    manifest=dict(source_video=str(VIDEO),source_sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),
        frame_numbering="User-specified one-based frames in 186-frame action-only video; preserves requested order, including 162 before 147",
        fps=30,full_frame_resolution=[width,height],centered_resolution=[crop_size,crop_size],
        operation="Exact video-frame decoding to PNG; centered images are native-pixel crops, no pose editing or AI enhancement; fixed crop size for all eight",
        frames=rows)
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    (OUT/"README_ZH.md").write_text(
        "# 指定动作帧截图\n\n来自浅色地板 GRIT 动作阶段视频，按第 1 帧开始计数；对应解码索引为帧号减 1。"
        "顺序保留为 1、21、66、76、107、162、147、183（没有交换 162 和 147）。\n\n"
        "full_frames/ 是 2560×1440 完整画面；centered/ 是全身居中原像素裁切，没有放大插值；overview_8frames.png 为缩略总览。"
        "只裁切取景，不修改姿态。同一裁切尺寸用于所有帧，弯腰姿态不会额外放大。完整过程视频对应时间需加 5 秒。\n",encoding="utf-8")
    with zipfile.ZipFile(OUT.with_suffix('.zip'),"x",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(BASE))
    with zipfile.ZipFile(OUT.with_suffix('.zip')) as z: assert z.testzip() is None
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
    print("READY: "+str(OUT),flush=True)


if __name__ == "__main__":
    main()
