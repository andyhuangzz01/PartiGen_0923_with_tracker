"""Extract matched source-frame indices from the original and GRIT videos."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[2] / "motion_examples_100_qpos"
OUTPUT = BASE / "12_005600_keyframes"
FRAMES = (0, 15, 60, 78, 136, 184)
STAGES = ("Initial pose", "Forward step", "Bend / reach down", "Rise", "Reach forward", "End pose")
VIDEOS = {"raw": BASE / "12_005600_side_raw.mp4", "grit": BASE / "12_005600_side_grit.mp4"}


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def main():
    OUTPUT.mkdir(exist_ok=False)
    rows = []
    selection = "+".join(f"eq(n,{n})" for n in FRAMES)
    for name, video in VIDEOS.items():
        info = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(video)]))
        stream = info["streams"][0]
        if stream["r_frame_rate"] != "30/1" or int(stream["nb_frames"]) <= max(FRAMES):
            raise ValueError(f"Unexpected video timing: {video}")
        dest = OUTPUT / name
        dest.mkdir()
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(video),
                        "-vf", f"select='{selection}'", "-vsync", "0", "-start_number", "1",
                        str(dest / "frame_%02d.png")], check=True)
        if len(list(dest.glob("frame_*.png"))) != len(FRAMES):
            raise RuntimeError("Not all selected frames were extracted")
        for order, (frame, stage) in enumerate(zip(FRAMES, STAGES), 1):
            target = dest / f"{order:02d}_frame{frame:03d}_t{frame/30:.3f}s.png"
            (dest / f"frame_{order:02d}.png").rename(target)
            with Image.open(target) as im:
                if im.size != (1280, 720):
                    raise ValueError("Unexpected screenshot dimensions")
            if name == "raw":
                rows.append(dict(order=order, source_frame_index_zero_based=frame,
                                 time_s=frame/30, stage=stage))
            rows[order-1][name] = str(target.relative_to(OUTPUT))

    # Keep individual screenshots untouched; only resize copies for the overview.
    cell_w, cell_h, label_h, gap = 640, 360, 56, 8
    sheet = Image.new("RGB", (6*cell_w + 7*gap, 2*cell_h + 2*label_h + 108), "#e8edf2")
    draw = ImageDraw.Draw(sheet)
    labels = {"raw": "Original sequence - direct rendering", "grit": "MuJoCo + GRIT tracking"}
    for row_index, name in enumerate(("raw", "grit")):
        y = 10 + row_index * (cell_h + label_h + 54)
        draw.text((gap, y), labels[name], font=font(28), fill="#172b40")
        for column, row in enumerate(rows):
            x = gap + column * (cell_w + gap)
            draw.text((x+8, y+41), f"{row['order']:02d}   {row['time_s']:.3f}s   {row['stage']}",
                      font=font(22), fill="#172b40")
            with Image.open(OUTPUT / row[name]) as im:
                sheet.paste(im.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS), (x, y+label_h+26))
    sheet.save(OUTPUT / "comparison_6frames.png")

    pairs = OUTPUT / "pairs"
    pairs.mkdir()
    for row in rows:
        pair = Image.new("RGB", (2560, 776), "#e8edf2")
        d = ImageDraw.Draw(pair)
        for col, name in enumerate(("raw", "grit")):
            d.text((col*1280+20, 12), f"{labels[name]}  |  t={row['time_s']:.3f}s",
                   font=font(27), fill="#172b40")
            with Image.open(OUTPUT / row[name]) as im:
                pair.paste(im.convert("RGB"), (col*1280, 56))
        pair_path = pairs / f"pair_{row['order']:02d}_t{row['time_s']:.3f}s.png"
        pair.save(pair_path)
        row["pair"] = str(pair_path.relative_to(OUTPUT))
    manifest = dict(fps=30, screenshots_per_video=6,
                    source_videos={k: dict(path=str(v), sha256=hashlib.sha256(v.read_bytes()).hexdigest())
                                   for k,v in VIDEOS.items()},
                    alignment="Same frame index in the two 30 FPS action-only videos; no time warping",
                    full_grit_video_time_offset_s=5.0, keyframes=rows)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    (OUTPUT / "README_ZH.md").write_text(
        "# 12_005600：六组对应关键帧\n\n"
        "- `comparison_6frames.png`：上排为原始序列直接渲染，下排为 MuJoCo + GRIT；每列对应同一时间。\n"
        "- `raw/`：6 张原始渲染视频截图，1280×720。\n"
        "- `grit/`：6 张 GRIT 追踪视频截图，1280×720。\n"
        "- `pairs/`：6 张左右对照图，左原始、右 GRIT。\n\n"
        "零基帧号：0、15、60、78、136、184；时间为 0、0.5、2、2.6、4.5333、6.1333 秒。\n"
        "时间以两个 6.2 秒动作段视频为准；若查阅完整 GRIT 视频，需要加 5 秒。\n"
        "单张截图直接来自视频解码，未裁切或修饰姿态；仅总览图缩小了图片尺寸。\n",
        encoding="utf-8")
    bundle = BASE / "12_005600_keyframes.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(BASE))
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
    print(f"READY: {OUTPUT}\nZIP: {bundle}")


if __name__ == "__main__":
    main()
