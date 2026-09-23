"""Re-extract the same video frames with reviewed whole-body close-up crops."""
from pathlib import Path
import json
import subprocess
import zipfile

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[2] / "motion_examples_100_qpos"
OUTPUT = BASE / "12_005600_keyframes_closeup"
# Reviewed pixel bounds include the head, both hands and both feet, excluding
# floor reflections. Crop sizes stay fixed within each video across all poses.
BOUNDS = {
    "raw": [(116,166,252,564), (238,165,397,567), (382,280,613,573),
            (445,168,587,567), (980,162,1155,560), (742,166,888,565)],
    "grit": [(202,248,311,578), (220,243,369,568), (343,340,546,575),
             (400,291,519,576), (797,285,946,574), (706,244,825,573)],
}
CROP_SIZE = {"raw": 448, "grit": 368}
VIDEOS = {"raw": BASE / "12_005600_side_raw.mp4", "grit": BASE / "12_005600_side_grit.mp4"}


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def main():
    source = json.loads((BASE / "12_005600_keyframes/manifest.json").read_text())
    rows = source["keyframes"]
    OUTPUT.mkdir(exist_ok=False)
    for name, video in VIDEOS.items():
        (OUTPUT / name).mkdir()
        size = CROP_SIZE[name]
        for row, bbox in zip(rows, BOUNDS[name]):
            left, top, right, bottom = bbox
            x = max(0, min(1280-size, round((left+right-size)/2)))
            y = max(0, min(720-size, round((top+bottom-size)/2)))
            assert x <= left < right <= x+size and y <= top < bottom <= y+size
            filename = f"{row['order']:02d}_frame{row['source_frame_index_zero_based']:03d}_t{row['time_s']:.3f}s.png"
            target = OUTPUT / name / filename
            filters = (f"select='eq(n,{row['source_frame_index_zero_based']})',"
                       f"crop={size}:{size}:{x}:{y}:exact=1,scale=768:768:flags=lanczos")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(video),
                            "-vf", filters, "-frames:v", "1", str(target)], check=True)
            row[name] = str(target.relative_to(OUTPUT))
            row.setdefault("crop_boxes_xyxy", {})[name] = [x,y,x+size,y+size]

    cell, gap, header = 512, 10, 70
    width, height = 6*cell+7*gap, 2*(cell+header)+3*gap
    sheet = Image.new("RGB", (width,height), "#eef1f5")
    draw = ImageDraw.Draw(sheet)
    labels = {"raw": "Original sequence", "grit": "MuJoCo + GRIT"}
    for r, name in enumerate(("raw", "grit")):
        y = gap+r*(cell+header+gap)
        draw.text((gap+8,y+4), labels[name], font=font(28), fill="#182c40")
        for c, row in enumerate(rows):
            x = gap+c*(cell+gap)
            draw.text((x+8,y+40), f"{row['order']:02d}  |  {row['time_s']:.3f} s", font=font(22), fill="#182c40")
            with Image.open(OUTPUT / row[name]) as im:
                sheet.paste(im.resize((cell,cell), Image.Resampling.LANCZOS), (x,y+header))
    sheet.save(OUTPUT / "comparison_6frames_closeup.png")
    (OUTPUT / "pairs").mkdir()
    for row in rows:
        pair = Image.new("RGB", (1536,824), "#eef1f5")
        draw = ImageDraw.Draw(pair)
        for c, name in enumerate(("raw", "grit")):
            draw.text((c*768+20,14), f"{labels[name]}  |  {row['time_s']:.3f} s", font=font(27), fill="#182c40")
            with Image.open(OUTPUT / row[name]) as im:
                pair.paste(im, (c*768,56))
        filename = f"pairs/pair_{row['order']:02d}_t{row['time_s']:.3f}s_closeup.png"
        pair.save(OUTPUT / filename)
        row["pair"] = filename
    source.update(keyframes=rows, screenshot_size=[768,768], crop_size_pixels=CROP_SIZE,
                  operation="Whole-body centering, square crop and uniform resize only; no pose editing",
                  purpose="Body-pose comparison; world translation intentionally removed by centering")
    (OUTPUT / "manifest.json").write_text(json.dumps(source,ensure_ascii=False,indent=2)+"\n")
    (OUTPUT / "README_ZH.md").write_text(
        "# 关键帧：全身放大居中版\n\n"
        "时间点与原截图完全一致。每张按机器人全身位置居中裁切，再等比例放大到 768×768。\n"
        "只改变取景，姿态未修改；保留头、手、脚。每组视频六帧采用固定裁切尺寸，"
        "避免弯腰等低姿态被单独放大。居中后不再表达全局位移，适合比较身体姿态。\n\n"
        "`raw/` 原始视频六张；`grit/` GRIT 六张；`pairs/` 六组左右对照；"
        "`comparison_6frames_closeup.png` 为上原始、下 GRIT 总览。\n", encoding="utf-8")
    bundle = BASE / "12_005600_keyframes_closeup.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob("*")):
            if p.is_file(): z.write(p, p.relative_to(BASE))
    with zipfile.ZipFile(bundle) as z: assert z.testzip() is None
    print(f"READY: {OUTPUT}\nZIP: {bundle}")


if __name__ == "__main__":
    main()
