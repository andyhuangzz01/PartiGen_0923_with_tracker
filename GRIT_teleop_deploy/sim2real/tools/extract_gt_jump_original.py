"""Extract the five requested non-mirrored GT jumps without modifying sequences."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

import joblib
import numpy as np
import zstandard

REPO = Path(__file__).resolve().parents[3]
ARCHIVE = REPO / "HML3D-G1.tar.zst"
OUTPUT = REPO / "GRIT_teleop_deploy/gt_jump_original"
SELECTED = {
    "008739": (24, "竖直向上跳"),
    "004938": (44, "双臂张开腾空跳"),
    "009875": (66, "连续开合跳"),
    "002602": (190, "下蹲后向上跳，同时举臂"),
    "000608": (255, "站立、向上跳一次、落地"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {OUTPUT}")
    wanted = {f"HML3D-G1/{kind}/{mid}.{ext}": (mid, f"{mid}.{ext}")
              for mid in SELECTED for kind, ext in (("motion", "pkl"), ("texts", "txt"))}
    before = ARCHIVE.stat()
    found = {}
    # Retain only these ten small members in memory. No broad archive extraction.
    with ARCHIVE.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                if member.name not in wanted:
                    continue
                if not member.isfile() or not 0 < member.size < 20_000_000 or member.name in found:
                    raise ValueError(f"Invalid or duplicate member: {member.name}")
                payload = archive.extractfile(member).read()
                if len(payload) != member.size:
                    raise ValueError(f"Incomplete member: {member.name}")
                found[member.name] = payload
                print(f"Found {member.name} ({len(found)}/10)", flush=True)
    after = ARCHIVE.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source archive changed during extraction")
    if set(found) != set(wanted):
        raise RuntimeError(f"Missing archive members: {set(wanted) - set(found)}")
    OUTPUT.mkdir()
    for member, payload in found.items():
        mid, filename = wanted[member]
        folder = OUTPUT / mid
        folder.mkdir(exist_ok=True)
        with (folder / filename).open("xb") as stream:
            stream.write(payload)
        if digest((folder / filename).read_bytes()) != digest(payload):
            raise RuntimeError(f"Extracted bytes differ: {filename}")

    rows = []
    for mid, (generated_index, description) in SELECTED.items():
        folder = OUTPUT / mid
        record = joblib.load(folder / f"{mid}.pkl")
        pos, quat, joints = (np.asarray(record[k]) for k in ("root_pos", "root_rot", "dof_pos"))
        frames, fps = len(pos), float(record["fps"])
        if (pos.shape != (frames, 3) or quat.shape != (frames, 4)
                or joints.shape != (frames, 29) or fps <= 0
                or any(not np.isfinite(a).all() for a in (pos, quat, joints))):
            raise ValueError(f"Invalid GT motion dimensions or values: {mid}")
        # Concatenate source values directly: no normalization, reorder or cast.
        qpos = np.concatenate((pos, quat, joints), axis=1)
        np.save(folder / "qpos.npy", qpos, allow_pickle=False)
        arrays = {k: np.asarray(v) for k, v in record.items()}
        if any(a.dtype.hasobject for a in arrays.values()):
            raise ValueError(f"Unsupported object field in source: {mid}")
        np.savez_compressed(folder / "motion_original.npz", **arrays)
        with np.load(folder / "motion_original.npz", allow_pickle=False) as restored:
            if set(restored.files) != set(arrays):
                raise ValueError("NPZ fields differ")
            for key, value in arrays.items():
                if restored[key].dtype != value.dtype or not np.array_equal(restored[key], value):
                    raise ValueError(f"NPZ differs from original: {mid}/{key}")
        restored_qpos = np.load(folder / "qpos.npy", allow_pickle=False)
        for slc, value in ((slice(0, 3), pos), (slice(3, 7), quat), (slice(7, 36), joints)):
            if not np.array_equal(restored_qpos[:, slc], value):
                raise ValueError(f"Qpos values differ from original: {mid}")
        text = (folder / f"{mid}.txt").read_text(encoding="utf-8")
        captions = [line.split("#", 1)[0].strip() for line in text.splitlines() if line.strip()]
        row = dict(gt_id=mid, description_zh=description, generated_prompt_match_index=generated_index,
                   pairing_note="用户指定的同标签生成组编号；不能由文本证明唯一配对关系",
                   fps=fps, frames=frames, duration_frames_over_fps_s=frames / fps,
                   first_to_last_timestamp_s=(frames - 1) / fps,
                   qpos_shape=list(qpos.shape), qpos_dtype=str(qpos.dtype),
                   qpos_layout="root_pos[0:3], root_rot_xyzw[3:7], dof_pos_29[7:36]",
                   captions=captions, mirrored=False,
                   archive_members={kind: dict(path=f"HML3D-G1/{kind}/{mid}.{ext}",
                        sha256=digest(found[f"HML3D-G1/{kind}/{mid}.{ext}"]))
                        for kind, ext in (("motion", "pkl"), ("texts", "txt"))},
                   fields={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
                   files={p.name: digest(p.read_bytes()) for p in sorted(folder.iterdir())})
        write_json(folder / "metadata.json", row)
        rows.append(row)
        print(f"Exported {mid}: {frames} frames @ {fps:g} Hz, qpos {qpos.shape}", flush=True)
    write_json(OUTPUT / "manifest.json", dict(
        archive=str(ARCHIVE), archive_size=after.st_size, archive_mtime_ns=after.st_mtime_ns,
        source="HML3D-G1 archive; five explicitly requested non-M GT IDs",
        transformations="none; PKL/text byte-identical, source arrays value/dtype verified",
        split="not established by this archive export", count=len(rows), motions=rows))
    readme = ["# 五条 GT 跳跃动作：原始序列", "",
              "来源：`HML3D-G1.tar.zst`，按指定编号提取 non-M 原版。", "",
              "原始 PKL 与文本和压缩包内容逐字节一致。数组没有裁帧、重采样、降速、"
              "平滑、幅度修改或四元数归一化，29 个关节全部保留。", "",
              "| GT ID | 同标签生成组编号 | 标签含义 | 原始帧数 | Hz | 时长 T/fps (s) |",
              "|---|---:|---|---:|---:|---:|"]
    readme += [f"| {r['gt_id']} | {r['generated_prompt_match_index']} | {r['description_zh']} | "
               f"{r['frames']} | {r['fps']:g} | {r['duration_frames_over_fps_s']:.3f} |" for r in rows]
    readme += ["", "每个 GT 编号目录包含：", "",
        "- `<GT_ID>.pkl`：压缩包中的原始完整动作字典。",
        "- `<GT_ID>.txt`：压缩包中的全部原始文本标签。",
        "- `qpos.npy`：`(T,36)`，依次为根位置 3 维、根四元数 xyzw 4 维、关节角 29 维。",
        "- `motion_original.npz`：原始字典的所有字段，包括 fps、root_pos、root_rot、"
        "dof_pos、local_body_pos 和 link_body_list。各数值字段保留原始 dtype。",
        "- `metadata.json`：帧数、字段定义、标签及 SHA256。", "",
        "这里的 `qpos.npy` 采用源数据 xyzw 顺序；MuJoCo 自由关节使用 wxyz，"
        "因此不能直接作为 MuJoCo 的 qpos 数组赋值。"
        "`motion_original.npz` 也不是 GRIT 50 Hz 参考动作格式。", "",
        "读取示例：", "", "```python", "import numpy as np", "import joblib",
        "qpos = np.load('008739/qpos.npy', allow_pickle=False)",
        "motion = np.load('008739/motion_original.npz', allow_pickle=False)",
        "record = joblib.load('008739/008739.pkl')", "```", ""]
    (OUTPUT / "README_ZH.md").write_text("\n".join(readme), encoding="utf-8")
    bundle = OUTPUT.parent / "gt_jump_original.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file():
                z.write(path, str(path.relative_to(OUTPUT.parent)))
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
    print(f"READY: {OUTPUT}\nZIP: {bundle}", flush=True)


if __name__ == "__main__":
    main()
