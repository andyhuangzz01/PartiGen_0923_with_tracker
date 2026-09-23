"""Read-only trajectory diagnostics; keep original benchmark results unchanged."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "gt_generated_initial_pelvis_comparison"
SOURCES = {"GT": BASE / "gt_random500_seed20260918/summary.json",
           "Generated": BASE / "gen_test2576_dar0911_mpjpe_0001_0500/summary.json"}


def summarize(rows):
    fields = [k for k in rows[0] if k.endswith("_mm")]
    result = {"motions": len(rows), "statistics": {}}
    for field in fields:
        v = np.array([r[field] for r in rows])
        result["statistics"][field] = dict(mean=float(v.mean()), median=float(np.median(v)), p90=float(np.percentile(v, 90)))
    result["initial_xy_over_100mm_count"] = sum(r["initial_xy_mm"] > 100 for r in rows)
    result["initial_xy_over_500mm_count"] = sum(r["initial_xy_mm"] > 500 for r in rows)
    return result


def main():
    OUT.mkdir(exist_ok=False)
    report = dict(method="Motion phase frame 0, exact saved actual/reference states. d0=actual_root[0]-reference_root[0]. Fixed-offset Global averages norm(actual_pos-reference_pos-d0) over all frames and 19 points. No rotation/scale/time alignment; not a replacement for original Global. Per-motion equal weights; trimmed subsets use ORIGINAL Global rank, ceil(10%*N), never reranked after correction.", groups={})
    for name, path in SOURCES.items():
        blob = path.read_bytes()
        source = json.loads(blob)
        valid = [r for r in source["motions"] if not r.get("error") and r.get("global_mpjpe_mm") is not None]
        ordered = sorted(valid, key=lambda r: (-r["global_mpjpe_mm"], r["index"]))
        removed = {r["index"] for r in ordered[:math.ceil(.1*len(valid))]}
        rows = []
        for r in valid:
            track = Path(r["attempt_dir"]) / "tracking.npz"
            with np.load(track, allow_pickle=False) as z:
                frames = z["motion_frame"]
                assert frames[0] == 0 and np.all(np.diff(frames) == 1)
                assert len(frames) == r["tracking_recorded_frames"]
                a, ref = z["actual_pos_w"], z["reference_pos_w"]
                ar, rr = z["actual_root_w"], z["reference_root_w"]
                assert a.shape == ref.shape and a.shape[1] == 19
                d = ar-rr
                e = a-ref
                global_mm = float(np.linalg.norm(e, axis=-1).mean()*1000)
                assert np.isclose(global_mm, r["global_mpjpe_mm"], atol=1e-5)
                corrected = float(np.linalg.norm(e-d[0][None,None,:],axis=-1).mean()*1000)
                row = dict(index=r["index"], gt_id=r.get("gt_id"), retained_trim10=r["index"] not in removed,
                    frames=len(frames), global_mpjpe_mm=global_mm, root_relative_mpjpe_mm=r["root_relative_mpjpe_mm"],
                    initial_xyz_mm=float(np.linalg.norm(d[0])*1000), initial_xy_mm=float(np.linalg.norm(d[0,:2])*1000),
                    initial_abs_z_mm=float(abs(d[0,2])*1000), initial_signed_z_mm=float(d[0,2]*1000),
                    final_xyz_mm=float(np.linalg.norm(d[-1])*1000), final_xy_mm=float(np.linalg.norm(d[-1,:2])*1000),
                    final_abs_z_mm=float(abs(d[-1,2])*1000),
                    root_mean_xyz_mm=float(np.linalg.norm(d,axis=-1).mean()*1000),
                    final_relative_displacement_xyz_mm=float(np.linalg.norm(d[-1]-d[0])*1000),
                    fixed_initial_offset_global_mm=corrected,
                    correction_reduction_mm=global_mm-corrected,
                    tracking_path=str(track), tracking_sha256=hashlib.sha256(track.read_bytes()).hexdigest())
                rows.append(row)
        with (OUT / f"{name.lower()}_per_motion.csv").open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        report["groups"][name] = dict(source=str(path), source_sha256=hashlib.sha256(blob).hexdigest(),
            all_valid=summarize(rows), trimmed10=summarize([r for r in rows if r["retained_trim10"]]))
        print(name, len(rows), "trajectories verified", flush=True)
    (OUT / "comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    lines = ["# 首帧 pelvis 与全局漂移诊断", "", "只读取已保存轨迹，不重跑仿真，不更改原始评估。单位 mm；表中为动作等权均值。", "",
        "首帧指动作播放第 0 帧（入场插值结束后）。固定首帧修正：d0=实际首帧 pelvis−参考首帧 pelvis，所有时刻、所有实际身体点减去同一个 d0，再算 Global MPJPE。仅平移，不做旋转/尺度/时间配准。", "",
        "| 范围 | 组别 | 数量 | 首帧 XY | 首帧绝对 Z | 首帧 XYZ | 末帧 XYZ | 原 Global | 固定首帧修正 Global |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scope in ("all_valid", "trimmed10"):
        for name, group in report["groups"].items():
            sub=group[scope]; st=sub["statistics"]
            values=[st[k]["mean"] for k in ("initial_xy_mm","initial_abs_z_mm","initial_xyz_mm","final_xyz_mm","global_mpjpe_mm","fixed_initial_offset_global_mm")]
            lines.append(f"| {scope} | {name} | {sub['motions']} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "trimmed10 使用此前原 Global 排名剔除最高约 10% 的相同样本，不按修正后误差重排。",
        "首帧修正仅作诊断：用观测状态消除初始平移，不能替代原始主指标，也不能单独证明入场阶段是唯一误差来源。两组非配对，GT 为 29DOF、生成组为 23DOF 腕部补零。",
        "逐条 CSV 包含首末帧、相对位移变化、轨迹路径和校验值；JSON 包含均值、中位数和 P90。", ""]
    (OUT / "report.md").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
