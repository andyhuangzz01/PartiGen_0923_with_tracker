"""Separate descriptive comparison after per-group upper-10% Global trimming."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "gt_generated_global_trim10_comparison"
SOURCES = {
    "GT": BASE / "gt_random500_seed20260918/summary.json",
    "Generated": BASE / "gen_test2576_dar0911_mpjpe_0001_0500/summary.json",
}


def stats(rows):
    out = {"motions": len(rows)}
    for field in ("global_mpjpe_mm", "root_relative_mpjpe_mm"):
        values = np.array([r[field] for r in rows], dtype=float)
        out[field] = {"motion_equal_mean_mm": float(values.mean()),
                      "median_mm": float(np.median(values))}
    return out


def main():
    OUT.mkdir(exist_ok=False)
    result = {"method": "Within each group, exclude ERROR/nonfinite results, sort by Global MPJPE descending, remove ceil(0.10*N) motions; ties broken by index ascending. Both metrics use identical retained motions. Motion-equal means. No stability filtering, pairing, or new simulations.", "groups": {}}
    for group, source in SOURCES.items():
        original = source.read_bytes()
        summary = json.loads(original)
        valid = [r for r in summary["motions"] if not r.get("error")
                 and all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k])
                         for k in ("global_mpjpe_mm", "root_relative_mpjpe_mm"))]
        ordered = sorted(valid, key=lambda r: (-r["global_mpjpe_mm"], r["index"]))
        count = math.ceil(len(ordered)*.1)
        removed, retained = ordered[:count], ordered[count:]
        assert len(removed)+len(retained) == len(valid)
        result["groups"][group] = dict(source=str(source), source_sha256=hashlib.sha256(original).hexdigest(),
            original=stats(valid), trimmed=stats(retained), excluded_invalid=len(summary["motions"])-len(valid),
            removed_count=count, removed_fraction=count/len(valid),
            minimum_removed_global_mm=removed[-1]["global_mpjpe_mm"],
            maximum_retained_global_mm=retained[0]["global_mpjpe_mm"])
        with (OUT / f"{group.lower()}_selection.csv").open("w", newline="") as f:
            fields = ["rank_global_desc", "selection", "index", "gt_id", "src_idx", "global_mpjpe_mm", "root_relative_mpjpe_mm", "rating", "text"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, row in enumerate(ordered):
                writer.writerow(dict({k:row.get(k) for k in fields[2:]}, rank_global_desc=i+1,
                    selection="removed" if i<count else "retained"))
    g, t = result["groups"]["Generated"]["trimmed"], result["groups"]["GT"]["trimmed"]
    result["generated_relative_to_gt"] = {k: dict(difference_mm=g[k]["motion_equal_mean_mm"]-t[k]["motion_equal_mean_mm"],
        change_percent=100*(g[k]["motion_equal_mean_mm"]/t[k]["motion_equal_mean_mm"]-1))
        for k in ("global_mpjpe_mm", "root_relative_mpjpe_mm")}
    lines = ["# GT 与生成组：剔除 Global MPJPE 最高 10% 后比较", "",
        "各组独立按动作级 Global MPJPE 降序排序，剔除 ceil(有效动作数×10%) 条。ERROR 先排除。",
        "两种指标使用相同保留样本；均值按动作等权，不另按 Root-relative 排名删除。原始结果不修改。", "",
        "| 组别 | 原有效数 | 剔除数 | 保留数 | 原 Global 均值 | 截尾 Global 均值 | 原 Root-relative 均值 | 截尾 Root-relative 均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, r in result["groups"].items():
        a, b = r["original"], r["trimmed"]
        lines.append(f"| {label} | {a['motions']} | {r['removed_count']} | {b['motions']} | {a['global_mpjpe_mm']['motion_equal_mean_mm']:.3f} | {b['global_mpjpe_mm']['motion_equal_mean_mm']:.3f} | {a['root_relative_mpjpe_mm']['motion_equal_mean_mm']:.3f} | {b['root_relative_mpjpe_mm']['motion_equal_mean_mm']:.3f} |")
    lines += ["", "单位 mm。GT 为随机非镜像全库样本（29 DOF），生成组为此前指定样本（23 DOF、腕部补零），不是配对实验。",
        "这是按观测误差截尾的补充描述性统计，不是无筛选主结果，不能证明生成模型优于 GT。",
        "*_selection.csv 保留每条动作的去留标记、排序、误差和原始编号；comparison.json 保存精确数值与源文件哈希。", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
