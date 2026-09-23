"""Compare two completed screen_generated_motions summary files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = ("global_mpjpe_mm", "root_relative_mpjpe_mm")


def summarize(summary: dict, requested: int = 500) -> dict:
    rows = summary.get("motions", [])
    valid = [
        row for row in rows
        if not row.get("error")
        and all(row.get(field) is not None and np.isfinite(row[field]) for field in FIELDS)
    ]
    stable = [
        row for row in valid
        if row.get("root_z_min", -np.inf) >= 0.60
        and row.get("root_tilt_max_deg", np.inf) <= 35.0
    ]

    def subset_stats(subset: list[dict]) -> dict | None:
        if not subset:
            return None
        result = {"motions": len(subset)}
        for field in FIELDS:
            values = np.asarray([row[field] for row in subset], dtype=np.float64)
            weights = np.asarray([row["tracking_recorded_frames"] for row in subset], dtype=np.float64)
            result[field] = {
                "mean_mm": float(values.mean()),
                "median_mm": float(np.median(values)),
                "p95_mm": float(np.percentile(values, 95)),
                "frame_weighted_mean_mm": float(np.average(values, weights=weights)),
            }
        return result

    return {
        "requested_motions": requested,
        "completed_motions": len(rows),
        "valid_tracking_runs": len(valid),
        "errors": len(rows) - len(valid),
        "stability_pass_runs": len(stable),
        "stability_pass_fraction_valid": len(stable) / len(valid) if valid else None,
        "hardware_screen_counts": summary.get("counts", {}),
        "all_valid": subset_stats(valid),
        "stability_pass_subset": subset_stats(stable),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generated_raw = json.loads(args.generated.read_text())
    baseline_raw = json.loads(args.baseline.read_text())
    generated = summarize(generated_raw)
    baseline = summarize(baseline_raw)
    generated_ids = {r["index"] for r in generated_raw["motions"]}
    baseline_ids = {r["index"] for r in baseline_raw["motions"]}
    complete = generated_ids == set(range(1, 501)) and baseline_ids == set(range(500))
    generated_by_src = {r["src_idx"]: r for r in generated_raw["motions"]}
    baseline_by_src = {r["src_idx"]: r for r in baseline_raw["motions"]}
    shared_src = sorted(generated_by_src.keys() & baseline_by_src.keys())
    pairs = []
    for src in shared_src:
        g, b = generated_by_src[src], baseline_by_src[src]
        row = {"src_idx": src, "generated_index": g["index"], "baseline_index": b["index"],
               "prompt_equal": g.get("text", "") == b.get("text", ""),
               "generated_status": g.get("rating"), "baseline_status": b.get("rating"),
               "both_valid": not g.get("error") and not b.get("error")
                   and all(g.get(f) is not None and b.get(f) is not None
                           and np.isfinite(g[f]) and np.isfinite(b[f]) for f in FIELDS)}
        for field in FIELDS:
            row["generated_" + field] = g.get(field)
            row["baseline_" + field] = b.get(field)
            row["delta_" + field] = g[field] - b[field] if row["both_valid"] else None
        pairs.append(row)
    valid_pairs = [r for r in pairs if r["both_valid"]]
    paired = {
        "shared_src_idx_count": len(shared_src),
        "both_valid_count": len(valid_pairs),
        "prompt_equal_count": sum(r["prompt_equal"] for r in pairs),
        "mean_generated_minus_baseline_mm": {
            f: float(np.mean([r["delta_" + f] for r in valid_pairs])) if valid_pairs else None
            for f in FIELDS
        },
    }
    result = {
        "complete": complete,
        "scope": {
            "generated": "sample IDs 1..500 inclusive from unpacked dar0911 output",
            "baseline": "list indices 0..499 from gen_all2576_dar0908_512.pkl",
        },
        "generated": generated,
        "baseline": baseline,
        "paired_by_src_idx": paired,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if pairs:
        with (args.output / "paired_by_src_idx.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    (args.output / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 500 条生成动作与 baseline 的闭环 MPJPE 对比",
        "",
        "状态：" + ("两组各 500 条均已完成。" if complete else "尚未完成全部指定样本；以下是阶段结果。"),
        "",
        "生成组为 dar0911 解压目录 sample ID 1–500；baseline 为 dar0908 PKL 的前 500 条（index 0–499）。",
        "两组均使用同一 GRIT、MuJoCo、19 个 body/link 原点和逐状态精确时间配对。",
        "MPJPE 只统计动作播放阶段；稳定性与 pipeline 粗分级另列。",
        "Baseline 名称按用户指定文件标注；本次运行不据文件名认证其为原始 GT。",
        "主结果包含完整记录的失稳动作；ERROR 排除并单列，未完成样本不算 ERROR。均值为动作等权均值。",
        "",
        "| 数据 | 请求数 | 有效 MPJPE | ERROR | 稳定性通过 | Global mean / median (mm) | Root-relative mean / median (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, data in (("生成动作", generated), ("Baseline", baseline)):
        stats = data["all_valid"]
        if stats is None:
            global_text = local_text = "—"
        else:
            global_text = f"{stats['global_mpjpe_mm']['mean_mm']:.3f} / {stats['global_mpjpe_mm']['median_mm']:.3f}"
            local_text = f"{stats['root_relative_mpjpe_mm']['mean_mm']:.3f} / {stats['root_relative_mpjpe_mm']['median_mm']:.3f}"
        lines.append(
            f"| {label} | {data['requested_motions']} | {data['valid_tracking_runs']} | "
            f"{data['errors']} | {data['stability_pass_runs']} | {global_text} | {local_text} |"
        )
    lines += ["", "## Pipeline 粗分级", ""]
    for label, data in (("生成动作", generated), ("Baseline", baseline)):
        lines.append(f"- {label}: `{data['hardware_screen_counts']}`")
    lines += ["", "## 稳定性通过子集与全帧加权统计", "",
              "稳定性通过率分母为有效闭环运行数；子集 MPJPE 仅作补充，不替代全部有效运行主结果。", "",
              "| 数据 | 稳定性通过率 | 稳定子集 Global / Root-relative (mm) | 全部有效帧加权 Global / Root-relative (mm) |",
              "|---|---:|---:|---:|"]
    for label, data in (("生成动作", generated), ("Baseline", baseline)):
        fraction = data["stability_pass_fraction_valid"]
        rate = "—" if fraction is None else f"{fraction:.2%} ({data['stability_pass_runs']}/{data['valid_tracking_runs']})"
        stable_stats, all_stats = data["stability_pass_subset"], data["all_valid"]
        stable_text = "—" if not stable_stats else " / ".join(f"{stable_stats[f]['mean_mm']:.3f}" for f in FIELDS)
        weighted_text = "—" if not all_stats else " / ".join(f"{all_stats[f]['frame_weighted_mean_mm']:.3f}" for f in FIELDS)
        lines.append(f"| {label} | {rate} | {stable_text} | {weighted_text} |")
    lines += ["", "## 按共同 src_idx 配对的补充比较", "",
              "两组选取区间相差一个编号，不能将列表位置直接配对。补充比较仅对齐相同 src_idx，仍分别对各自执行参考计算闭环误差。",
              f"共同 src_idx：{len(pairs)}；双方有效：{len(valid_pairs)}；文本相同：{paired['prompt_equal_count']}。文本不同的配对不视为相同 prompt 实验。", ""]
    for field, delta in paired["mean_generated_minus_baseline_mm"].items():
        lines.append(f"- {field} 生成组减 baseline 的配对平均差：" + ("—" if delta is None else f"{delta:.3f} mm"))
    lines += [
        "",
        "注：没有新增 MPJPE 合格阈值；CANDIDATE/REVIEW/NOT_RECOMMENDED 仍按既有稳定性、关节误差、速度、限位和文字风险规则判定。",
    ]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
