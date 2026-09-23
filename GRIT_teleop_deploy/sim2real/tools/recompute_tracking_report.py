"""Reconstruct tracking statistics from saved per-joint RMSE and sample counts.

No simulation or hardware execution. This is NOT a frame-level recomputation:
the existing batch did not save actual per-frame joint states.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

LEGS = tuple(f"{side}_{joint}_joint" for side in ("left", "right") for joint in
             ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
JOINTS = set(LEGS) | {f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")} | {
    f"{side}_{joint}_joint" for side in ("left", "right") for joint in
    ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")}
LIMITS = {"rmse_all_rad": 0.15, "rmse_legs_rad": 0.12, "max_joint_error_rad": 0.6}


def rms(values):
    values = list(values)
    return math.sqrt(math.fsum(v * v for v in values) / len(values))


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def aggregate(rows, field):
    if not rows:
        return None
    values = [r[field] for r in rows]
    result = dict(motion_mean=math.fsum(values) / len(values), median=quantile(values, .5),
                  p90=quantile(values, .9), p95=quantile(values, .95), maximum=max(values))
    if field != "max_joint_error_rad":
        n = sum(r["reference_samples"] for r in rows)
        result["sample_weighted_rmse"] = math.sqrt(math.fsum(r["reference_samples"] * r[field] ** 2 for r in rows) / n)
    return result


def audit(result, metrics):
    if result.get("error") or result.get("rating") == "ERROR":
        raise ValueError(result.get("error") or "original ERROR")
    by_joint = metrics["reference_joint_rmse_by_name_rad"]
    if set(by_joint) != JOINTS:
        raise ValueError("joint names differ from the 29-joint G1 contract")
    n = metrics["reference_samples"]
    if not isinstance(n, int) or n <= 0 or n > metrics["control_samples"]:
        raise ValueError("invalid reference sample count")
    if any(not math.isfinite(v) or v < 0 for v in by_joint.values()):
        raise ValueError("invalid per-joint RMSE")
    all_rms, legs_rms = rms(by_joint.values()), rms(by_joint[j] for j in LEGS)
    for key, value in (("reference_joint_rmse_rad", all_rms), ("reference_lower_body_rmse_rad", legs_rms)):
        if not math.isclose(value, metrics[key], abs_tol=1e-9, rel_tol=1e-9):
            raise ValueError(f"inconsistent reconstructed metric: {key}")
        if not math.isclose(value, result[key], abs_tol=1e-9, rel_tol=1e-9):
            raise ValueError(f"result/metrics mismatch: {key}")
    peak = metrics["reference_joint_error_max_rad"]
    z, tilt = metrics["root_z_min"], metrics["root_tilt_max_deg"]
    if not all(math.isfinite(v) for v in (peak, z, tilt)) or peak < max(by_joint.values()):
        raise ValueError("invalid maximum error or stability metrics")
    stable = z >= .60 and tilt <= 35
    row = dict(reference_samples=n, rmse_all_rad=all_rms, rmse_legs_rad=legs_rms,
               rmse_23_nonwrist_rad=rms(v for j, v in by_joint.items() if "wrist" not in j),
               max_joint_error_rad=peak, stability_pass=stable)
    row["failed_tracking_checks"] = [key for key, limit in LIMITS.items() if row[key] > limit]
    row["tracking_pass"] = not row["failed_tracking_checks"]
    row["stable_and_tracking_pass"] = stable and row["tracking_pass"]
    return row, by_joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="batch output directory")
    parser.add_argument("--output", type=Path, required=True, help="new report directory; must not exist")
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    total = int(json.loads((args.input / "summary.json").read_text())["total"])
    rows, joints, hashes = [], {}, {}
    for index in range(total):
        path = args.input / f"{index:04d}" / "result.json"
        row = dict(index=index, valid=False)
        try:
            raw = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            result = json.loads(raw)
            if result["index"] != index:
                raise ValueError("index mismatch")
            row.update(src_idx=result.get("src_idx"), text=result.get("text", ""),
                       original_rating=result.get("rating"), attempt_dir=result["attempt_dir"])
            if result.get("error") or result.get("rating") == "ERROR":
                raise ValueError(result.get("error") or "original ERROR")
            metric_path = Path(result["attempt_dir"]) / "metrics.json"
            raw = metric_path.read_bytes()
            hashes[str(metric_path)] = hashlib.sha256(raw).hexdigest()
            values, by_joint = audit(result, json.loads(raw))
            row.update(values, valid=True)
            joints[index] = by_joint
        except (ValueError, KeyError, TypeError, OSError) as exc:
            row["invalid_reason"] = str(exc)
        rows.append(row)
    valid = [r for r in rows if r["valid"]]
    stable = [r for r in valid if r["stability_pass"]]
    tracked = [r for r in valid if r["tracking_pass"]]
    combined = [r for r in valid if r["stable_and_tracking_pass"]]
    counts = dict(total=total, valid=len(valid), invalid=total-len(valid), stable=len(stable),
                  tracking_pass=len(tracked), stable_and_tracking_pass=len(combined),
                  stable_but_tracking_fail=len(stable)-len(combined))
    rates = {name: dict(numerator=len(subset), denominator_total=total,
                       percent_total=100 * len(subset) / total,
                       denominator_valid=len(valid), percent_valid=100 * len(subset) / len(valid) if valid else None)
             for name, subset in (("stability", stable), ("tracking", tracked), ("stable_and_tracking", combined))}
    stats = {name: {field: aggregate(subset, field) for field in
                   ("rmse_all_rad", "rmse_legs_rad", "rmse_23_nonwrist_rad", "max_joint_error_rad")}
             for name, subset in (("all_valid_including_falls", valid), ("stable_only", stable))}
    checks = {key: sum(r[key] > limit for r in valid) for key, limit in LIMITS.items()}
    per_joint = []
    if valid:
        samples = sum(r["reference_samples"] for r in valid)
        for name in sorted(JOINTS):
            pooled = math.sqrt(math.fsum(r["reference_samples"] * joints[r["index"]][name] ** 2 for r in valid) / samples)
            per_joint.append(dict(joint=name, sample_weighted_rmse_rad=pooled,
                                  motion_mean_rmse_rad=math.fsum(joints[r["index"]][name] for r in valid)/len(valid)))
        per_joint.sort(key=lambda r: r["sample_weighted_rmse_rad"], reverse=True)
    args.output.mkdir(parents=True, exist_ok=False)
    payload = dict(method="Reconstruction from saved per-joint RMSE; NOT raw-frame recomputation",
                   counts=counts, rates=rates, thresholds_rad=LIMITS, statistics_rad=stats,
                   failed_checks_nonexclusive=checks, per_joint=per_joint, motions=rows)
    (args.output / "tracking_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (args.output / "input_sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    fields = ["index", "src_idx", "valid", "original_rating", "stability_pass", "tracking_pass", "stable_and_tracking_pass",
              "reference_samples", "rmse_all_rad", "rmse_legs_rad", "rmse_23_nonwrist_rad", "max_joint_error_rad",
              "failed_tracking_checks", "invalid_reason", "text", "attempt_dir"]
    with (args.output / "tracking_motions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 2576 条动作：跟踪指标独立重算", "",
             "本报告读取每个 index 最新 result.json 指向的 metrics.json，不覆盖原报告、不重跑仿真。",
             "保存文件没有实际逐帧状态：本次从逐关节 RMSE 重建全身/下肢 RMSE，并用采样数恢复平方误差权重。",
             "最大误差沿用原统计值，不能由 RMSE 重建。统计包含入场、动作、回默认及尾部观察；不是纯动作阶段指标。", "",
             "公式：单动作组 RMSE = sqrt(sum(逐关节 RMSE²)/关节数)；全数据 pooled RMSE = sqrt(sum(N_i × RMSE_i²)/sum(N_i))。",
             "全身 29 关节；下肢为左右髋、膝、踝共 12 关节，不含腰。额外列出不含腕的 23 关节诊断指标。", "",
             "## 有效性与达标数", "", f"`{counts}`", "",
             "运行错误不算动作跟踪失败，归入无效试验。有效集包含 FALL/MARGINAL，避免只统计成功样本。",
             "全测试集比例将无效试验计入分母但不计为成功；另列排除无效试验后的比例。", "",
             "| 项目 | 数量 | 占全部动作 | 占有效试验 |", "|---|---:|---:|---:|"]
    for name, r in rates.items():
        pct = "N/A" if r["percent_valid"] is None else f"{r['percent_valid']:.2f}%"
        lines.append(f"| {name} | {r['numerator']} | {r['percent_total']:.2f}% | {pct} |")
    lines += ["", "跟踪达标定义：全身 RMSE ≤0.15 rad、12 下肢关节 RMSE ≤0.12 rad、最大单关节参考误差 ≤0.6 rad，三项同时满足。",
              "该定义不含速度、力矩或描述关键词筛选，不等于真机候选标准。稳定且跟踪达标还要求高度 ≥0.60 m、倾角 ≤35°。",
              "这些是事先使用的工程阈值，不是通用成功标准；没有根部位置/朝向跟踪评价，也没有相位对齐。", "",
              "## 误差分布（rad）", "", "均值/中位数/P90/P95 均针对每条动作的指标，不能解释为逐帧误差分位数。", "",
              "| 样本集 | 指标 | 动作等权均值 | 中位数 | P90 | P95 | 最大值 | 采样数加权 RMSE |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for subset, metrics in stats.items():
        for field, stat in metrics.items():
            if stat:
                vals = [f"{stat[k]:.6f}" for k in ("motion_mean", "median", "p90", "p95", "maximum")]
                pooled = f"{stat['sample_weighted_rmse']:.6f}" if "sample_weighted_rmse" in stat else "—"
                lines.append(f"| {subset} | {field} | " + " | ".join(vals+[pooled]) + " |")
    lines += ["", "## 超限原因（可重叠，不可直接相加）", "", f"`{checks}`", "",
              "## 各关节采样数加权 RMSE", "", "| 关节 | rad |", "|---|---:|"]
    lines += [f"| {r['joint']} | {r['sample_weighted_rmse_rad']:.6f} |" for r in per_joint]
    lines += ["", "## 无效试验", "", str(dict(Counter(r['invalid_reason'] for r in rows if not r['valid']))), "",
              "## 使用限制", "", "要从逐帧状态重新计算 MAE、逐帧 P95、动作阶段指标、根部/足端误差或持续超限时间，需要新增记录并重新仿真；本报告不能恢复这些信息。",
              "逐动作数据见 tracking_motions.csv / tracking_summary.json；输入哈希见 input_sha256.json。"]
    (args.output / "tracking_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(counts=counts, rates=rates, failed_checks=checks), ensure_ascii=False, indent=2))
    print(f"Report: {args.output / 'tracking_report.md'}")


if __name__ == "__main__":
    main()
