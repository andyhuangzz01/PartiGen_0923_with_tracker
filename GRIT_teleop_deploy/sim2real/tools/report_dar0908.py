"""Reproduce 0908 statistics and export a separate, fully audited report."""
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile
from collections import Counter

import numpy as np

BASE = Path(__file__).resolve().parents[2]
SOURCE = BASE / "baseline_dar0908_mpjpe_0000_0499"
OUT = BASE / "dar0908_complete_report"
METRICS = ("global_mpjpe_mm", "root_relative_mpjpe_mm")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def stats(rows):
    ans = dict(motions=len(rows), frames=sum(r["tracking_recorded_frames"] for r in rows))
    for k in METRICS:
        a = np.array([r[k] for r in rows])
        ans[k] = dict(mean=float(a.mean()), median=float(np.median(a)),
            p95=float(np.percentile(a, 95)),
            frame_weighted_mean=float(np.average(a, weights=[r["tracking_recorded_frames"] for r in rows])))
    return ans


def write_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    original = json.loads((SOURCE / "summary.json").read_text())
    run = json.loads((SOURCE / "run.json").read_text())
    rows = sorted(original["motions"], key=lambda r:r["index"])
    assert [r["index"] for r in rows] == list(range(500))
    valid = [r for r in rows if not r.get("error") and all(r.get(k) is not None and math.isfinite(r[k]) for k in METRICS)]
    stable = [r for r in valid if r["root_z_min"] >= .60 and r["root_tilt_max_deg"] <= 35]
    stable_ids = {r["index"] for r in stable}
    nonstable = [r for r in valid if r["index"] not in stable_ids]
    invalid = [r for r in rows if r not in valid]
    assert len(valid) == 495 and len(stable) == 464 and len(invalid) == 5
    assert stable_ids == {r["index"] for r in rows if r["rating"] == "PASS"}
    current_source_hash = digest(Path(run["input"]))
    assert current_source_hash == run["input_sha256"]
    tracked_files_match = {p:Path(p).exists() and digest(Path(p)) == h for p,h in run["files"].items()}
    OUT.mkdir(exist_ok=False)
    diagnostic = {}
    for n,r in enumerate(valid,1):
        track = Path(r["attempt_dir"]) / "tracking.npz"
        with np.load(track,allow_pickle=False) as z:
            a,ref = z["actual_pos_w"], z["reference_pos_w"]
            ar,rr = z["actual_root_w"],z["reference_root_w"]
            assert a.shape == ref.shape and a.shape[1] == 19
            assert np.array_equal(z["motion_frame"],np.arange(len(a)))
            assert len(a) == r["tracking_recorded_frames"] == r["expected_motion_frames"]
            assert all(np.isfinite(x).all() for x in (a,ref,ar,rr))
            g=float(np.linalg.norm(a-ref,axis=-1).mean()*1000)
            root=float(np.linalg.norm((a-ar[:,None,:])-(ref-rr[:,None,:]),axis=-1).mean()*1000)
            assert np.isclose(g,r[METRICS[0]],atol=1e-5) and np.isclose(root,r[METRICS[1]],atol=1e-5)
            d=ar-rr
            one=dict(initial_xy_mm=float(np.linalg.norm(d[0,:2])*1000),initial_abs_z_mm=float(abs(d[0,2])*1000),
                initial_xyz_mm=float(np.linalg.norm(d[0])*1000), final_xyz_mm=float(np.linalg.norm(d[-1])*1000),
                fixed_initial_offset_global_mm=float(np.linalg.norm((a-ref)-d[0][None,None,:],axis=-1).mean()*1000),
                tracking_file=str(track),tracking_sha256=digest(track))
        with np.load(r["npz_file"],allow_pickle=False) as z:
            pos=z["body_pos_w"][:,0]; q=z["body_quat_w"][:,0].astype(float)
            q/=np.linalg.norm(q,axis=1,keepdims=True)
            tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
            one.update(reference_root_z_min_m=float(pos[:,2].min()), reference_root_tilt_max_deg=float(tilt.max()),
                reference_crosses_stability_threshold=bool(pos[:,2].min()<.6 or tilt.max()>35))
        diagnostic[r["index"]]=one
        if n%100==0: print(f"Verified trajectories: {n}/495",flush=True)
    subsets={"all_valid":valid,"stable":stable,"nonstable":nonstable}
    ordered=sorted(valid,key=lambda r:(-r[METRICS[0]],r["index"]))
    remove_count=math.ceil(.1*len(valid)); trimmed=ordered[remove_count:]
    statistics={k:stats(v) for k,v in subsets.items()}
    for label, stored_key in (("all_valid","all_valid"),("stable","stability_pass_subset")):
        for k in METRICS:
            saved=original["tracking_statistics"][stored_key][k]
            assert np.isclose(statistics[label][k]["mean"],saved["motion_equal_mean_mm"])
            assert np.isclose(statistics[label][k]["median"],saved["motion_median_mm"])
    offset_stats={label:{k:dict(mean=float(np.mean([diagnostic[r["index"]][k] for r in subset])),
        median=float(np.median([diagnostic[r["index"]][k] for r in subset]))) for k in
        ("initial_xy_mm","initial_abs_z_mm","initial_xyz_mm","final_xyz_mm","fixed_initial_offset_global_mm")}
        for label,subset in subsets.items() if label != "nonstable"}
    reference_bias=dict(all_crossing=sum(d["reference_crosses_stability_threshold"] for d in diagnostic.values()),
        failed_and_reference_crossing=sum(diagnostic[r["index"]]["reference_crosses_stability_threshold"] for r in nonstable),
        passed_and_reference_crossing=sum(diagnostic[r["index"]]["reference_crosses_stability_threshold"] for r in stable))
    body=json.loads((SOURCE/"body_points.json").read_text())
    data=dict(dataset="dar0908 generated motions, not original GT",source=str(SOURCE),input=run["input"],
        input_sha256=current_source_hash,summary_sha256=digest(SOURCE/"summary.json"),run_sha256=digest(SOURCE/"run.json"),
        original_input_total=original["total"],requested=500,index_range=[0,499],valid=495,errors=5,
        stable=464,stability_rate_valid=464/495,stability_rate_requested=464/500,
        status_counts=dict(Counter(r["rating"] for r in rows)),hardware_screen_counts=original["counts"],
        statistics=statistics,initial_offset_diagnostic=offset_stats,reference_threshold_diagnostic=reference_bias,
        trim10=dict(rule="Drop ceil(0.1*N) by original Global descending, same retained rows for both metrics",removed=remove_count,
            removed_indices=[r["index"] for r in ordered[:remove_count]],statistics=stats(trimmed)),
        body_points=body,runtime_versions=run["runtime_versions"],original_run_files_current_hash_match=tracked_files_match,
        integrity=dict(verified_tracking_files=len(diagnostic),all_trajectory_metrics_match_summary=True,
            recorded_frames=sum(r["tracking_recorded_frames"] for r in valid),
            missing_state_snapshots=sum(r["tracking_missing_state_snapshots"] for r in valid),
            duplicate_frames=sum(r["tracking_duplicate_frames"] for r in valid),
            invalid_references=sum(r["tracking_invalid_references"] for r in valid),
            max_abs_sim_dt_error_s=max(r["tracking_sim_dt_max_abs_error_s"] for r in valid)))
    fields=["index","src_idx","rating","stable_subset","global_mpjpe_mm","root_relative_mpjpe_mm",
        "root_z_min","root_tilt_max_deg","tracking_recorded_frames","expected_motion_frames",
        "source_frames","source_fps","trimmed_frames","hardware_screen","error","text","attempt_dir"]
    export=[dict(r,stable_subset=r["index"] in stable_ids,**diagnostic.get(r["index"],{})) for r in rows]
    fields += list(next(iter(diagnostic.values())).keys())
    write_csv(OUT/"all_500_motions.csv",export,fields)
    write_csv(OUT/"stable_464_motions.csv",[r for r in export if r["stable_subset"]],fields)
    write_csv(OUT/"nonstable_31_motions.csv",[r for r in export if not r["stable_subset"] and not r.get("error")],fields)
    write_csv(OUT/"errors_5.csv",[r for r in export if r.get("error")],fields)
    (OUT/"statistics.json").write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=["# 0908 完整闭环跟踪评估报告", "", "## 1. 范围与结论", "",
        "本报告仅汇总、核对已有实验，不重跑仿真，不覆盖原始结果。0908 是生成动作 baseline，不是 HML3D-G1 原始 GT。",
        f"输入：`{run['input']}`，文件含 2576 条；本批只评估列表零基 index 0–499，共 500 条。",
        "500 条均已有运行结论：495 条有效，5 条 ERROR 排除。稳定性代理判据通过 464 条：464/495 = 93.74%；若分母为全部请求数则为 464/500 = 92.80%。",
        "核心结果：全部有效 Global/Root-relative 均值为 275.61/40.01 mm；稳定子集为 252.47/35.20 mm。稳定子集是补充诊断，不替代全部有效主结果。", "",
        "## 2. 评估设置与指标定义", "",
        "- MuJoCo + 同一 GRIT ONNX 策略的闭环跟踪误差，不是生成动作与数据集 GT 之间的重建误差。",
        "- 原始 23 DOF 输入映射到 29 DOF 模型，缺少的腕部关节补零；源四元数 xyzw 转 wxyz。",
        "- trim_start=0，保留全部原帧，不裁头、不降速；参考从 30 Hz 重采样到 50 Hz。",
        "- 播放参考先按锚点对齐初始水平位置和 yaw，保持原参考高度；Global 比较的是该执行参考的世界坐标，不是未对齐的原文件坐标。",
        "- 首帧过渡为 5 秒；MPJPE 只统计 motion 播放阶段，不含入场、回默认和尾部观察。高度/倾角稳定性统计覆盖完整控制流程。",
        "- K=19，包含 pelvis；均为 MuJoCo 身体坐标系原点 xpos，不是质心 xipos。19 个位置点不等于 19 DoF。",
        "- 每帧参考与控制器读取的实际状态按状态 ID 配对，不做事后时间平移或最佳姿态匹配。", "", "身体点列表：", "", "```text",
        *[p["name"] for p in body["points"]], "```", "",
        r"设实际位置为 $p_{t,j}$、执行参考为 $\hat p_{t,j}$（m），根节点 $r$ 为 pelvis：", "",
        r"$$E_G=\frac{1000}{TK}\sum_{t,j}\|p_{t,j}-\hat p_{t,j}\|_2,$$", "",
        r"$$E_R=\frac{1000}{TK}\sum_{t,j}\|(p_{t,j}-p_{t,r})-(\hat p_{t,j}-\hat p_{t,r})\|_2.$$", "",
        "Root-relative 仅逐帧减去各自根平移，不去除根旋转、不做尺度/Procrustes 配准；pelvis 的零误差项仍计入 K=19 分母。",
        "先获得每条动作的平均 MPJPE，再对动作等权求均值/中位数。P95 也指动作均值的第 95 百分位，不是逐帧 P95；全帧加权均值另列。", "",
        "## 3. 全部有效与稳定子集结果", "", "单位均为 mm；没有剔除最高 10%，没有首帧偏移校正。", "",
        "| 范围 | 动作数 | 动作帧数 | Global 均值 | Global 中位数 | Root-relative 均值 | Root-relative 中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for key,label in (("all_valid","全部有效（主结果）"),("stable","稳定子集"),("nonstable","未通过稳定性判据")):
        s=statistics[key]; g,root=[s[k] for k in METRICS]
        lines.append(f"| {label} | {s['motions']} | {s['frames']} | {g['mean']:.2f} | {g['median']:.2f} | {root['mean']:.2f} | {root['median']:.2f} |")
    lines += ["", "| 范围 | Global P95 | Root-relative P95 | Global 全帧加权均值 | Root-relative 全帧加权均值 |", "|---|---:|---:|---:|---:|"]
    for key,label in (("all_valid","全部有效"),("stable","稳定子集"),("nonstable","未通过判据")):
        g,root=[statistics[key][k] for k in METRICS]
        lines.append(f"| {label} | {g['p95']:.2f} | {root['p95']:.2f} | {g['frame_weighted_mean']:.2f} | {root['frame_weighted_mean']:.2f} |")
    lines += ["", "## 4. 稳定性判据及局限", "",
        "稳定通过条件为有效运行且 root_z_min ≥ 0.60 m、root_tilt_max_deg ≤ 35°，与原日志 rating=PASS 一致。",
        "状态分布：PASS 464、MARGINAL 19、FALL 12、ERROR 5。FALL 也是高度/倾角代理标签，不是人工确认的实际跌倒次数。",
        "更严格的真机初筛使用另一套阈值（例如高度 0.65 m、倾角 20°，并包含速度、限位等），不能与以上稳定性判据混用。原初筛为 REVIEW 464、NOT_RECOMMENDED 36，Tier1/Tier2 均为 0；稳定通过不等于真机放行。",
        f"参考动作本身越过 0.60 m/35° 阈值的有 {reference_bias['all_crossing']} 条，其中实际未通过 {reference_bias['failed_and_reference_crossing']} 条、实际通过 {reference_bias['passed_and_reference_crossing']} 条。",
        "因此，有意下蹲、弯身等动作可能即使完美跟踪也不通过；反之，实际姿态未充分达到参考动作也可能通过。该比例应称高度/倾角稳定性代理判据通过率，而非无条件动作成功率。", "",
        "## 5. 数据完整性与 ERROR", "",
        f"离线逐条核对了全部 {len(valid)} 份有效 tracking.npz：动作帧从 0 连续到末帧，均完整，按轨迹重算两种 MPJPE 与 summary 一致。",
        f"有效动作帧合计 {data['integrity']['recorded_frames']}；缺失状态快照 {data['integrity']['missing_state_snapshots']}；重复动作帧 {data['integrity']['duplicate_frames']}；无效参考 {data['integrity']['invalid_references']}；最大相邻仿真时间间隔偏差 {data['integrity']['max_abs_sim_dt_error_s']:.3g} s。",
        "下面 5 条不纳入任何有效集/稳定子集 MPJPE 统计；本次未补跑或改变状态。", "",
        "| index（零基） | src_idx | 原错误原因 |", "|---:|---:|---|"]
    for r in invalid: lines.append(f"| {r['index']} | {r['src_idx']} | {r['error']} |")
    lines += ["", "## 6. 附加诊断（不替代主结果）", "", "### 6.1 剔除 Global 最高约 10%", "",
        f"按原 Global 降序剔除 ceil(495×10%)={remove_count} 条，两种指标用相同剩余 {len(trimmed)} 条；不叠加稳定筛选。", "",
        "| Global 均值 / 中位数 | Root-relative 均值 / 中位数 |", "|---:|---:|"]
    g,root=[data["trim10"]["statistics"][k] for k in METRICS]
    lines += [f"| {g['mean']:.2f} / {g['median']:.2f} | {root['mean']:.2f} / {root['median']:.2f} |", "",
        "### 6.2 首帧 pelvis 偏移", "",
        "首帧指入场插值后的动作播放第 0 帧。每条动作取 d=实际首帧 pelvis−参考首帧 pelvis（XYZ），所有帧所有身体点减去同一个固定 d 后再计算 Global；不删除首帧，不做旋转/尺度/时间配准。以下均为动作等权均值，单位 mm。", "",
        "| 范围 | 首帧 XY | 首帧绝对 Z | 首帧 XYZ | 末帧 XYZ | 固定首帧 XYZ 校正后 Global |", "|---|---:|---:|---:|---:|---:|"]
    for label,key in (("全部有效","all_valid"),("稳定子集","stable")):
        d=offset_stats[key]
        lines.append(f"| {label} | "+" | ".join(f"{d[k]['mean']:.2f}" for k in ("initial_xy_mm","initial_abs_z_mm","initial_xyz_mm","final_xyz_mm","fixed_initial_offset_global_mm"))+" |")
    lines += ["", "## 7. 与 0911 / 随机 GT 的描述性对照", "", "这不是同 prompt 配对实验；GT 是全库随机非镜像 29 DOF，而两组生成动作是 23 DOF、腕部补零。", "",
        "| 数据 | 有效数 | 稳定数 | 稳定率（有效数为分母） | 全部 Global / Root 均值 | 稳定子集 Global / Root 均值 |", "|---|---:|---:|---:|---:|---:|"]
    for label,folder in (("0908",SOURCE),("0911",BASE/"gen_test2576_dar0911_mpjpe_0001_0500"),("随机 GT",BASE/"gt_random500_seed20260918")):
        s=json.loads((folder/"summary.json").read_text()); av=s["tracking_statistics"]["all_valid"]; st=s["tracking_statistics"]["stability_pass_subset"]
        a=" / ".join(f"{av[k]['motion_equal_mean_mm']:.2f}" for k in METRICS); t=" / ".join(f"{st[k]['motion_equal_mean_mm']:.2f}" for k in METRICS)
        lines.append(f"| {label} | {s['valid_tracking_runs']} | {s['stability_pass_valid_runs']} | {s['stability_pass_valid_runs']/s['valid_tracking_runs']:.2%} | {a} | {t} |")
    lines += ["", "0908 的 Root-relative 误差和代理稳定通过率优于这两批样本，但 0911 的 Global 更低；不能据此作出无条件的模型优劣结论。", "",
        "## 8. 复核与文件", "",
        f"输入 SHA256：`{current_source_hash}`。原 run.json 记录的代码/配置/模型文件当前哈希全部一致：{all(tracked_files_match.values())}。",
        f"原运行版本：`{json.dumps(run['runtime_versions'])}`。", "",
        "- `stable_464_motions.csv`：用户所需稳定子集，含 index、src_idx、prompt、误差与诊断字段。",
        "- `all_500_motions.csv`：全部 500 条去留、错误和结果，首帧诊断仅有效动作提供。",
        "- `nonstable_31_motions.csv`、`errors_5.csv`：非稳定和错误样本分开。",
        "- `statistics.json`：精确数值、核对结果、截尾剔除名单、哈希及身体点列表。",
        f"- 原始目录：`{SOURCE}`；原始 summary/report/run 与逐条日志保持不变。",
        "- 复现报告：在仓库根目录运行 `GRIT_teleop_deploy/sim2real/.venv/bin/python GRIT_teleop_deploy/sim2real/tools/report_dar0908.py`。为防覆盖，已有输出目录时会拒绝运行。", ""]
    (OUT/"REPORT_ZH.md").write_text("\n".join(lines),encoding="utf-8")
    with zipfile.ZipFile(OUT.with_suffix('.zip'),"x",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.iterdir()): z.write(path,Path(OUT.name)/path.name)
    with zipfile.ZipFile(OUT.with_suffix('.zip')) as z: assert z.testzip() is None
    print(json.dumps({k:data[k] for k in ['statistics','initial_offset_diagnostic','integrity','reference_threshold_diagnostic']},indent=2))
    print(f"READY: {OUT}",flush=True)


if __name__ == "__main__":
    main()
