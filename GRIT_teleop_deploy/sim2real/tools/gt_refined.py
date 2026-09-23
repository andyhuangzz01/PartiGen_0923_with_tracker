"""Read-only launch resolution and three-run qualification for refined GT 005225."""
import argparse
import json
from pathlib import Path

from generated_dataset import digest
from refine_gt_005225 import DATA, SPECS
from screen_generated_motions import LIMITS, atomic_json, classify

REQUIRED_RUNS = 3
CANDIDATES = {"CANDIDATE_TIER1", "CANDIDATE_TIER2"}


def qualified(result):
    return (not result.get("error") and result.get("rating")=="PASS"
            and result.get("hardware_screen") in CANDIDATES
            and result.get("tracking_motion_frames_complete")
            and result.get("tracking_recorded_frames")==result.get("expected_motion_frames")
            and classify(result)[1] in CANDIDATES)


def checked_prepared(name):
    if name not in SPECS: raise ValueError("Unknown refinement")
    paths=sorted((DATA/"variants").glob(f"{name}_*/prepared.json"),key=lambda p:p.stat().st_mtime_ns)
    if not paths: raise ValueError("调整版尚未准备；先运行 refine_gt_005225.py。")
    path=paths[-1]
    result=json.loads(path.read_text())
    if result["adjustment"]!=SPECS[name]: raise ValueError("调整参数已变化")
    for file,sha in {**result["signature"],**result["artifacts"]}.items():
        if str(file).startswith("/") and (not Path(file).is_file() or digest(Path(file))!=sha):
            raise ValueError(f"输入/配置/调整文件已变化：{file}")
    return path,result


def latest_runs(prepared):
    return sorted(Path(prepared["attempt_dir"]).glob("eval_*/result.json"),key=lambda p:p.stat().st_mtime_ns)


def confirmed_result(prepared, paths):
    if len(paths)!=REQUIRED_RUNS: raise ValueError("需要最近连续 3 次完整通过，不能挑选历史最好结果。")
    results=[json.loads(p.read_text()) for p in paths]
    for r in results:
        if (r["signature"]!=prepared["signature"] or r["artifacts"]!=prepared["artifacts"]
                or not qualified(r)):
            raise ValueError("最近 3 次中有未通过/不完整/文件不一致的运行，真机保持拦截。")
    aggregate=dict(results[-1])
    for key in LIMITS:
        aggregate[key]=(min if key=="root_z_min" else max)(r[key] for r in results)
    aggregate["rating"],aggregate["hardware_screen"],aggregate["reasons"]=classify(aggregate)
    aggregate["confirmed_runs"]=REQUIRED_RUNS
    aggregate["confirmation_metrics"]=[{k:r[k] for k in
        ("global_mpjpe_mm","root_relative_mpjpe_mm","root_tilt_max_deg","reference_joint_error_max_rad")}
        for r in results]
    return aggregate


def publish(name):
    path,prepared=checked_prepared(name)
    runs=latest_runs(prepared)[-REQUIRED_RUNS:]
    result=confirmed_result(prepared,runs)
    selection=dict(variant=name,prepared_file=str(path),prepared_sha256=digest(path),
                   runs={str(p):digest(p) for p in runs})
    atomic_json(DATA/"selection.json",selection)
    print(f"Selected {name}: {result['hardware_screen']}, {REQUIRED_RUNS} consecutive complete runs")


def get_result(index):
    if index!=5225: raise ValueError("当前调整版仅支持 GT 005225。")
    selection_path=DATA/"selection.json"
    if not selection_path.is_file():
        _,prepared=checked_prepared("gentle90")
        runs=latest_runs(prepared)
        result=json.loads(runs[-1].read_text()) if runs else prepared
        if result.get("hardware_screen") in CANDIDATES:
            result=dict(result,hardware_screen="REVIEW",reasons=["尚未确认最近连续 3 次完整通过。"])
        return result
    selection=json.loads(selection_path.read_text())
    path,prepared=checked_prepared(selection["variant"])
    if str(path)!=selection["prepared_file"] or digest(path)!=selection["prepared_sha256"]:
        raise ValueError("已确认的调整版准备记录已变化")
    runs=latest_runs(prepared)[-REQUIRED_RUNS:]
    if {str(p):digest(p) for p in runs}!=selection["runs"]:
        raise ValueError("确认后出现了新运行或记录变化；需重新确认，不能复用旧放行。")
    return confirmed_result(prepared,runs)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select",choices=list(SPECS),required=True)
    publish(parser.parse_args().select)
