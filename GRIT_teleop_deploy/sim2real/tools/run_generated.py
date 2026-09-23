"""Launch dar0911 generated motions, or preview dar0908 PKL with --baseline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "gen_test2576_dar0911_mpjpe_0001_0500"
PYTHON = str(ROOT / ".venv/bin/python")
CANDIDATES = {"CANDIDATE_TIER1", "CANDIDATE_TIER2"}


def build_commands(index: int, hardware: bool, baseline: bool = False, dry_run: bool = False, gt: bool = False, slow: bool = False, refined: bool = False, force: bool = False):
    if refined and (not gt or slow):
        raise ValueError("--refined 仅用于 --gt，且不能与 --slow 同时使用。")
    if slow and not gt:
        raise ValueError("--slow 仅用于 --gt 的独立降速副本。")
    if gt and baseline:
        raise ValueError("--gt 与 --baseline 不能同时使用。")
    if gt and refined:
        from gt_refined import get_result
        result = get_result(index)
    elif gt:
        from gt_arm_leg import DATA, get_result, gt_id
        if slow:
            from gt_slow import DATA, get_result
        if dry_run and not (DATA / "prepared" / gt_id(index) / "prepared.json").is_file():
            raise ValueError("GT 尚未转换；请先运行 tools/gt_arm_leg.py，dry-run 不创建文件。")
        result = get_result(index)
    elif baseline:
        if hardware:
            raise ValueError("--baseline 当前仅支持仿真，不启动真机控制器。")
        from baseline_preview import prepare
        result = prepare(index, dry_run=dry_run)
    else:
        if not 0 <= index < 2576:
            raise ValueError("生成组 sample ID 范围为 0–2575。")
        if hardware and not 1 <= index <= 500:
            raise ValueError("真机入口仅支持原已评估的 1–500；新增动作仅支持 --sim。")
        result_path = RESULTS / f"{index:04d}" / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        elif hardware:
            raise ValueError(f"没有此编号的评估结果：{result_path}")
        else:
            from generated_preview import prepare
            result = prepare(index, dry_run=dry_run)
    if result.get("index") != index:
        raise ValueError("结果文件中的 index 与请求不一致。")
    if result.get("error"):
        raise ValueError(f"index {index} 评估存在错误：{result['error']}，--force 不能放行。")
    if hardware and (result.get("rating") != "PASS"
                     or result.get("hardware_screen") not in CANDIDATES
                     or not result.get("tracking_motion_frames_complete")):
        message = (
            f"index {index} 为 {result.get('rating')} / {result.get('hardware_screen')}，"
            "未进入本批真机候选。请使用 --sim 复核。\n"
            + "; ".join(result.get("reasons", []))
        )
        if not force:
            raise ValueError(message)
        print(f"[run_generated] 警告：--force 强行放行，硬件初筛未通过，风险自负。\n{message}", flush=True)
    if hardware and gt:
        import yaml
        tracking = yaml.safe_load((ROOT / "config/g1/tracking.yaml").read_text())
        controller = yaml.safe_load((ROOT / "config/g1/controller.yaml").read_text())
        if (tracking["motion_source"]["npz"]["first_frame_transition_s"] != 5.0
                or tracking["motion_source"]["npz"]["loop"]
                or controller["startup_interpolation_s"] != 5.0):
            raise ValueError("GT 真机要求两段 5 秒插值及单次播放。")
    motion = Path(result.get("npz_file", ""))
    if not motion.is_file() and not result.get("preview_not_prepared"):
        raise ValueError(f"该动作缺少转换后的 NPZ：{motion}")
    deploy = [PYTHON, str(ROOT / "src/deploy.py"), "--robot", "g1",
              "--tracking-config", str(ROOT / "config/g1/tracking.yaml"),
              "--policy-path", str(ROOT / "checkpoints/policy.onnx"),
              "--motion-file", str(motion)]
    if hardware:
        deploy += ["--controller-config", str(ROOT / "config/g1/controller.yaml")]
        return result, None, deploy

    attempt = Path(result["attempt_dir"])
    recorded = result.get("preview_commands")
    if recorded is None:
        recorded = json.loads((attempt / "commands.json").read_text(encoding="utf-8"))
    # Reuse the exact isolated ports and run duration from this evaluation.
    # New playback does not overwrite its metrics or tracking.npz.
    def flag_value(command, flag):
        return command[command.index(flag) + 1]

    deploy += ["--controller-config", str(attempt / "controller.yaml"),
               "--publish-reference", "--auto-start", "--max-policy-steps",
               flag_value(recorded["deploy"], "--max-policy-steps")]
    sim = [PYTHON, str(ROOT / "src/sim2sim.py"), "--robot", "g1",
           "--bridge-config", str(attempt / "bridge.yaml"),
           "--xml_path", str(ROOT / "config/g1/assets/g1.xml"),
           "--auto-start", "--max-control-seconds",
           flag_value(recorded["sim"], "--max-control-seconds")]
    return result, sim, deploy


def stop_child(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", nargs="?", type=int, help="生成组 sample ID 0–2575；--baseline 为 PKL index 0–2575")
    parser.add_argument("--baseline", action="store_true", help="仿真 gen_all2576_dar0908_512.pkl 的动作")
    parser.add_argument("--gt", action="store_true", help="HML3D-G1 真正 GT；使用六位原始文件编号")
    parser.add_argument("--slow", action="store_true", help="使用 GT 独立降速副本，筛选规则不变")
    parser.add_argument("--refined", action="store_true", help="使用 005225 独立调整版；真机需连续三次完整通过")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sim", action="store_true", help="自动启动 MuJoCo 窗口并播放一次（默认）")
    modes.add_argument("--hardware", action="store_true", help="仅启动真机控制器；需先手动启动 bridge")
    parser.add_argument("--dry-run", action="store_true", help="检查标签和路径、打印命令，不启动进程")
    parser.add_argument("--force", action="store_true", help="真机模式强行放行 REVIEW 动作（绕过硬件初筛，风险自负）")
    parser.add_argument("--prepare-only", action="store_true", help="仅准备动作转换文件，不启动仿真或控制器")
    parser.add_argument("--headless", action="store_true", help="仿真不显示窗口")
    args = parser.parse_args()
    if args.refined and (not args.gt or args.slow):
        parser.error("--refined 仅用于 --gt，且不能与 --slow 同时使用。")
    if args.slow and not args.gt:
        parser.error("--slow 仅用于 --gt；可用 bash run_gt.sh 005225 --slow --sim。")
    if args.gt and args.baseline:
        parser.error("--gt 与 --baseline 不能同时使用。")
    if args.hardware and (args.baseline or args.headless or args.prepare_only):
        parser.error("--baseline、--headless、--prepare-only 仅用于仿真。")
    if args.force and not args.hardware:
        parser.error("--force 仅用于 --hardware。")
    if args.index is None:
        if not sys.stdin.isatty():
            parser.error("请提供 index，例如：bash run_generated.sh 21 --sim")
        try:
            scope = ("GT 文件编号，例如 005225" if args.gt else
                     "baseline PKL 0–2575" if args.baseline else "生成动作 0–2575（真机仅原候选）")
            args.index = int(input(f"请输入 index（{scope}）：").strip())
        except (ValueError, EOFError):
            parser.error("index 必须是整数。")
    try:
        result, sim_command, deploy_command = build_commands(
            args.index, args.hardware, baseline=args.baseline, dry_run=args.dry_run, gt=args.gt, slow=args.slow, refined=args.refined, force=args.force)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"[run_generated] {exc}\n")
    print("数据源：" + ("HML3D-G1 (GT, non-M)" if args.gt else
                      "gen_all2576_dar0908_512.pkl (baseline)" if args.baseline
                      else "gen_test2576_dar0911_512 (generated)"), flush=True)
    if args.gt:
        print(f"GT ID={result['gt_id']}；相同 prompt 的生成组 index={result['generated_index']}", flush=True)
        if args.refined:
            print(f"独立调整版：{result['variant']}，调整参数={result['adjustment']}；"
                  f"完整确认次数={result.get('confirmed_runs', 0)}。", flush=True)
        if args.slow:
            print(f"独立降速副本：原速的 {result['playback_speed']:.3f} 倍；"
                  f"动作时长 {result['stretched_duration_s']:.2f}s；"
                  f"时长放大 {result['slowdown_factor']} 倍。", flush=True)
    print(f"index={args.index:04d}  src_idx={result.get('src_idx')}  "
          f"{result.get('rating')} / {result.get('hardware_screen')}", flush=True)
    print(result.get("text", ""), flush=True)
    print(f"Global MPJPE={result.get('global_mpjpe_mm')} mm; "
          f"Root-relative MPJPE={result.get('root_relative_mpjpe_mm')} mm", flush=True)
    print(f"motion={result['npz_file']}", flush=True)
    if args.headless and sim_command is not None:
        sim_command.append("--headless")
    if args.dry_run:
        if result.get("preview_not_prepared"):
            print("尚未转换；正式运行时会自动创建独立目录并转换。")
        if sim_command:
            print("SIM: " + shlex.join(sim_command))
        print("DEPLOY: " + shlex.join(deploy_command))
        return
    if args.prepare_only:
        print("动作文件已就绪，未启动仿真或控制器。", flush=True)
        return
    if args.hardware:
        print("真机模式：先确认 bridge 已收到 LowState；保持吊架支撑。"
              "手柄 START=默认站姿，A=播放，SELECT=锁存阻尼。"
              "终端 x=阻尼退出。", flush=True)
        os.chdir(ROOT)
        os.execv(PYTHON, deploy_command)

    print("仿真模式：" + ("无窗口" if args.headless else "自动打开窗口")
          + "并播放一次，结束后关闭；Ctrl+C 可中断。", flush=True)
    sim = deploy = None
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        sim = subprocess.Popen(sim_command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
        time.sleep(1)
        if sim.poll() is not None:
            raise RuntimeError("MuJoCo 启动失败，请检查上方日志、图形窗口及端口占用。")
        deploy = subprocess.Popen(deploy_command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
        while deploy.poll() is None:
            if sim.poll() is not None:
                # The simulator can finish immediately on the controller's
                # final damping packet, just before deploy finishes cleanup.
                try:
                    deploy.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("MuJoCo 已关闭；停止本次仿真控制器。")
                if sim.returncode:
                    raise RuntimeError(f"MuJoCo 异常退出：{sim.returncode}")
                break
            time.sleep(0.1)
        if deploy.returncode:
            raise RuntimeError(f"控制器异常退出：{deploy.returncode}")
        try:
            sim.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    except KeyboardInterrupt:
        print("\n停止本次仿真。", flush=True)
        raise SystemExit(130)
    except RuntimeError as exc:
        parser.exit(1, f"[run_generated] {exc}\n")
    finally:
        stop_child(deploy)
        stop_child(sim)


if __name__ == "__main__":
    main()
