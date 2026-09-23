"""Combine extracted-motion metadata and MuJoCo metrics into a report."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load_tiers(path: Path) -> dict[int, str]:
    tiers = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tier, index, *_ = line.split()
        tiers[int(index)] = tier
    return tiers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--tiers", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    metadata = {int(item["index"]): item for item in manifest}
    tiers = load_tiers(args.tiers)
    combined = []
    for result in results:
        index = int(result["index"])
        item = dict(metadata[index])
        item.update(result)
        item["hardware_tier"] = tiers.get(index)
        combined.append(item)
    combined.sort(key=lambda item: int(item["index"]))

    counts = dict(collections.Counter(item["rating"] for item in combined))
    payload = {"counts": counts, "motions": combined}
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tier1 = [item for item in combined if item.get("hardware_tier") == "tier1"]
    tier2 = [item for item in combined if item.get("hardware_tier") == "tier2"]
    falls = [item for item in combined if item["rating"] == "FALL"]
    marginal = [item for item in combined if item["rating"] == "MARGINAL"]
    sim_only_pass = [
        item
        for item in combined
        if item["rating"] == "PASS" and not item.get("hardware_tier")
    ]

    def ids(items: list[dict]) -> str:
        return " ".join(f"{int(item['index']):02d}" for item in items)

    lines = [
        "# motion_examples_100：仿真与真机运行手册",
        "",
        "> 这是本批 100 条动作的唯一操作入口，包含运行命令、按键顺序、真机白名单、",
        "> 明确不推荐名单以及逐条仿真指标。真机白名单只是保守的初筛结果，不是安全保证。",
        "",
        "## 1. 数据与转换结果",
        "",
        "源文件 `motion_examples_100.pkl` 包含 100 条 23DOF、30 Hz 轨迹，根四元数为",
        "`xyzw`。轨迹已转换为 GRIT 所需的 29DOF、50 Hz、根四元数 `wxyz` 的 NPZ；",
        "源数据缺失的 6 个腕关节填零。转换后的动作位于",
        "`sim2real/config/g1/motions/examples100/`。",
        "",
        "## 2. MuJoCo 仿真运行",
        "",
        "每次测试一个动作时，打开两个终端。终端 1 启动仿真：",
        "",
        "```bash",
        "cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real",
        "./run_sim.sh",
        "```",
        "",
        "终端 2 启动控制器。把 `96` 换成 `00` 到 `99` 中的动作编号：",
        "",
        "```bash",
        "cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real",
        "./run_deploy_example100.sh 96",
        "```",
        "",
        "终端 2 的按键顺序：",
        "",
        "1. 等待仿真窗口和控制器均已启动。",
        "2. 按 `s`，进入默认站姿；等机器人完全稳定。",
        "3. 按 `a`，从头播放所选动作。",
        "4. 按 `s` 可回默认站姿；按 `x` 进入阻尼并退出控制器。",
        "",
        "切换动作时重启仿真和控制器，以保证初始状态一致。不要同时启动两个",
        "`run_sim.sh`，否则 UDP 端口会冲突。",
        "",
        "## 3. G1 真机运行",
        "",
        "### 3.1 安全前提",
        "",
        "- 首次真机测试必须使用吊架或可靠的物理支撑，周围至少 3 m 净空。",
        "- 必须由受训操作员站在动作范围外掌握物理急停。软件按键不能代替物理急停。",
        "- 不得同时运行 MuJoCo bridge 和真机 bridge。先测 tier1，再考虑 tier2。",
        "- 若默认站姿不稳、明显振荡、脚底打滑或姿态持续偏离，立即物理急停，不要按 `a`。",
        "",
        "### 3.2 首次编译",
        "",
        "```bash",
        "cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/g1_sim2real",
        "bash scripts/build.sh",
        "```",
        "",
        "### 3.3 每次运行",
        "",
        "2026-09-12 实机检查确认：`eth0` 是 G1 有线接口，`eth4` 是上网接口。WSL",
        "重启后静态地址可能丢失，每次先检查并在缺失时配置 `192.168.123.222/24`；",
        "必须确认到 G1 运控 PC `192.168.123.161` 的路由走 `eth0` 且 ping 无丢包：",
        "",
        "```bash",
        "ip -br -4 addr show dev eth0",
        "sudo ip addr add 192.168.123.222/24 dev eth0  # 仅在该地址缺失时执行",
        "# 若 eth0 同时存在 169.254.x.x，必须删除，避免 CycloneDDS 绑定错误地址",
        "sudo ip addr del 169.254.81.104/16 dev eth0  # 按实际显示的 169.254 地址填写",
        "sudo ip link set eth0 up",
        "ip route get 192.168.123.161",
        "ping -I eth0 -c 3 192.168.123.161",
        "```",
        "",
        "机器人重启后，先用宇树手柄进入 G1 的 Debug/低层控制状态（按项目当前固件的",
        "组合键 `L2+R2`、`L2+A`、`L2+B`），再启动 bridge。若手柄组合键在你的固件中",
        "不同，以宇树控制器界面显示为准；此时不要按 START/A。",
        "",
        "Bridge 会把 Python 约 50 Hz 的最新指令缓存，并以 G1 低层控制频率 500 Hz",
        "重复发布到 `rt/lowcmd`；日志中的 `command=~50 Hz` 是 Python 更新率，",
        "应同时看到 `lowcmd_publish=~500 Hz`。启动 bridge 后可用 `ss -uapn` 检查 DDS 单播套接字：它应绑定",
        "`192.168.123.222`，不能绑定 `169.254.x.x`。启动脚本会忽略继承的",
        "`CYCLONEDDS_URI`，以 `G1_NET=eth0` 的 SDK 网卡选择为准。修改网卡地址后必须重启 bridge。",
        "",
        "终端 1 启动真机 bridge：",
        "",
        "```bash",
        "cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/g1_sim2real",
        "G1_NET=eth0 taskset -c 2-3 bash scripts/run_bridge.sh",
        "```",
        "",
        "终端 2 启动 GRIT 控制器（示例为 tier1 的动作 96）：",
        "",
        "```bash",
        "cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real",
        "taskset -c 4-7 ./run_deploy_example100.sh 96 --hardware",
        "```",
        "",
        "启动后的正常日志顺序是：Bridge 先连接 DDS LowState，控制器再发送安全零力矩",
        "包，Bridge 显示 `Waiting for G1 gamepad START before releasing native motion service`。",
        "此时原生运控尚未释放，机器人不会因为等待 START 而提前进入阻尼；按 START 后",
        "Bridge 才释放原生运控，控制器开始 5 秒默认站姿插值。若 Bridge 仍停在",
        "`Waiting for DDS lowstate`，不要启动控制器，也不要按 START，应先修复网卡/DDS。",
        "",
        "`--hardware` 会拒绝不在下方白名单中的编号。真机正常操作使用宇树无线手柄：",
        "",
        "1. **START：只进入默认站姿。**从机器人当前实测关节角出发，用 5 秒",
        "   minimum-jerk 曲线缓慢进入 `default_qpos`。首帧从 10% Kp/阻尼 Kd=8 起步，",
        "   Kp/Kd 与位置曲线同步渐变；插值两端速度和加速度均为零。完成后继续",
        "   保持默认站姿，不会自动启动动作。",
        "2. 默认站姿完全稳定后，短按 **A：只启动动作。**以当前默认参考姿态为起点，",
        "   用 5 秒 minimum-jerk 参考插值进入轨迹第一帧，随后播放原始轨迹。A 键",
        "   不会重新抓取机器人实测姿态，也不会重复执行 START 的起身流程。",
        "3. **SELECT：低层锁存急停。**C++ bridge 在约 50 Hz 的状态链路上直接检测，",
        "   立即拒绝后续策略指令，并持续发布 `Kp=0、Kd=8` 的阻尼命令。",
        "4. SELECT 触发后没有手柄复位组合键。松开 SELECT、停止 Python 控制器，并重启",
        "   bridge 才能清除锁存；不得自动恢复位置控制。",
        "",
        "键盘 `s`、`a` 可分别作为 START、A 的调试备用；正常结束可在控制器终端按",
        "`x`，再到 bridge 终端按 `q`。SELECT/阻尼会撤销位置保持，机器人可能变软并",
        "下落，所以吊架不能省略；它也不能替代机器人独立的物理急停。",
        "",
        "两个插值时间可在以下配置中调整：",
        "",
        "- 起身时间：`sim2real/config/g1/controller.yaml` 的 `startup_interpolation_s`。",
        "- 起身初始 Kp 比例：同一文件的 `startup_kp_floor`（当前为 0.10）。",
        "- 轨迹首帧时间：`sim2real/config/g1/tracking.yaml` 的",
        "  `motion_source.npz.first_frame_transition_s`。",
        "",
        "## 4. 哪些动作推荐真机",
        "",
        f"批量 MuJoCo 结果：**PASS {counts.get('PASS', 0)}，MARGINAL {counts.get('MARGINAL', 0)}，"
        f"FALL {counts.get('FALL', 0)}，ERROR {counts.get('ERROR', 0)}**。",
        "",
        "### Tier 1：优先真机验证",
        "",
        "这些动作以站立和上肢动作为主，但仍必须使用吊架：",
        "",
    ]
    for item in tier1:
        lines.append(f"- `{int(item['index']):02d}` — {item['description']}")
    lines.extend(
        [
            "",
            "### Tier 2：Tier 1 全部稳定后再验证",
            "",
            "这些动作含迈步或行走，风险高于 tier1：",
            "",
        ]
    )
    for item in tier2:
        lines.append(f"- `{int(item['index']):02d}` — {item['description']}")
    lines.extend(
        [
            "",
            "## 5. 哪些动作目前不推荐真机",
            "",
            f"- **仿真摔倒，禁止首轮真机：** `{ids(falls)}`",
            f"- **仿真边缘稳定，不推荐真机：** `{ids(marginal)}`",
            f"- **仿真 PASS，但未进入保守真机白名单：** `{ids(sim_only_pass)}`",
            "",
            "最后一类并不等于动作一定危险，只表示目前证据不足，不应跳过单独复核和渐进式",
            "安全验证。若以后要放行某条动作，应重新检查完整仿真、足端接触、关节速度/力矩",
            "和真机限位，再明确加入 `hardware_candidates.txt`；不要绕过 `--hardware` 检查。",
            "",
            "## 6. 评级含义",
            "",
            "- `PASS`：批量仿真中根高度最低值不小于 0.60 m，且最大倾角不超过 35°。",
            "- `FALL`：根高度低于 0.45 m，或最大倾角超过 60°。",
            "- `MARGINAL`：介于上述条件之间。",
            "- `tier1/tier2`：从 PASS 中进一步人工保守筛选出的初测顺序，不是安全认证。",
            "",
            "## 7. 100 条动作详细结果",
            "",
            "| ID | 仿真评级 | 真机级别 | 最低根高度 (m) | 最大倾角 (deg) | 关节 RMSE (rad) | 源轨迹最大速度 (rad/s) | 动作描述 |",
            "|---:|:---:|:---:|---:|---:|---:|---:|:---|",
        ]
    )
    for item in combined:
        description = str(item["description"]).replace("|", "\\|")
        lines.append(
            f"| {int(item['index']):02d} | {item['rating']} | "
            f"{item.get('hardware_tier') or '-'} | {item['root_z_min']:.3f} | "
            f"{item['root_tilt_max_deg']:.1f} | "
            f"{item['reference_joint_rmse_rad']:.3f} | "
            f"{item['max_joint_speed_rad_s']:.2f} | {description} |"
        )
    lines.extend(
        [
            "",
            "## 8. 相关文件",
            "",
            "- 真机白名单：`sim2real/config/g1/motions/examples100/hardware_candidates.txt`",
            "- 机器可读结果：`motion_examples_100_validation.json`",
            "- 单动作运行器：`sim2real/run_deploy_example100.sh`",
        ]
    )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
