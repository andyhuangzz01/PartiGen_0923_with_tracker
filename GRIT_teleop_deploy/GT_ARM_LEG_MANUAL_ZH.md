# HML3D-G1：6 条手脚配合 GT 动作

本批从本地 `HML3D-G1.tar.zst` 仅提取 6 个非 M 版本的动作和对应文本。
不是 dar0911 生成动作，也不是之前 dar0908 baseline。按完全相同 caption 找到对应项；
这不证明生成组采用了非 M 版本作为逐帧参考。包内没有划分文件，暂不认证为 val。

## 编号与文件

| GT 文件编号 | 同 prompt 的生成组 index | 动作 |
| --- | --- | --- |
| 005225 | 1183 | 左右迈步、伸臂挥手 |
| 004558 | 222 | 来回走路、双臂画圈 |
| 001987 | 1804 | 举双手向前走 |
| 012511 | 307 | 小范围绕圈滑步跳舞 |
| 002248 | 1587 | 右手投掷、向前移动 |
| 005585 | 950 | 夸张摆臂、往返走路 |

文件相对本手册目录的位置（以 005225 为例）：

- 原始 GT：`gt_hml3d_arm_leg/source/motion/005225.pkl`
- 原文：`gt_hml3d_arm_leg/source/texts/005225.txt`
- GRIT 动作：`gt_hml3d_arm_leg/prepared/005225/motion.npz`
- 转换审计：`gt_hml3d_arm_leg/prepared/005225/prepared.json`
- 最新筛选结果：`gt_hml3d_arm_leg/prepared/005225/latest_evaluation.json`
- 每次筛选的日志、MPJPE、逐帧数据：对应 `eval_*/` 目录

转换保留 GT 的全部 29 个关节（包含手腕），未裁帧、未平滑、未降速或更改根高度；
读取 GT 的 xyzw 四元数并转换成 GRIT wxyz，30 Hz 重采样到 50 Hz。
利用 GT 自带 `local_body_pos` 在所有源帧上核对各关节对应身体的 FK 后才允许转换。
模型、策略、配置、控制代码或源数据变化时拒绝复用旧筛选结论。

## 一键 MuJoCo

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
bash run_gt.sh 005225 --sim
```

替换六位 GT 编号即可，不要填生成组 index。也可 `bash run_gt.sh --sim` 后按提示输入。
自动启动两个本机仿真进程；默认站姿过渡 5 秒、插值到首帧 5 秒、整段播放、回默认后退出。
Ctrl+C 中断。一次运行一个，不需要真机 bridge。仿真端口在 65000–65011。

仅查看路径/命令：`bash run_gt.sh 005225 --sim --dry-run`。
无窗口：`bash run_gt.sh 005225 --sim --headless`。

## 真机入口与限制

当前原速 GT 的源关节速度均超过既有保守筛选阈值，因此不能仅因它是真实 GT 就直接放行。
`--hardware` 仅接受无错误、完整记录、PASS 且 CANDIDATE_TIER1/2 的筛选结果。
REVIEW/FALL/MARGINAL/未评估均会在启动控制进程前拒绝；没有绕过开关。
阈值是本项目的保守初筛规则，不是厂商认证，也不是物理极限。

检查当前拒绝原因（不启动机器人）：

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
bash run_gt.sh 005225 --hardware --dry-run
```

只有后续动作版本通过相同筛选、操作员完成吊架/物理急停/净空/网络及 GUI 回放检查后，
才使用以下既有接口。不要直接调用 deploy.py 绕过筛选。

终端 1（现有 bridge）：

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/g1_sim2real
G1_NET=eth0 taskset -c 2-3 bash scripts/run_bridge.sh
```

终端 2（当前这批原速动作会被拒绝）：

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
bash run_gt.sh 005225 --hardware
```

真机命令不使用 auto-start，不自动启动 bridge，不发布仿真参考包。
START：实测关节角到默认站姿，5 秒。站稳后短按 A：5 秒插值到首帧，再播放一次。
SELECT：现有 C++ bridge 锁存阻尼急停，持续 Kp=0、Kd=8 并拒收策略控制，需重启 bridge 清除。
阻尼不是位置锁定，必须有物理支撑和物理急停；不能把 Python 退出或网络超时等同于 SELECT。
现有 bridge 的命令超时只告警并保留最后指令，不提供自动断网阻尼保障。

本次只执行本机转换与仿真，未启动真机 bridge 或真机控制器。

## 本次短仿真结果（2026-09-15）

以下 MPJPE 是 GRIT 执行轨迹相对各自 GT 参考的闭环跟踪误差，
不是 dar0911 生成轨迹与 GT 的逐帧误差。单位 mm。

| GT ID | 稳定性/初筛 | 完整帧 | Global MPJPE | Root-relative MPJPE |
| --- | --- | --- | --- | --- |
| 005225 | PASS / REVIEW | 331/331 | 270.38 | 33.11 |
| 004558 | PASS / REVIEW | 331/331 | 845.32 | 95.10 |
| 001987 | PASS / REVIEW | 246/246 | 255.72 | 41.89 |
| 012511 | PASS / REVIEW | 331/331 | 173.10 | 49.94 |
| 002248 | PASS / REVIEW | 197/197 | 161.67 | 38.34 |
| 005585 | ERROR / NOT_RECOMMENDED | 330/331 | 不采纳 | 不采纳 |

001987 首轮帧记录不完整，单独重跑后完整；005585 单独重跑后仍缺参考帧 63，
因此保留错误状态，不把部分帧 MPJPE 当成有效结果。两次日志均保留在各自 eval 目录。
其余 5 条虽满足既有宽松稳定性 PASS 条件，仍未通过真机初筛。
6 条原速源关节速度峰值约 11.0–55.4 rad/s，超过既有 4 rad/s 保守筛选阈值；
部分还超过实际关节速度、关节跟踪误差或倾角的保守阈值。
当前 6 条真机入口均已验证会拒绝，未添加绕过开关。

转换、GT 路由和原有控制契约测试共 42 项通过。未进行 GUI 目视评估或真机验证。
