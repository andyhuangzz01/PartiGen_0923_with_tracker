# 005225 独立调整版：真机候选操作说明

这是从 GT 005225 派生的 **gentle90 调整版**，不是原始 GT，也不是仅降速的版本。
2026-09-15，同一 NPZ 最近连续 3 次完整通过原有筛选，标签为 CANDIDATE_TIER2。
这是项目仿真初筛，不是厂商认证或实机安全保证。首次仍须防跌保护、净空、受训操作员和物理急停。

## 使用哪一个版本

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
bash run_gt.sh 005225 --refined --sim
```

**使用 `--refined`，不要同时加 `--slow`。**
先在 MuJoCo 窗口目视检查，播放结束或 Ctrl+C 退出后，再做真机。
本次自动验证是无窗口仿真，没有替代操作员目视检查。

实际文件：

`gt_005225_refinement/variants/gentle90_baa009b8cbc3/motion.npz`

从原 GT 的独立调整内容：

- 原速的 1/4，动作正文约 26.4 秒，50 Hz、1321 帧。
- 下肢/腰部关节相对默认站姿的幅度保留 90%，水平根位移幅度保留 90%。
- 骨盆参考 roll/pitch 保留 80%，yaw 不缩小。
- 左手腕 pitch 保留 60%；其余手臂关节参考不变，29 个关节全部保留。
- 每个源帧保持左右脚尖中较低者的参考高度，根高度最大补偿约 4.8 mm。
- 没有裁帧、修改策略或降低筛选阈值；原始 GT、原速版和单纯降速版不变。

## 真机启动

首先只检查绑定文件和资格，不发控制指令：

```bash
bash run_gt.sh 005225 --refined --hardware --dry-run
```

应显示 `gentle90`、`完整确认次数=3`、`PASS / CANDIDATE_TIER2`。
不要绕过此入口直接手填旧 NPZ；`--slow --hardware` 和原速真机命令仍应被拦截。

确认网络、机器人模式及防跌保护满足既有要求；保持单个 bridge、单个控制器。
如果已有 bridge 正常收到 LowState，不要再启动第二个。
若尚未启动，使用既有接口（网卡名需符合当前实际连接）：

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/g1_sim2real
G1_NET=eth0 taskset -c 2-3 bash scripts/run_bridge.sh
```

另一个终端，确认 bridge 收到 LowState 后：

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
bash run_gt.sh 005225 --refined --hardware
```

不使用 auto-start，不自动启动 bridge，不向真机发布仿真参考包。

1. START：5 秒插值到默认站姿。
2. 站姿稳定后短按 A：5 秒插值到动作第 0 帧，再完整播放一次。
3. SELECT：现有 C++ bridge 锁存阻尼并拒绝策略控制，需重启 bridge 清除。

SELECT 是阻尼，不是位置锁定，不替代物理支撑或物理急停。
网络超时/关闭 Python 不等同于 SELECT；既有 bridge 超时策略仍是告警并保留最后指令。
本次没有启动真机控制器，也没有停止或修改用户已有的 bridge。

## 逐帧诊断和确认依据

新增独立观察入口 `sim2sim_diagnostics.py`，不修改原仿真动力学或既有指标实现。
每个 200 Hz 控制样本记录：阶段、动作帧号、仿真/控制时间、29 关节实测角度/速度、
关节参考/误差、根姿态/倾角。`diagnostics.json` 的峰值与原汇总指标严格交叉核对。
`tracking.npz` 仍记录按精确状态 ID 对齐的 50 Hz MPJPE 数据。

诊断复现定位到左手腕 pitch 在动作末尾（帧 1320）误差最大，
骨盆倾角峰值在正文约 9 秒、帧 451–453 附近。并非首帧插值问题。
初次诊断和调整版首测各有漏帧，因此它们没有用于放行；原日志保留。

调整版随后三次完整确认，全部 1321/1321 帧，且均通过原筛选：

| 次数 | 最大倾角 ° | 最大关节误差 rad | Global MPJPE mm | Root-relative MPJPE mm |
| --- | --- | --- | --- | --- |
| 1 | 17.308 | 0.487258 | 251.775 | 29.264 |
| 2 | 18.084 | 0.466221 | 250.373 | 23.923 |
| 3 | 17.317 | 0.487129 | 251.719 | 29.280 |

三次最坏实际关节速度约 2.284 rad/s，最坏下肢关节 RMSE 约 0.1111 rad。
各筛选指标按三次最坏值复核，不取最好的一次。
以上 MPJPE 是相对**调整后的参考**的闭环误差，不是与原始 GT 的误差。

确认清单：`gt_005225_refinement/selection.json`。
逐次结果/日志/诊断位于该版本的 `eval_*/`。动作、模型、策略及签名中记录的配置/代码变化会拦截；
出现新的评估或记录变化后也必须重新确认，不能沿用旧资格。
新增诊断、调整和确认逻辑与既有控制契约共 51 项测试通过；未进行真机验证。
