# 2576 条模型输出：MuJoCo 自动初筛

入口脚本：`sim2real/tools/screen_generated_motions.py`。
它运行真正的 GRIT ONNX 闭环控制和 MuJoCo 动力学，不是逐帧设置 qpos 的动画播放。
只使用本机回环 UDP 56000 起的隔离端口，不启动 DDS/真机 bridge，不修改硬件白名单。

## 1. 数据约定

已检查 `gen_all2576_dar0908_512.pkl`：2576 条、30 Hz、每条 74–322 帧。
原始动作总时长约 3.78 小时。`qpos` 为 `(T,30)`，但根四元数实际为 **xyzw**，
与内层 `motion.root_rot` 完全相同；脚本先转成 wxyz，再按已有转换器映射为
29DOF、50 Hz 的 GRIT NPZ，缺失的六个腕关节填零。不会添加根高度偏移或平滑动作。

- `index`：此 PKL 列表的零基编号，0–2575。
- `src_idx`：原测试集编号；与 index 不同，也与旧 examples100 编号无关。
- 原 PKL 保持不变。默认不删帧；若确认要剔除生成开头两帧，显式加 `--trim-start 2`。
- 开头跳变会影响速度指标；删帧版和原版必须分开输出目录，不能混用结论。
- PKL/joblib 可执行代码，只能读取可信来源文件。

## 2. 先跑两条验证环境

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
uv sync
.venv/bin/python tools/screen_generated_motions.py \
  --indices 0,2 --parallel 1 --output ../gen_all2576_my_smoke
```

首次依赖安装需要联网；已安装依赖时直接使用 `.venv/bin/python`。
环境必须允许本机 UDP 套接字，否则会记录 `PermissionError`，不能据此判断动作质量。

## 3. 全量运行与断点续跑

保留全部原始帧的版本：

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
.venv/bin/python tools/screen_generated_motions.py \
  --parallel 1 --output ../gen_all2576_screen
```

若决定沿用之前“删除开头两帧”的处理，改用独立目录：

```bash
.venv/bin/python tools/screen_generated_motions.py \
  --trim-start 2 --parallel 1 --output ../gen_all2576_trim2_screen
```

每完成一条就保存结果并刷新报告。中断后恢复（这里以保留原帧版为例）：

```bash
.venv/bin/python tools/screen_generated_motions.py \
  --parallel 1 --output ../gen_all2576_screen --resume --retry-errors
```

删帧版恢复时仍需 `--trim-start 2`。`--resume` 跳过已有结果；
`--retry-errors` 重跑超时、通信失败、被中断等 ERROR，不重复成功完成的动作。
模型、源文件、配置、脚本或主要运行库版本发生变化时，脚本拒绝复用旧结果，请换目录。

批测时不要操作真实机器人。这里的 Ctrl+C 只停止脚本创建的仿真/控制器子进程，
不能把它当作真机安全退出流程。不要同时启动另一个使用相同端口段的批测。

默认单路，避免多个 ONNX 控制器争用当前固定的 CPU 4–7。可以试 `--parallel 2`，
但若出现丢包/仿真时钟不匹配，应单路重测；更多并发不保证更快。
按两条试跑约 22 秒/条估算，全量单路约 16 小时，实际取决于动作长度和机器负载。
这是一次名义环境测试，不是多随机种子或扰动鲁棒性测试。没有自动替你启动整夜全量任务。

## 4. 输出和推荐如何看

输出目录包含：

- `report.md`：各级编号名单、逐条原因与动作描述。
- `summary.csv`：可用表格软件排序、筛选的指标。
- `summary.json`：完整机器可读结果。
- `run.json`：源文件、模型、代码/配置指纹和运行库版本。
- `0002/result.json`：该编号最新一次结果。
- `0002/attempt_.../`：转换后 qpos/NPZ、仿真/控制器配置、命令、日志和指标。
  重试创建新 attempt，旧日志不覆盖。

| 真机初筛标签 | 含义 |
|---|---|
| CANDIDATE_TIER1 | 所有保守阈值通过、位移和下肢变化较少：优先人工复核候选 |
| CANDIDATE_TIER2 | 所有阈值通过，但位移或下肢变化较多：次级候选 |
| REVIEW | 仿真未倒，但误差/速度等指标或动作描述需复核，暂不直接上真机 |
| NOT_RECOMMENDED | 跌倒代理指标触发、边缘稳定或测试失败：不推荐 |

仿真 PASS 仅表示最低根高度 ≥0.60 m、最大倾角 ≤35°，**不等于真机推荐**。
候选还要求：高度 ≥0.65 m、倾角 ≤20°、全身参考 RMSE ≤0.15 rad、
下肢参考 RMSE ≤0.12 rad、单关节最大参考误差 ≤0.6 rad、
源关节速度 ≤4 rad/s、实测仿真关节速度 ≤6 rad/s、全关节力矩饱和比例 ≤1%、
源关节超模型限位 ≤0.01 rad。阈值是保守经验筛选值，不是宇树官方硬件限制。
跑跳、跪坐等高风险文字描述另行转入人工复核，不能依靠文字证明安全。

测试覆盖当前配置的首帧插值（5 秒）、整段动作、回默认插值（2 秒）、
3 秒稳定观察和额外 1 秒缓冲。起身阶段另外运行，不计入这段高层策略时间。
检查播放、回默认和正常结束日志；未完整运行、无参考数据、明显时钟漂移或丢包
都不会被标成 PASS。统计包含全过程，不会把结束后的摔倒过滤掉。

局限：仿真起始姿态有固定支撑阶段，不等价于真机从任意坐地姿态起身。
未覆盖真实电池/温度、实机限位、足底滑移、自碰撞、根位置跟踪、通信失联恢复和扰动。
低位蹲姿也可能触发保守“跌倒”代理阈值，需要看回放确认。
候选仍需逐条回放、检查实机限制、吊架保护下由受训操作员渐进验证。
脚本不会自动加入 `examples100/hardware_candidates.txt`，不要把新编号传给旧 100 条启动器。

## 5. 单条有窗口复核

以全量输出中的 index 2 为例：

```bash
.venv/bin/python tools/screen_generated_motions.py \
  --output ../gen_all2576_screen --replay-index 2
```

它只打印两个终端的**仿真命令**，不会启动机器人。分别复制执行：先 sim，再 deploy。
两边启动后，在 deploy 终端按 `s` 进入默认姿态，稳定后按 `a` 播放；`x` 阻尼退出。
回放采用该次测试的端口与配置，不能与对应编号的批测同时运行。
若某条在转换/启动前已经 ERROR，没有命令文件，先查看该条 `result.json` 和转换日志。

## 6. 已完成的小批验证

保留原帧的首轮 index 0、2 均完整跑完：仿真 PASS，但都是 REVIEW，暂不直接推荐真机。
index 0 的实测关节峰值速度约 19.81 rad/s，index 2 约 10.50 rad/s，超过初筛阈值。
这两条的结果不能外推到剩余 2574 条；最终名单以全量运行生成的 `report.md` 为准。

另做了 `--trim-start 2 --parallel 2` 测试，结果保存在 `gen_all2576_trim2_smoke/report.md`：

| index | src_idx | 删两帧后结果 | 实测峰值关节速度 |
|---|---|---|---|
| 0 | 456 | PASS / REVIEW，仍不直接推荐真机 | 22.64 rad/s |
| 2 | 1126 | PASS / CANDIDATE_TIER2，需人工复核的次级候选 | 2.03 rad/s |

index 2 的源轨迹峰值关节速度也从 36.85 降为 3.18 rad/s，说明开头帧跳变对筛选影响明显。
本次还验证了断点恢复会跳过两条已有结果；36 项单元测试通过。
小批结果仅覆盖这些版本和配置，不能把候选解释为已通过真机验证。
