# 生成动作 1–500：G1 真机操作手册

适用批次：`TextOpRobotMDAR/gen_test2576_dar0911_512`，已评估的 sample index 为 **1–500**。
本手册参考 [旧 100 条动作验证文件](motion_examples_100_validation.json) 的逐动作指标组织方式，
使用[本批评估结果](gen_test2576_dar0911_mpjpe_0001_0500/summary.json)确定候选，不沿用旧批次编号或白名单。

本批候选共 **56 条：Tier 1 为 50 条，Tier 2 为 6 条**。候选表示通过既有仿真初筛，
仍需吊架下逐条实机验证。MPJPE 未另设放行阈值；它表示对生成参考的闭环跟踪误差，不验证动作是否符合 prompt。

## 1. 直接使用的入口

- 真机 bridge：[g1_sim2real/scripts/run_bridge.sh](g1_sim2real/scripts/run_bridge.sh)，执行现有 `g1_udp_bridge`。
- 按编号选动作：[sim2real/run_generated.sh](sim2real/run_generated.sh)，读取本批 `INDEX/result.json` 的 `npz_file`。
- 运控程序：现有 [sim2real/src/deploy.py](sim2real/src/deploy.py)，使用原 GRIT ONNX 和 `tracking.yaml`。
- 不使用 `run_deploy_example100.sh` 选择本批动作：旧 index 21 和新 index 21 不是同一条动作。

`--hardware` 只启动控制器，不自动启动 bridge；仅接受无 ERROR、完整记录、PASS 且标签为
`CANDIDATE_TIER1/2` 的动作。启动前会显示 index、src_idx、描述、标签、MPJPE 和实际文件路径。
真机分支不添加 `--auto-start`，等待操作员按 START/A。

## 2. 每次运行前

1. 使用吊架或可靠物理支撑，周围至少 3 m 净空，受训操作员掌握物理急停。
2. 先仿真复核完全相同的动作，再做实机。首次实机先选 Tier 1；默认站姿未稳定时不按 A。
3. 退出上一轮 MuJoCo、仿真控制器或重复的真机控制器；每次只运行一个真机 bridge 和一个控制器。
4. 确认机器人处于对应固件要求的 Debug/低层控制状态。不要仅根据固定手柄组合键判断状态，以机器人界面为准。
5. 接好 G1 有线网卡，按下面命令核对网络。不要把上网接口当作机器人接口。

本机既有配置使用 `eth0`、地址 `192.168.123.222/24`，机器人运控 PC 为 `192.168.123.161`。
每次检查实际状态，不假设 WSL 重启后仍然可用：

```bash
ip -br -4 addr show dev eth0
ip route get 192.168.123.161
ping -I eth0 -c 3 192.168.123.161
```

路由应包含 `dev eth0 src 192.168.123.222`。若接口为 DOWN，接线后执行：

```bash
sudo ip link set eth0 up
```

仅当地址确实缺失时添加：

```bash
sudo ip addr add 192.168.123.222/24 dev eth0
```

若同时出现 `169.254.x.x` 地址，按实际地址排除冲突再启动 bridge；现有启动脚本会检测并拒绝这种混用。
网卡名不同时替换 `G1_NET`，路由和 ping 必须同时使用实际机器人网卡。

## 3. 先按 index 运行仿真

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
bash run_generated.sh 21 --sim
```

自动打开 MuJoCo 窗口，完成默认站姿、5 秒首帧插值、整段动作、回默认和尾部观察后退出。
可用 Ctrl+C 中断本次仿真。仿真使用评估时的隔离 UDP 端口，不覆盖原评估文件。
如果窗口启动失败，先处理图形环境或端口冲突。

只查看标签和最终命令，不启动任何控制进程：

```bash
bash run_generated.sh 21 --hardware --dry-run
```

也可不带 index，按提示输入：

```bash
bash run_generated.sh --hardware --dry-run
```

## 4. 真机启动：两个终端

### 终端 1：原生 bridge

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/g1_sim2real
G1_NET=eth0 taskset -c 2-3 bash scripts/run_bridge.sh
```

当前已有编译产物。仅在二进制缺失或桥接源码更新后重新编译：

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/g1_sim2real
bash scripts/build.sh
```

先确认 bridge 已收到 DDS LowState。若停在 `Waiting for DDS lowstate`，不要继续 START/A，
先修复机器人状态、网卡或 DDS 通信。

### 终端 2：按编号启动现有控制器

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
bash run_generated.sh 21 --hardware
```

将 `21` 改为本手册候选编号即可；也可以运行 `bash run_generated.sh --hardware` 后输入编号。
无需手填 `attempt_时间戳`，脚本自动定位本条实际 NPZ。

正常启动后 bridge 会等待有效 Python 指令，再显示：

```text
Waiting for G1 gamepad START before releasing native motion service
```

此时使用下方手柄流程。当前 bridge 的 Python 指令更新约 50 Hz，低层发布约 500 Hz；
这两个频率含义不同。

## 5. 必须保留的两段插值

| 操作 | 过程 | 当前配置 |
|---|---|---|
| 按 START | 当前实测关节角 → 默认站姿 | 5 秒，minimum-jerk，增益同步渐变 |
| 默认站姿稳定后短按 A | 当前默认参考 → 动作第 0 帧 | **5 秒、250 个参考步、minimum-jerk** |
| 首帧插值完成 | 从动作第 0 帧开始完整播放 | 单次播放，不循环 |
| 动作完成 | 返回默认站姿并保持 | 返回插值 100 步，约 2 秒 |

START 的 5 秒起身插值和 A 的 5 秒首帧插值是**两段独立流程**。
A 不会重新执行起身过程；动作首帧插值沿现有参考流衔接。
不要连续按 A；再次按 A 会重新发起播放请求。
根四元数使用旋转插值，位置及关节参考使用 minimum-jerk 时间曲线。

[tracking.yaml](sim2real/config/g1/tracking.yaml) 中应保留：

```yaml
reference_fps: 50.0
transition_steps: 100
motion_source:
  type: npz
  npz:
    loop: false
    first_frame_transition_s: 5.0
```

以上仅展示相关字段，**不要用此片段覆盖整个配置文件**。
`transition_steps: 100` 是通用/返回插值；首帧专用的 5 秒由
`first_frame_transition_s` 覆盖，不能误以为 A 只有 2 秒。

[controller.yaml](sim2real/config/g1/controller.yaml) 中为：

```yaml
control_freq: 50
startup_interpolation_s: 5.0
startup_kp_floor: 0.10
```

启动日志应包含：

```text
first_frame_transition=250 steps/5.00s
```

按 A 后的追加参考日志应包含 `transition=250` 和 `easing=minimum_jerk`。
若配置/日志不符，退出并恢复上述配置后再运行。当前编号入口读取配置，不会自动把被修改的插值时间改回 5 秒。

实现入口：[LocalNpzMotionSource.play_from_start](sim2real/src/runtime/motion_sources.py)，
复用既有 GRIT 参考插值程序，不另写真机轨迹播放器。

## 6. SELECT 急停：低层锁存阻尼

**宇树手柄 SELECT → C++ bridge 锁存急停 → 拒绝后续策略命令 → 持续发布阻尼。**

当前 [g1_bridge.yaml](g1_sim2real/config/g1_bridge.yaml) 的阻尼增益为 `Kd=8.0`。
急停命令所有电机的 `Kp=0`、`Kd=8`、前馈力矩为零；这会撤销位置保持，
机器人可能变软下落，必须保持吊架支撑。它不是断电急停。

SELECT 检测来自 bridge 约 50 Hz 的机器人状态处理链路；在低层控制激活后触发立即发一次阻尼，
低层发布线程继续以配置的约 500 Hz 发布阻尼。这里是软件名义频率，不是实测最坏急停延迟保证。

应看到日志：

```text
G1 gamepad SELECT E-STOP LATCHED: rejecting policy commands and continuously publishing damping. Restart bridge to clear.
```

- 不依赖 Python 是否及时处理按键；C++ 接收命令及低层发布路径均检查锁存状态。
- 松开 SELECT、再按 START/A 均不会清除锁存，不会自动恢复位置控制。
- 急停后先保持物理支撑、停止 Python 控制器、退出 bridge；排除原因后松开 SELECT，
  重新启动 bridge 和控制器，再从 START 站姿检查开始。
- 正常结束：控制器终端按 `x` 发送阻尼退出，再在 bridge 终端按 `q` 退出。
  键盘 `x` 不应当被描述成与手柄 SELECT 完全相同的低层锁存机制。
- Python 超时与 SELECT 不同：当前 bridge 的 `command_timeout_s: 0.2` 超时逻辑只报警、
  保留 LowCmd 等待恢复，**不自动进入急停阻尼**。不要依赖拔网线或停止 Python 来替代 SELECT/物理急停。
- SELECT 仍依赖手柄信息到达 bridge；通信故障时使用独立物理急停。

实现见 [g1_udp_bridge.cpp](g1_sim2real/src/g1_udp_bridge.cpp) 的
`send_state_snapshot`、`latch_gamepad_estop`、`on_udp_command`、`command_writer_loop`。

## 7. 本批候选清单

编号是生成目录 `sample_XXXX` 的 index，`src_idx` 是原测试集索引，两者不能混用。
Global/Root-relative MPJPE 单位均为 mm，点集固定为 19 个 body/link 原点。
点击 index 查看原动作文件和 MP4；原 MP4 用于复核生成参考，不是闭环实际执行视频。

### Tier 1：优先复核（50 条）

| index | src_idx | Global (mm) | Root-relative (mm) | 最低根高 (m) | 最大倾角 (°) | 描述 |
|---|---:|---:|---:|---:|---:|---|
| [21](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0021) | 2232 | 24.35 | 13.02 | 0.754 | 12.12 | the person is acting like a human elephant. |
| [22](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0022) | 1718 | 42.66 | 27.31 | 0.754 | 3.82 | a person holds an object with both of their hands. |
| [31](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0031) | 1138 | 35.43 | 28.42 | 0.754 | 7.50 | a man picks something up with both hands and then puts it back down. |
| [51](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0051) | 1481 | 24.62 | 13.44 | 0.754 | 2.79 | a person flattens their dress and starts to curtsey. |
| [56](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0056) | 933 | 25.06 | 17.16 | 0.754 | 6.80 | a body stands up and moves backward diagonally to the left. |
| [59](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0059) | 953 | 32.08 | 24.43 | 0.754 | 5.71 | a person is standing with their arms out to their sides and uses their left arm to make pulling motions, outwards, downwards, and inwards. |
| [62](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0062) | 2544 | 35.54 | 15.80 | 0.754 | 3.55 | this person standing still folds his arms in front then lowers his arms to the sides. |
| [71](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0071) | 2495 | 28.49 | 15.69 | 0.754 | 4.83 | the person  was pushed while standing up street. |
| [72](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0072) | 700 | 43.27 | 26.14 | 0.754 | 6.90 | looks as if its waiting for something hands on hips looking curious |
| [75](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0075) | 669 | 45.85 | 28.48 | 0.754 | 5.91 | the person looks around while slightly slouched over |
| [89](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0089) | 864 | 45.76 | 35.04 | 0.754 | 5.27 | a man stands, bends foward looking down and then moves his right hand trying to scratch his body. |
| [94](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0094) | 1620 | 32.78 | 26.32 | 0.754 | 6.91 | figure looks around then backs up |
| [104](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0104) | 1754 | 31.88 | 13.00 | 0.754 | 3.10 | a person raises his hands above his hands and waves. |
| [110](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0110) | 2087 | 32.69 | 23.21 | 0.758 | 4.52 | a man very slowly stretches his arms out. |
| [115](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0115) | 626 | 29.27 | 22.57 | 0.754 | 4.93 | the person threw some thing forward. |
| [118](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0118) | 2442 | 31.89 | 18.45 | 0.754 | 4.02 | walking quickly in a sideways pattern. |
| [127](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0127) | 47 | 31.36 | 22.21 | 0.754 | 6.74 | a person sways their upper body in a circle while they hold their hands as if they have a partner. |
| [134](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0134) | 1780 | 31.60 | 18.25 | 0.754 | 3.47 | a person stumbles to the right and recovers their balance. |
| [143](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0143) | 1222 | 29.54 | 15.52 | 0.754 | 3.90 | walking backwards and stopping. |
| [146](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0146) | 2460 | 33.83 | 16.95 | 0.754 | 4.12 | a person standing up folds arms by placing right arm over left. |
| [149](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0149) | 2209 | 46.58 | 30.44 | 0.754 | 5.42 | this person stands and suddenly steps to the right. |
| [161](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0161) | 2485 | 33.36 | 15.26 | 0.754 | 6.65 | a person takes one large side step to their right. |
| [171](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0171) | 676 | 42.96 | 23.29 | 0.754 | 4.66 | she is doing the dance to the song i'm a little teapot. |
| [213](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0213) | 1667 | 35.49 | 21.90 | 0.754 | 4.39 | the person standing with their legs crossed trying to balance. |
| [221](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0221) | 1912 | 40.99 | 35.08 | 0.754 | 3.63 | a person appears to put on a coat, they put their right hand in to a sleeve and then button up the front. |
| [228](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0228) | 1018 | 30.60 | 19.69 | 0.754 | 4.85 | a person swings his left arm low, then steps forward, turns away, and holds his arms out wide to either side. |
| [232](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0232) | 1837 | 31.89 | 15.65 | 0.754 | 3.44 | a person wipes in a circular movement with left hand. |
| [248](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0248) | 681 | 39.90 | 28.72 | 0.754 | 4.29 | walking to the side and waving arms. |
| [251](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0251) | 1971 | 24.02 | 14.03 | 0.754 | 4.68 | a person unzips their pants. |
| [257](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0257) | 8 | 28.10 | 13.66 | 0.754 | 3.46 | a person holding something up |
| [269](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0269) | 239 | 35.81 | 18.56 | 0.754 | 8.78 | a figure winds up then throws as hard as they can |
| [282](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0282) | 761 | 26.79 | 16.03 | 0.754 | 7.36 | the person is feeling his head like he is dizzy. |
| [309](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0309) | 2202 | 42.83 | 27.04 | 0.754 | 4.88 | a person bends over and touches something with his right hand. |
| [314](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0314) | 1429 | 39.16 | 26.71 | 0.754 | 4.04 | the person moves his right hand around |
| [326](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0326) | 424 | 52.48 | 32.70 | 0.754 | 4.61 | a person stands still and does not move. |
| [355](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0355) | 2336 | 41.34 | 32.24 | 0.754 | 7.13 | a man seems to have a spasm. |
| [357](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0357) | 147 | 36.43 | 14.77 | 0.754 | 3.41 | a person is standing still while waving his right hand. |
| [369](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0369) | 421 | 81.96 | 51.09 | 0.754 | 7.38 | person climbs up words while using their right hand to hold onto something. |
| [385](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0385) | 1566 | 30.28 | 20.84 | 0.754 | 6.21 | a man steps forward on his left foot and uses his left hand to brush off his upward facing right palm, then turns left to repeat the same motion. |
| [390](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0390) | 1885 | 36.11 | 14.83 | 0.754 | 9.61 | a person swinging both arms sideways |
| [396](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0396) | 791 | 36.92 | 20.81 | 0.754 | 16.46 | a person is standing in place slightly bent over and holding something, almost like a golf club |
| [404](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0404) | 1637 | 35.61 | 22.29 | 0.754 | 5.94 | the person is extending their arms straight out from their sides. they do this twice. |
| [417](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0417) | 2094 | 23.23 | 12.29 | 0.754 | 3.54 | the person is leaving something in circular motions. |
| [427](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0427) | 1491 | 40.72 | 19.79 | 0.754 | 3.92 | a person steps to his right. |
| [428](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0428) | 1766 | 41.02 | 25.73 | 0.754 | 6.10 | a person stands with arms at sides, feet angled out, and knees bent and holds the pose for a while before raising both hands out and stopping just above shoulder-height. |
| [444](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0444) | 712 | 44.83 | 26.34 | 0.754 | 3.88 | a person raised the both hand |
| [446](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0446) | 1663 | 28.48 | 14.17 | 0.754 | 3.30 | a man raises his hands in front of his face then lowers then back down. |
| [453](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0453) | 1904 | 34.25 | 14.12 | 0.754 | 3.32 | a person holds out their right hand. |
| [487](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0487) | 720 | 39.11 | 29.55 | 0.754 | 3.98 | a person swats at something |
| [499](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0499) | 1826 | 27.66 | 15.15 | 0.754 | 4.38 | a person raises their arms in front of them at shoulder hight with elbows to the sides. |

### Tier 2：下肢或位移较多（6 条）

| index | src_idx | Global (mm) | Root-relative (mm) | 最低根高 (m) | 最大倾角 (°) | 描述 |
|---|---:|---:|---:|---:|---:|---|
| [151](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0151) | 2 | 49.56 | 20.35 | 0.754 | 12.28 | a person standing upright and bending with both arms touching the floor and getting back up. |
| [156](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0156) | 1486 | 45.36 | 26.61 | 0.754 | 8.61 | a person is slowly moving around the room. |
| [158](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0158) | 980 | 44.56 | 21.63 | 0.754 | 9.63 | the person stands still but is pushed by something where his upper body moves slightly to his right before returning to his original stance. |
| [233](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0233) | 574 | 75.24 | 29.07 | 0.752 | 12.51 | a person walks with a slightly angled direction, and smooth hip motion. |
| [372](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0372) | 633 | 117.92 | 37.01 | 0.754 | 8.07 | she carefully sliced some of the zucchini and then backed away from the table. |
| [392](../TextOpRobotMDAR/gen_test2576_dar0911_512/sample_0392) | 1250 | 69.99 | 33.74 | 0.740 | 5.71 | a person stands on the spot with both their arms up by their chest |

## 8. 常见问题与验证范围

| 现象 | 处理 |
|---|---|
| `Motion file not found` | 使用按 index 入口；不同动作的 attempt 时间戳不同，不能只替换旧长路径中的编号 |
| `未进入本批真机候选` | 如 index 30 为 REVIEW，仅使用 `--sim` 复核；不要通过旧启动器规避本批筛选 |
| `Address already in use` | 检查旧仿真/bridge/控制器进程，确认所属后正常退出；不要盲目结束其他任务 |
| `Waiting for DDS lowstate` | 检查 G1 低层状态、网卡地址、eth0 路由与通信；先不按 START/A |
| 默认站姿振荡、脚底滑移、姿态偏离 | 保持物理支撑并急停检查，先不播放动作 |
| A 后直接跳变或首帧时长不符 | 检查 5 秒专用配置及日志，确认使用本手册入口和同一份 NPZ |

2026-09-15 本次整理已完成：

- 读取旧 100 条验证 JSON（100 条，PASS 83 / MARGINAL 9 / FALL 8），仅作手册结构参考。
- 核对本批 500 条结果及 56 条候选；完整指标见 [summary.csv](gen_test2576_dar0911_mpjpe_0001_0500/summary.csv)。
- 现有 GRIT 33 项契约单元测试通过，覆盖首帧 250 步、minimum-jerk、四元数插值及单次播放返回等行为。
- 读取当前配置，确认首帧 5 秒、起身 5 秒、50 Hz 参考、单次播放与阻尼 Kd=8。
- 检查 C++ SELECT 锁存和持续阻尼路径；现有 bridge 二进制含相应锁存日志标识。
- 本次未启动真机。上述是代码、配置及单元测试核对，不能替代吊架下的实体急停和首帧插值验证。

所有真机控制仍由原生 bridge 和原 GRIT 控制器执行。本手册及编号入口不调整权重、增益或筛选阈值。

