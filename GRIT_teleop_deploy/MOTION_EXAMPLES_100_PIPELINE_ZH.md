# motion_examples_100：仿真与真机运行手册

> 这是本批 100 条动作的唯一操作入口，包含运行命令、按键顺序、真机白名单、
> 明确不推荐名单以及逐条仿真指标。真机白名单只是保守的初筛结果，不是安全保证。

## 1. 数据与转换结果

源文件 `motion_examples_100.pkl` 包含 100 条 23DOF、30 Hz 轨迹，根四元数为
`xyzw`。轨迹已转换为 GRIT 所需的 29DOF、50 Hz、根四元数 `wxyz` 的 NPZ；
源数据缺失的 6 个腕关节填零。转换后的动作位于
`sim2real/config/g1/motions/examples100/`。

## 2. MuJoCo 仿真运行

每次测试一个动作时，打开两个终端。终端 1 启动仿真：

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
./run_sim.sh
```

终端 2 启动控制器。把 `96` 换成 `00` 到 `99` 中的动作编号：

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
./run_deploy_example100.sh 96
```

终端 2 的按键顺序：

1. 等待仿真窗口和控制器均已启动。
2. 按 `s`，进入默认站姿；等机器人完全稳定。
3. 按 `a`，从头播放所选动作。
4. 按 `s` 可回默认站姿；按 `x` 进入阻尼并退出控制器。

切换动作时重启仿真和控制器，以保证初始状态一致。不要同时启动两个
`run_sim.sh`，否则 UDP 端口会冲突。

## 3. G1 真机运行

### 3.1 安全前提

- 首次真机测试必须使用吊架或可靠的物理支撑，周围至少 3 m 净空。
- 必须由受训操作员站在动作范围外掌握物理急停。软件按键不能代替物理急停。
- 不得同时运行 MuJoCo bridge 和真机 bridge。先测 tier1，再考虑 tier2。
- 若默认站姿不稳、明显振荡、脚底打滑或姿态持续偏离，立即物理急停，不要按 `a`。

### 3.2 首次编译

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/g1_sim2real
bash scripts/build.sh
```

### 3.3 每次运行

2026-09-12 实机检查确认：`eth0` 是 G1 有线接口，`eth4` 是上网接口。WSL
重启后静态地址可能丢失，每次先检查并在缺失时配置 `192.168.123.222/24`；
必须确认到 G1 运控 PC `192.168.123.161` 的路由走 `eth0` 且 ping 无丢包：

```bash
ip -br -4 addr show dev eth0
sudo ip addr add 192.168.123.222/24 dev eth0  # 仅在该地址缺失时执行
# 若 eth0 同时存在 169.254.x.x，必须删除，避免 CycloneDDS 绑定错误地址
sudo ip addr del 169.254.81.104/16 dev eth0  # 按实际显示的 169.254 地址填写
sudo ip link set eth0 up
ip route get 192.168.123.161
ping -I eth0 -c 3 192.168.123.161
```

机器人重启后，先用宇树手柄进入 G1 的 Debug/低层控制状态（当前固件通常按
`L2+R2`、`L2+A`、`L2+B`），再启动 bridge；如果你的固件组合键不同，以宇树控制器界面为准。
此时不要按 START/A。

Bridge 会把 Python 约 50 Hz 的最新指令缓存，并以 G1 低层控制频率 500 Hz
重复发布到 `rt/lowcmd`；日志中的 `command=~50 Hz` 是 Python 更新率，应同时看到
`lowcmd_publish=~500 Hz`。启动 bridge 后可用 `ss -uapn` 检查 DDS 单播套接字：它应绑定
`192.168.123.222`，不能绑定 `169.254.x.x`。启动脚本会忽略继承的
`CYCLONEDDS_URI`，以 `G1_NET=eth0` 的 SDK 网卡选择为准。修改网卡地址后必须重启 bridge。

终端 1 启动真机 bridge：

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/g1_sim2real
G1_NET=eth0 taskset -c 2-3 bash scripts/run_bridge.sh
```

终端 2 启动 GRIT 控制器（示例为 tier1 的动作 96）：

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
taskset -c 4-7 ./run_deploy_example100.sh 96 --hardware
```

启动后的正常日志顺序是：Bridge 先连接 DDS LowState，控制器再发送安全零力矩
包，Bridge 显示 `Waiting for G1 gamepad START before releasing native motion service`。
此时原生运控尚未释放，机器人不会因为等待 START 而提前进入阻尼；按 START 后
Bridge 才释放原生运控，控制器开始 5 秒默认站姿插值。若 Bridge 仍停在
`Waiting for DDS lowstate`，不要启动控制器，也不要按 START，应先修复网卡/DDS。

`--hardware` 会拒绝不在下方白名单中的编号。真机正常操作使用宇树无线手柄：

1. **START：只进入默认站姿。**从机器人当前实测关节角出发，用 5 秒
   minimum-jerk 曲线缓慢进入 `default_qpos`。首帧从 10% Kp/阻尼 Kd=8 起步，
   Kp/Kd 与位置曲线同步渐变；插值两端速度和加速度均为零。完成后继续
   保持默认站姿，不会自动启动动作。
2. 默认站姿完全稳定后，短按 **A：只启动动作。**以当前默认参考姿态为起点，
   用 5 秒 minimum-jerk 参考插值进入轨迹第一帧，随后播放原始轨迹。A 键
   不会重新抓取机器人实测姿态，也不会重复执行 START 的起身流程。
3. **SELECT：低层锁存急停。**C++ bridge 在约 50 Hz 的状态链路上直接检测，
   立即拒绝后续策略指令，并持续发布 `Kp=0、Kd=8` 的阻尼命令。
4. SELECT 触发后没有手柄复位组合键。松开 SELECT、停止 Python 控制器，并重启
   bridge 才能清除锁存；不得自动恢复位置控制。

键盘 `s`、`a` 可分别作为 START、A 的调试备用；正常结束可在控制器终端按
`x`，再到 bridge 终端按 `q`。SELECT/阻尼会撤销位置保持，机器人可能变软并
下落，所以吊架不能省略；它也不能替代机器人独立的物理急停。

两个插值时间可在以下配置中调整：

- 起身时间：`sim2real/config/g1/controller.yaml` 的 `startup_interpolation_s`。
- 起身初始 Kp 比例：同一文件的 `startup_kp_floor`（当前为 0.10）。
- 轨迹首帧时间：`sim2real/config/g1/tracking.yaml` 的
  `motion_source.npz.first_frame_transition_s`。

## 4. 哪些动作推荐真机

批量 MuJoCo 结果：**PASS 83，MARGINAL 9，FALL 8，ERROR 0**。

### Tier 1：优先真机验证

这些动作以站立和上肢动作为主，但仍必须使用吊架：

- `08` — a person is standing and stretches both arms up over his head, he then stretches one arm to the opposite side and then the other arm to the opposite side for a full upper body stretch.
- `22` — a man holds his hands in front of him at waist-level
- `55` — a man stirs something anti-clock wise with his left hand.
- `56` — a man makes a "t" shape in the air with his right hand.
- `61` — a person reaches around in front of them with their left hand.
- `90` — the person is doing upper body twist.
- `96` — a person stands still then slightly moves their arms

### Tier 2：Tier 1 全部稳定后再验证

这些动作含迈步或行走，风险高于 tier1：

- `02` — a person takes a step forward.
- `10` — a person takes some slow small steps forward.
- `16` — a man slowly walks forward a few steps, stopping in a standing position.
- `23` — a person side stepping to their right, then side stepping to their left, and back again.
- `27` — person walks forward in slow manner
- `37` — a person walks slowly backwards
- `40` — a person taking strafing steps to their right
- `48` — the person walks forward slowly and then stops

## 5. 哪些动作目前不推荐真机

- **仿真摔倒，禁止首轮真机：** `03 25 38 42 49 78 86 87`
- **仿真边缘稳定，不推荐真机：** `01 05 06 07 12 45 58 63 79`
- **仿真 PASS，但未进入保守真机白名单：** `00 04 09 11 13 14 15 17 18 19 20 21 24 26 28 29 30 31 32 33 34 35 36 39 41 43 44 46 47 50 51 52 53 54 57 59 60 62 64 65 66 67 68 69 70 71 72 73 74 75 76 77 80 81 82 83 84 85 88 89 91 92 93 94 95 97 98 99`

最后一类并不等于动作一定危险，只表示目前证据不足，不应跳过单独复核和渐进式
安全验证。若以后要放行某条动作，应重新检查完整仿真、足端接触、关节速度/力矩
和真机限位，再明确加入 `hardware_candidates.txt`；不要绕过 `--hardware` 检查。

## 6. 评级含义

- `PASS`：批量仿真中根高度最低值不小于 0.60 m，且最大倾角不超过 35°。
- `FALL`：根高度低于 0.45 m，或最大倾角超过 60°。
- `MARGINAL`：介于上述条件之间。
- `tier1/tier2`：从 PASS 中进一步人工保守筛选出的初测顺序，不是安全认证。

## 7. 100 条动作详细结果

| ID | 仿真评级 | 真机级别 | 最低根高度 (m) | 最大倾角 (deg) | 关节 RMSE (rad) | 源轨迹最大速度 (rad/s) | 动作描述 |
|---:|:---:|:---:|---:|---:|---:|---:|:---|
| 00 | PASS | - | 0.710 | 29.0 | 0.175 | 26.05 | a person runs backwards, turns halfway then runs forward. |
| 01 | MARGINAL | - | 0.593 | 16.8 | 0.098 | 5.53 | the person stands still in a slight squat and then turns to their right and walks. |
| 02 | PASS | tier2 | 0.755 | 12.5 | 0.079 | 4.52 | a person takes a step forward. |
| 03 | FALL | - | 0.414 | 40.8 | 0.154 | 12.38 | person is leaning down then getting up and walking. |
| 04 | PASS | - | 0.654 | 32.8 | 0.153 | 24.36 | a man uses his hands to bounce a basketball twice and takes two big steps forward, then jumps up high with the basketball in his hands and brings his hands up and lets go of the basketball to score a point. |
| 05 | MARGINAL | - | 0.593 | 18.5 | 0.091 | 11.68 | a person walking over to sit down. |
| 06 | MARGINAL | - | 0.597 | 11.3 | 0.150 | 26.25 | a person in boxing class |
| 07 | MARGINAL | - | 0.717 | 46.8 | 0.168 | 31.58 | someone tryiing to fight while moving his body |
| 08 | PASS | tier1 | 0.755 | 5.1 | 0.113 | 5.79 | a person is standing and stretches both arms up over his head, he then stretches one arm to the opposite side and then the other arm to the opposite side for a full upper body stretch. |
| 09 | PASS | - | 0.741 | 12.0 | 0.116 | 8.01 | walks in a wide clockwise bend then cuts a sharp left before completing a circle. |
| 10 | PASS | tier2 | 0.755 | 10.6 | 0.091 | 4.39 | a person takes some slow small steps forward. |
| 11 | PASS | - | 0.756 | 18.7 | 0.109 | 25.41 | a person walks forward and gets pushed, he stumbles to the right and then returns to his original path |
| 12 | MARGINAL | - | 0.695 | 56.8 | 0.116 | 6.62 | a person walks forward, picks something up off the ground, and places it on a higher object |
| 13 | PASS | - | 0.738 | 15.3 | 0.144 | 9.29 | the person is walking around looking around for something. |
| 14 | PASS | - | 0.740 | 17.6 | 0.101 | 8.79 | a person walks quickly in a very feminine way. |
| 15 | PASS | - | 0.755 | 12.6 | 0.103 | 7.91 | person does light bouncy jump, skips forward with right leg swinging forward |
| 16 | PASS | tier2 | 0.751 | 9.2 | 0.105 | 6.37 | a man slowly walks forward a few steps, stopping in a standing position. |
| 17 | PASS | - | 0.755 | 10.4 | 0.097 | 7.24 | a man steps back, picks something up and put it to his head and then puts it back. |
| 18 | PASS | - | 0.721 | 19.8 | 0.146 | 12.22 | a person jogs in a circle to the left. |
| 19 | PASS | - | 0.747 | 17.3 | 0.127 | 7.74 | a person walks in a counterclockwise semi-circle |
| 20 | PASS | - | 0.753 | 11.6 | 0.117 | 6.70 | the man takes 9 proud steps forward and turns around. |
| 21 | PASS | - | 0.754 | 28.5 | 0.108 | 13.80 | a person doing an elephant impression |
| 22 | PASS | tier1 | 0.756 | 7.6 | 0.078 | 4.26 | a man holds his hands in front of him at waist-level |
| 23 | PASS | tier2 | 0.729 | 8.3 | 0.097 | 4.84 | a person side stepping to their right, then side stepping to their left, and back again. |
| 24 | PASS | - | 0.665 | 24.4 | 0.096 | 17.81 | a person jumps forward once. |
| 25 | FALL | - | 0.425 | 34.6 | 0.136 | 10.48 | a person walks forward, then squats down deeply with both arms outstretched in front of the body as if picking something up. |
| 26 | PASS | - | 0.706 | 16.4 | 0.082 | 14.89 | a person hops into the air one time and then stops. |
| 27 | PASS | tier2 | 0.751 | 11.0 | 0.102 | 5.17 | person walks forward in slow manner |
| 28 | PASS | - | 0.738 | 16.4 | 0.127 | 8.31 | person is walking towards the right and then the last and then back to the right. |
| 29 | PASS | - | 0.642 | 13.2 | 0.116 | 18.40 | this person has both arms extended to his sides then drops his left arm. |
| 30 | PASS | - | 0.755 | 9.6 | 0.087 | 9.23 | the person does an underhand throw |
| 31 | PASS | - | 0.755 | 16.1 | 0.092 | 4.44 | the man picks something upright then lays it back down. |
| 32 | PASS | - | 0.739 | 12.8 | 0.115 | 7.67 | a person is walking in a circle while swinging his arms around. |
| 33 | PASS | - | 0.755 | 12.2 | 0.125 | 16.83 | person turns around and raises right arm to touch something above his or her head |
| 34 | PASS | - | 0.757 | 10.2 | 0.102 | 11.00 | a person walks down a set of stairs. |
| 35 | PASS | - | 0.755 | 14.8 | 0.099 | 6.64 | a person is being pushed from behind hard and he moves forward. |
| 36 | PASS | - | 0.755 | 9.1 | 0.096 | 7.47 | a person steps forward with their right foot, and wipes across a surface with their right hand. |
| 37 | PASS | tier2 | 0.755 | 6.7 | 0.084 | 4.12 | a person walks slowly backwards |
| 38 | FALL | - | 0.351 | 44.7 | 0.137 | 4.69 | a person sits cross legged and then stands back up. |
| 39 | PASS | - | 0.754 | 7.9 | 0.092 | 14.50 | a person lifts their hand to wave and then brings it back to their side. |
| 40 | PASS | tier2 | 0.741 | 6.7 | 0.089 | 3.81 | a person taking strafing steps to their right |
| 41 | PASS | - | 0.726 | 15.3 | 0.106 | 6.44 | a person is walking and bends to the right |
| 42 | FALL | - | 0.069 | 110.7 | 0.331 | 22.06 | a man bends over an all fours and arches his back. |
| 43 | PASS | - | 0.755 | 12.2 | 0.118 | 6.24 | the person walked to the right and then walked back to the left. |
| 44 | PASS | - | 0.729 | 19.3 | 0.101 | 15.74 | a person jumps with their arms extended to the side. |
| 45 | MARGINAL | - | 0.589 | 15.0 | 0.127 | 7.17 | a person is standing with knees bent with his hands out to his side, then places his hands on his knees, then brings his hands back up. |
| 46 | PASS | - | 0.755 | 9.8 | 0.106 | 17.06 | a person stand still water fishing |
| 47 | PASS | - | 0.726 | 6.1 | 0.102 | 14.29 | a person jogs down stairs then jumps up with both feet. |
| 48 | PASS | tier2 | 0.749 | 13.2 | 0.096 | 6.01 | the person walks forward slowly and then stops |
| 49 | FALL | - | 0.690 | 69.8 | 0.126 | 9.07 | a man is pretending to be a chicken. constantly pecking at the ground and waving his arms like a chicken. |
| 50 | PASS | - | 0.720 | 24.3 | 0.127 | 14.93 | the person walks in a straight line at a angle to their left, then turns around and jogs back to the start. |
| 51 | PASS | - | 0.755 | 7.3 | 0.089 | 8.65 | a person walks forward in a gingerly manner |
| 52 | PASS | - | 0.754 | 13.0 | 0.109 | 7.30 | a person walks backwards in a counterclockwise curve. |
| 53 | PASS | - | 0.755 | 9.6 | 0.082 | 3.53 | a person flattens their dress and starts to curtsey. |
| 54 | PASS | - | 0.742 | 23.3 | 0.212 | 12.09 | the person is balancing on one leg using his hands to help balance |
| 55 | PASS | tier1 | 0.755 | 6.4 | 0.081 | 7.59 | a man stirs something anti-clock wise with his left hand. |
| 56 | PASS | tier1 | 0.755 | 9.4 | 0.087 | 8.88 | a man makes a "t" shape in the air with his right hand. |
| 57 | PASS | - | 0.747 | 12.2 | 0.111 | 6.58 | a person walks in a circle clockwise. |
| 58 | MARGINAL | - | 0.740 | 40.7 | 0.088 | 5.78 | a man stands up from bowing and then steps back. |
| 59 | PASS | - | 0.749 | 22.7 | 0.115 | 6.07 | a person walks clock-wise in an oval shape with both hands stretched put in front. |
| 60 | PASS | - | 0.755 | 30.4 | 0.112 | 6.60 | the man pick something up and hung it on the wall. |
| 61 | PASS | tier1 | 0.755 | 10.7 | 0.094 | 8.69 | a person reaches around in front of them with their left hand. |
| 62 | PASS | - | 0.755 | 12.0 | 0.110 | 9.91 | a person walks around and stops. |
| 63 | MARGINAL | - | 0.589 | 16.5 | 0.096 | 5.69 | a person walks forward, takes a set to its right. |
| 64 | PASS | - | 0.731 | 13.3 | 0.094 | 7.27 | the man takes 5 curved steps. |
| 65 | PASS | - | 0.754 | 8.2 | 0.100 | 12.70 | a person jumped while raising the hands and repiting it |
| 66 | PASS | - | 0.749 | 11.7 | 0.099 | 9.82 | the person does several jumping jacks. |
| 67 | PASS | - | 0.740 | 13.2 | 0.105 | 14.64 | the person was walking forward on a balance beam. |
| 68 | PASS | - | 0.752 | 10.3 | 0.108 | 5.00 | a man walks from side to side and turn round and walks to his place. |
| 69 | PASS | - | 0.751 | 11.8 | 0.102 | 7.59 | a man walk and make a left turn. |
| 70 | PASS | - | 0.755 | 23.4 | 0.113 | 5.89 | a man kicks with his left leg and then kicks with his right leg. |
| 71 | PASS | - | 0.755 | 21.2 | 0.105 | 5.23 | a person leans to his left. |
| 72 | PASS | - | 0.755 | 17.9 | 0.104 | 6.14 | a person puts their hands on their hips while looking left then right, then the person steps |
| 73 | PASS | - | 0.754 | 9.0 | 0.086 | 20.61 | a person flips his left arm in frustration. |
| 74 | PASS | - | 0.610 | 21.3 | 0.111 | 5.73 | the man is making gestures |
| 75 | PASS | - | 0.755 | 27.2 | 0.089 | 2.40 | a person hunches with hands forward then shakes his head right to left. |
| 76 | PASS | - | 0.755 | 14.3 | 0.085 | 21.00 | moving body side to side. |
| 77 | PASS | - | 0.711 | 30.4 | 0.124 | 21.08 | a person walks to the left with a limp, then turns around and sprints back to where he started |
| 78 | FALL | - | 0.390 | 75.5 | 0.200 | 7.22 | a person kneels down firstly on his left, then his right. |
| 79 | MARGINAL | - | 0.707 | 51.4 | 0.177 | 28.92 | a person kicks with their left leg twice, and then once with their right. |
| 80 | PASS | - | 0.755 | 5.9 | 0.099 | 14.85 | a person lifts an object to his mouth with his right hand as he tilts his head back. |
| 81 | PASS | - | 0.749 | 13.6 | 0.096 | 7.06 | a person walks forward and looks up. |
| 82 | PASS | - | 0.739 | 9.7 | 0.091 | 7.96 | person steps forwards while turning to his right, then spinning around so he is facing where he originally started, then he steps forwards once, then steps backwards once, and shuffles to the left. |
| 83 | PASS | - | 0.752 | 11.0 | 0.106 | 7.47 | person walks forward in casual way |
| 84 | PASS | - | 0.755 | 10.8 | 0.103 | 9.86 | a person is walking around aimlessly as if pondering something. |
| 85 | PASS | - | 0.755 | 8.1 | 0.100 | 15.14 | the person is trying to talk with his hands. |
| 86 | FALL | - | 0.082 | 134.3 | 0.517 | 30.85 | a man sits, pulls his phone out, answers the phone and puts it to his head. |
| 87 | FALL | - | 0.092 | 113.4 | 0.308 | 16.62 | person looks like they are making arm and leg circles as if they are doing breast stroke. |
| 88 | PASS | - | 0.755 | 11.2 | 0.068 | 11.87 | someone is holding something to their ear |
| 89 | PASS | - | 0.754 | 28.3 | 0.099 | 6.22 | a man is holding something in both hands and his right arm starts moving. |
| 90 | PASS | tier1 | 0.755 | 12.0 | 0.128 | 4.40 | the person is doing upper body twist. |
| 91 | PASS | - | 0.755 | 12.1 | 0.091 | 10.89 | person stands straight with both arms stretched outward, claps their hands five times and returns both hands down to his side |
| 92 | PASS | - | 0.754 | 12.3 | 0.261 | 24.42 | a person moves their hands up above their head, appearing to stretch their back. |
| 93 | PASS | - | 0.755 | 9.5 | 0.098 | 17.11 | a person stand still water fishing |
| 94 | PASS | - | 0.750 | 13.6 | 0.100 | 10.75 | a person walks backwards and then steps up stairs backwards as well |
| 95 | PASS | - | 0.760 | 7.7 | 0.083 | 10.62 | a man waves his left hand. |
| 96 | PASS | tier1 | 0.755 | 9.4 | 0.098 | 1.18 | a person stands still then slightly moves their arms |
| 97 | PASS | - | 0.754 | 18.4 | 0.090 | 1.60 | a person looks left, then looks right, then confirms left by looking left again. |
| 98 | PASS | - | 0.755 | 27.3 | 0.070 | 5.28 | a person washes himself with his left arm |
| 99 | PASS | - | 0.755 | 11.2 | 0.093 | 8.64 | a person walks forward with a little run in one of the steps. |

## 8. 相关文件

- 真机白名单：`sim2real/config/g1/motions/examples100/hardware_candidates.txt`
- 机器可读结果：`motion_examples_100_validation.json`
- 单动作运行器：`sim2real/run_deploy_example100.sh`
