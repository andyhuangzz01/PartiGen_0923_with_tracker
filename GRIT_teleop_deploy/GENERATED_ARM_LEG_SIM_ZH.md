# dar0911 手脚配合动作：一键仿真

数据源：`TextOpRobotMDAR/gen_test2576_dar0911_512/sample_XXXX/qpos.npy`。
这里使用生成组 sample ID，不是 src_idx；不要加 `--baseline`。

```bash
cd /home/andy/python_code/dartG1-main/GRIT_teleop_deploy/sim2real
bash run_generated.sh 1183 --sim
```

| index | 生成 prompt 的含义 |
| --- | --- |
| 1183 | 左右迈步，同时伸臂挥手 |
| 222 | 来回走路，同时双臂画大圈 |
| 1804 | 举起双手、以庆祝姿态向前走 |
| 307 | 小范围绕圈滑步跳舞 |
| 1587 | 右手投掷，同时小幅向前移动 |
| 950 | 夸张摆臂走路，转身走回来 |

替换命令中的 index 即可。也可运行 `bash run_generated.sh --sim`，按提示输入编号。
自动启动 MuJoCo 和控制器，默认站姿过渡后，用 5 秒插值到动作第一帧，
播放一次并回默认姿态，随后退出。Ctrl+C 中断。不要启动真机 bridge；一次只运行一个。

222 和 307 复用原生成组评估的转换文件；其余 4 个已转换到
`GRIT_teleop_deploy/generated_sim_previews/sample_XXXX_*/motion.npz`。
新转换保留全部源帧，以 xyzw 四元数读取，30 Hz 重采样到 50 Hz，缺失的 6 个手腕关节填零。
源数据、转换脚本或相关配置变化后，启动器会新建转换目录，不覆盖旧文件。

仅检查路径和命令：`bash run_generated.sh 1183 --sim --dry-run`。
仅准备文件：`bash run_generated.sh 1183 --sim --prepare-only`。
无窗口运行：`bash run_generated.sh 1183 --sim --headless`。

这些是仿真入口，不构成真机许可。新增预览标记为 NOT_EVALUATED / SIM_ONLY，
不是正式筛选结果；222 的原生成组评估为 FALL，不应上真机。
