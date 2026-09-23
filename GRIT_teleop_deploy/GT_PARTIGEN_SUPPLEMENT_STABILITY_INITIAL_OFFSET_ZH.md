# GT / PartiGen 补充诊断：稳定子集与首帧平移校正

核对日期：2026-09-20。读取已有逐动作结果并重新汇总稳定子集；未重跑仿真，未更改任何原始表格、评估结果或首帧校正值。

## 1. 稳定子集 MPJPE

输入：

- PartiGen：`gen_test2576_dar0911_mpjpe_0001_0500/summary.json`，496 条有效记录。
- GT：`gt_random500_seed20260918/summary.json`，499 条有效记录。

筛选沿用 `screen_generated_motions.py::write_report`：先排除 ERROR，然后要求 `root_z_min >= 0.60 m` 且 `root_tilt_max_deg <= 35.0°`。核对筛选结果恰好等于已有 `rating == PASS` 的样本：PartiGen 409 条，GT 427 条。这是原有仿真稳定性代理判据，不是更严格的真机候选筛选，也不保证位置跟踪误差小。

统计使用未经首帧偏移校正的原始 MPJPE；不叠加剔除最高 10% 的筛选。每条动作先对动作播放阶段的全部帧和 19 个身体点求均值，再对动作等权求均值或中位数。所有数值单位为 mm。

| 方法 | 稳定通过数 | Global 均值 | Global 中位数 | Root-relative 均值 | Root-relative 中位数 |
|---|---:|---:|---:|---:|---:|
| PartiGen | 409 | 198.00 | 97.08 | 42.84 | 37.98 |
| GT | 427 | 297.81 | 161.34 | 44.26 | 36.04 |

用于解释的补充均值（不替换原表）：

| 方法 / 范围 | 动作数 | Global 均值 | Root-relative 均值 |
|---|---:|---:|---:|
| PartiGen 全部有效 | 496 | 252.40 | 57.99 |
| PartiGen 稳定通过 | 409 | 198.00 | 42.84 |
| PartiGen 未通过稳定性判据 | 87 | 508.15 | 129.24 |
| GT 全部有效 | 499 | 399.78 | 56.24 |
| GT 稳定通过 | 427 | 297.81 | 44.26 |
| GT 未通过稳定性判据 | 72 | 1004.55 | 127.30 |

解释：未通过稳定性判据的动作明显拉高两组总体误差。但稳定子集中 GT 的 Global 均值仍高于 PartiGen，而 Root-relative 均值接近。稳定性代理判据主要约束高度和倾角，并不排除整体位置漂移。GT 与生成组非配对且自由度配置不同，不能把以上差异直接解释为生成模型优于 GT。

## 2. 已有首帧偏移校正的准确公式

代码：`sim2real/tools/analyze_initial_pelvis.py`，核心实现：

```python
d = actual_root_w - reference_root_w
e = actual_pos_w - reference_pos_w
corrected = np.linalg.norm(e - d[0][None, None, :], axis=-1).mean() * 1000
```

对于第 i 条动作，p 为实际世界坐标，p_hat 为执行参考世界坐标（单位 m），T_i 为动作播放帧数，K=19 且包含 pelvis：

\[
d_i=p^{(i)}_{0,\mathrm{pelvis}}-\hat p^{(i)}_{0,\mathrm{pelvis}}\in\mathbb R^3,
\qquad
E^{(i)}_{\mathrm{corrected}}=
\frac{1000}{19T_i}\sum_{t=0}^{T_i-1}\sum_{j=1}^{19}
\left\|(p^{(i)}_{t,j}-\hat p^{(i)}_{t,j})-d_i\right\|_2.
\]

组均值为每条有效动作校正后误差的等权平均：

\[
\overline E_{\mathrm{corrected}}=\frac1N\sum_{i=1}^{N}E^{(i)}_{\mathrm{corrected}}.
\]

- 校正的是 **XYZ 三维偏移**，没有将 Z 分量置零。
- 每条动作有各自的一个固定 d_i；该动作的所有时刻、所有 19 个身体点减去同一个 d_i。
- t=0 是入场插值后的动作播放第 0 帧；首帧仍参与计算，**不是删除首帧**。
- 不逐帧重新计算平移校正；不做旋转、尺度或时间配准。因此后续新增漂移仍计入。
- 不是 Root-relative MPJPE：后者每帧分别减去实际/参考 pelvis 位置。
- 校正值为诊断量，不能代替原始 Global MPJPE；数值不保证一定下降。

| 方法 | 该校正结果的样本范围 | 固定首帧 XYZ 偏移校正后 Global 均值（mm） |
|---|---|---:|
| GT | 全部 499 条有效动作，非仅稳定子集，未截尾 | 308.89 |
| PartiGen | 全部 496 条有效动作，非仅稳定子集，未截尾 | 256.79 |

已有校正结果位于 `gt_generated_initial_pelvis_comparison/comparison.json`；本次仅核对实现与数值，没有重算或改变这两个值。
