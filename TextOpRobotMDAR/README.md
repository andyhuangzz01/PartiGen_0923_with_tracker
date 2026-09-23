# PartiGen reconstruction on TextOpRobotMDAR

本目录基于现存的 TextOpRobotMDAR 和partigen打包的仓库，出于轻量化考虑，已经略去了HML3D-G1数据集、各项机器人配置和MVAE+MLD+Evaluator的checkpoint。

## 训练参数由研究者另行提供

按当前要求，新增训练配置**训练超参数默认值都为必填实验项**。学习率、优化器参数、步数、
训练与验证 batch size、dropout、文本丢弃率、扩散步数、各损失权重、阶段激活位置、
随机种子、保存与评估频率均为 Hydra **必填项**。代码不会用隐式默认值补齐它们。
网络宽度、层数、关节分组、特征布局等结构信息来自论文。推理配置单独给出论文的
候选长度、CFG 系数和 EOS 阈值。

旧 `train_*humanml3d*.yaml` 等历史文件保留原有参数及路径，只纠正特征注释并标明遗留入口；
它们不是新增 PartiGen 训练入口的默认配置。服务器已有数据路径、机器人模型路径和 checkpoint 文件均未改动。

## 实现入口

- `robotmdar/partigen/networks.py`：BP-MVAE、DA-LDM、四层帧级 EOS MLP 和 CFG。
- `robotmdar/partigen/data.py`：直接读取现有 PKL、CLIP 缓存与训练集 `meanstd.pkl`。
- `robotmdar/partigen/eos.py`：末帧标签、有效帧加权 BCE、包含越阈值帧的截断。
- `robotmdar/partigen/losses.py`：Huber、FK、时序差分、二阶关节平滑和分层损失。
- `robotmdar/partigen/training.py`：AdamW 两阶段训练、退火、梯度裁剪、恢复和验证。
- `robotmdar/partigen/generation.py`：完整候选生成后执行 EOS，输出归一化动作特征。

检查配置模板，不启动训练：

```bash
python -m robotmdar.cli --config-name train_partigen_mvae --cfg job
python -m robotmdar.cli --config-name train_partigen_dar --cfg job
```

建议在服务器自己的配置目录创建实验文件，继承相应模板，然后显式填写全部 `???`：

```yaml
# my_mvae.yaml; add the remaining required fields from the template.
defaults:
  - train_partigen_mvae
  - _self_
```

```bash
python -m robotmdar.cli --config-dir /path/to/private-configs --config-name my_mvae
python -m robotmdar.cli --config-dir /path/to/private-configs --config-name my_dar
```

第二阶段需要 `ckpt.vae`，或包含 VAE 和 denoiser 的新格式 `ckpt.dar`。
VAE 与 LDM 必须采用相同的原语长度。新 checkpoint 保存网络、优化器、步数、
配置、随机状态；DAR checkpoint 同时包含冻结 VAE，生成时不依赖另一个权重文件。
配置中的 checkpoint 路径由运行者指向服务器文件，不会搜索并擅自替换已有权重。

## 论文结构对照

| 项目 | 实现 |
| --- | --- |
| 帧表示 | 57 维，沿用 FeatureVersion 3；关节角 11–33，角增量 34–56 |
| 共享输入 | 每个头均能看到 0–10 的根节点/接触特征和全部角增量 |
| 解剖掩码 | 四个头对应左腿、右腿、躯干、双臂；掩码先于独立 Q/K/V 投影 |
| BP-MVAE | 编解码各九层；每两层掩码注意力后一层标准注意力，重复三次 |
| 潜变量 | 每原语一个共享 128 维向量，不分配独立部位潜变量 |
| DA-LDM | 八层 Transformer；预测干净潜变量，历史通过 cross-attention 保留 |
| 条件 | CLIP ViT-B/32；训练只丢弃文本条件，不丢弃历史 |
| 扩散 | cosine beta、START_X、FIXED_SMALL、uniform timestep、无 timestep rescale |
| 重构和几何 | mean Huber；VAE 几何项统一乘外层系数；LDM 不使用 KL |
| EOS | 四层 MLP，前三层 GELU，最后隐藏层宽度 128；valid-frame weighted BCE |
| 推理 | 全部 DDPM 反向更新、无 clean-latent clipping；先完整生成再判断 EOS |
| 帧数 | 8 帧原语生成 40 段，16 帧原语生成 20 段；候选均为 320 帧 |
| 截断 | 第一个概率严格大于 0.9 的帧，包含该帧；未越阈值则保留全部候选 |

训练阶段边界、每个损失的启用阶段及所有系数由实验配置显式提供。
FK 复用现有服务器 G1 模型，沿用组件四元数 Huber（未改为测地线损失）。
接触损失的非接触零元素仍计入有效帧的均值分母。

根四元数按 xyzw 正确转换为轴角后送入 FK，`orient_delta` 只监督根节点姿态变化。
关节速度使用前向差分；没有下一帧时以最后一个有效差分延拓，单帧序列速度为零。
这是论文未指定的末帧边界约定，不改变中间帧的速度定义。
FK 的全局速度平滑仅沿时间轴执行，各关节和坐标通道独立；单帧全局角速度返回三维零向量。
训练时去噪头与 EOS 头共用同一次采样得到的文本丢弃结果，历史始终保留。


## 生成

用训练时相同的训练集统计量归一化历史，准备一个 PyTorch 文件，包含
`history: [B, 2, 57]` 和 `text: [B, 512]`（CLIP ViT-B/32 embedding）。

```bash
python -m robotmdar.cli --config-name generate_partigen checkpoint=/path/to/ckpt.pth inputs=/path/to/inputs.pt output=/path/to/generated.pt
```

输出包含完整 `candidate`、`eos_probabilities`、`lengths` 和按帧截断的 `motions`。
这些是**归一化的 57 维动作特征**，不是可直接发送给机器人的控制指令；现有
`dataset.reconstruct_motion` 可结合原始初始位姿重建轨迹，后续跟踪接口仍由服务器提供。
