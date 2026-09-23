# PartiGen reconstruction on TextOpRobotMDAR

本目录基于现存的 TextOpRobotMDAR 代码和 `main.pdf`（PartiGen）补建生成模型。
这里没有 `.git` 历史，本文档不代表找回了原始仓库或实验权重。

## 训练参数由研究者另行提供

按当前要求，新增训练配置**不发布训练超参数默认值**。学习率、优化器参数、步数、
训练与验证 batch size、dropout、文本丢弃率、扩散步数、各损失权重、阶段激活位置、
随机种子、保存与评估频率均为 Hydra 必填项 `???`。代码不会用隐式默认值补齐它们。
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

## 原论文没有充分指定的细节

以下是可运行重建所采用的约定，**不能声称与丢失代码逐行一致**：

1. 分布/潜变量 token 的原始 57 维分支填零，保留学习到的隐藏 token。
   解码器读取真实历史和全零未来占位；未来真实特征不会泄漏进解码器。
   掩码层始终读取这些原始输入，标准注意力层融合共享隐藏状态。
2. EOS 输入是解码帧、该原语共享潜变量、文本向量，以及相对于候选长度归一化的
   帧位置。论文未说明输入拼接方式；前两个 EOS 隐藏层宽度也必须显式提供。
3. 注释区间视为 `[start, end)`，秒转换到 30 Hz 帧索引；最后真实帧为正标签。
   随机抽取动作及原语，保留包含末尾的不足整段样本；右侧重复末帧，填充帧不参与
   重构、几何或 EOS 损失。超出候选上限而被裁掉的动作不会制造假 EOS。
   EOS 标签边界与物理序列边界分开：注释末尾若存在真实下一帧，保留该帧计算
   Table S3 的位移/角度增量，避免人为制造静止信号；只有物理序列末尾才重复状态。
   历史早于序列起点时重复起始帧。
4. 论文阶段说明是定性的，没有给出逐项 ramp 函数。这里提供显式的按阶段开关，
   不替研究者推断损失开启时刻。
5. 旧 BodyPart 实现是在隐藏特征上处理注意力，无法等同于论文的原始特征掩码。
   新模型采用独立类和严格权重检查；不做 `strict=False` 式的部分加载。
   现有服务器 checkpoint 是否兼容，须以其实际结构为准。

这次没有重建 G1 评估器、GMR 重定向、GRIT 追踪器或硬件部署系统，也没有复现论文
报告的指标。它们属于独立资产与评估流程。

## 生成

用训练时相同的训练集统计量归一化历史，准备一个 PyTorch 文件，包含
`history: [B, 2, 57]` 和 `text: [B, 512]`（CLIP ViT-B/32 embedding）。

```bash
python -m robotmdar.cli --config-name generate_partigen checkpoint=/path/to/ckpt.pth inputs=/path/to/inputs.pt output=/path/to/generated.pt
```

输出包含完整 `candidate`、`eos_probabilities`、`lengths` 和按帧截断的 `motions`。
这些是**归一化的 57 维动作特征**，不是可直接发送给机器人的控制指令；现有
`dataset.reconstruct_motion` 可结合原始初始位姿重建轨迹，后续跟踪接口仍由服务器提供。

## 验证与依赖

CPU 行为检查：

```bash
python -m unittest discover -s tests -v
```

核心网络和 EOS 测试需要 PyTorch、NumPy；配置检查另外需要 hydra-core。
服务器端数据/FK 训练沿用原项目依赖，包括 `isaac_utils`、OpenAI CLIP、
`easydict`、`joblib`、`loguru`、`scipy`、`tensorboard` 和 G1 XML 及关联资源。
本机没有服务器数据、G1 XML 或 `isaac_utils`；本地测试用可微合成 FK 验证损失和
梯度传递，并对实际使用的四元数转换、速度差分进行数学回归检查。
2026-09-23 修复后完整 CPU 回归测试共 37 项，全部通过；修复追踪见 `PAPER_CONSISTENCY_AUDIT.md`。
这些检查不能替代真实机器人资产的端到端训练。
