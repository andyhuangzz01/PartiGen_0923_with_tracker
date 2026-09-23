# PartiGen 代码与论文一致性审查

## 修复追踪（2026-09-23）

按后续“直接修改、不再寻找缺失配置”的要求，已实施：

- A1：xyzw 四元数正确转换为轴角，保留可微性。
- A2：`orient_delta` 仅计算根节点，不再平均所有 body。
- A3：关节角速度末帧延拓最后有效差分，单帧速度为零；这一边界约定已单独注明。
- FK 附加修复：速度平滑严格沿时间轴，不再混合关节或坐标通道；单帧全局角速度返回三维向量，并校验速度计算的时间间隔。
- B2：分开注释 EOS 边界和物理序列边界，有真实下一帧时保留 look-ahead 增量。
- B3 的文本条件问题：去噪头和 EOS 头使用同一次文本丢弃结果。
- D 的特征布局错误注释已纠正，相关旧配置标明遗留实验及新入口。
- 新训练入口改用专用扩散配置，防止 Hydra 合并时保留旧配置的 `num_timesteps: 5`；扩散步数必须显式提供。

训练超参数继续必填；未寻找原始仓库、服务器配置、评估器权重或机器人资产。
修复配置继承后，MVAE / DAR 模板实际必填字段分别为 43 / 36 项。
B1 的解码器 raw 来源、B3 的 EOS 输入组合等论文未唯一规定的部分仍保留明确说明，
没有凭空新增隐藏层或把未证实的结构宣称为论文原实现。
修复后完整 CPU 回归测试 **37 项全部通过**，涵盖旋转数学与梯度、速度边界与平滑、
root-only 损失、EOS look-ahead/padding、共享文本掩码、配置必填和模型训练/生成流程。
这些检查使用合成输入，不包含真实 G1 XML、服务器数据或 checkpoint 验证，也不代表论文指标复现。

## 修复前的历史审查记录

下文保留**修复前的审查记录与诊断数值**，不代表上述已修问题仍存在。

审查日期：2026-09-23。依据：`main.pdf` 正文第 3–4 节、附录 S2–S3。
范围：新增 `robotmdar/partigen/` 及其实际调用的旧特征转换/FK/扩散代码，同时检查保留的旧训练入口。
本次不修改模型、训练代码、服务器路径或训练超参数。下列数值是审查诊断输入，不是训练默认值。

## 结论

**当前实现尚不能称为与论文严格一致。**

- 已确认一个影响 FK 正确性的数学错误，以及一个与 Table S4 明确定义不符的损失。
- 关节速度还有一个终端索引问题；论文没有给出终端速度约定，须区分代码边界错误与论文未说明部分。
- 解码器未来 raw 输入、EOS 输入/标签/文本掩码仍有高影响未决约定。
- 论文中的测试集评估及执行系统没有在本目录恢复。
- 新训练超参数按用户要求留空是正确行为，不计入缺陷；实际实验参数尚不可核验。

13 项现有 CPU 测试仍全部通过。这些测试证明局部行为、形状、梯度和序列截断可运行，不能证明真实 G1 FK 正确或复现论文指标。

## A. 已确认、应优先处理的问题

### A1 · P1 · 根姿态欧拉角被作为轴角送入 FK

位置：`robotmdar/skeleton/robot.py:58`、`:73–82`；`robotmdar/skeleton/forward_kinematics.py:218–220`。
影响调用点：`robotmdar/partigen/losses.py:44–46`。

`quaternion_to_euler_angles(root_rot)` 返回 roll/pitch/yaw，但结果命名为 `root_rot_aa`，随后与关节轴角一起传入 `fk_batch`。
后者调用 `axis_angle_to_quaternion`。欧拉角和轴角不是同一种表示；多轴同时旋转时，这条路径改变了根姿态。

数值诊断：

- 正确根姿态 RPY = `[0.4, 0.3, 1.0]` rad。
- 将对应四元数转回 Euler，再误当作 axis-angle 转回旋转，旋转误差为 **0.2507724 rad（约 14.37°）**。
- 同一个相对根节点偏移 `[0, 0, -0.8]` m，经两种旋转得到的位置相差 **0.1698595 m**。这是合成偏移诊断，不是实测 G1 关节误差。
- 单轴 yaw 情况通常不会暴露问题，因此仅用直立/平面旋转检查会漏检。

论文关联：Eq. (2)、S2.5 第 19 页要求对重建轨迹进行 G1 FK 几何监督。当前计算的身体姿态/位置不是该输入根姿态对应的物理 FK。
这一问题来自旧代码，新训练入口继承了它。预测和真值都经过同一错误映射，也不能使该映射等价于正确 FK。

处理方向：使用正确的 quaternion→axis-angle 转换，或让 FK 根节点直接接收正确旋转矩阵，并严格保持四元数 xyzw/wxyz 约定。
增加混合 roll/pitch/yaw 的矩阵一致性、关节位置和梯度检查。

### A2 · P2 · `orient_delta` 惩罚了全身，而论文规定根姿态变化

位置：`robotmdar/partigen/losses.py:71–75`。

Table S4 第 17 页将 `delta orient` 定义为 **“Temporal change in root orientation”**。
当前把整块 `global_rotation`（`[B,T,J,4]`）做时间差分，没有选择 root body。
这会额外惩罚所有非根关节的姿态变化，并改变损失的均值分母及相对权重。

调用实际 `geometry_terms` 的最小诊断：

- 两个 body：root 和一个 child；root 在全部帧保持单位四元数。
- 真值 child 始终不动；预测 child 在未来第一帧旋转 1 rad 后保持。
- 应有的 root-only `orient_delta = 0`。
- 当前函数返回 **0.0076510906**。

处理方向：至少将该项限制到 `global_rotation[:, :, 0]`。
论文没有进一步唯一指定姿态差分必须采用相对四元数还是组件差分，不应借此擅自修改整个旋转损失定义。
该项在 VAE 启用时影响目标；若按论文关闭 DAR 的额外 delta 项，DAR 不受此项直接影响。

### A3 · P2 · 末帧角速度复制了倒数第二个间隔

位置：`robotmdar/skeleton/forward_kinematics.py:266–268`。

计算前向差分后，代码用 `dof_vel[:, -2:-1]` 补齐最后一帧。
例如 `q=[0,1,3,6]`、`dt=1/30`，三段真实差分为 `[30,60,90]`，实际输出 **`[30,60,90,60]`**。
末尾复制的是 60，而不是最后相邻间隔的 90。输入只有两帧时只返回一帧速度，只有一帧时返回空速度。

新训练一般有两帧历史和完整 padded primitive，不会直接遇到 T=1/2 的形状问题；但完整 primitive 最后一帧的速度仍采用上述旧间隔。

论文关联：S2.5 第 19 页定义角速度为 `(q[t+1]-q[t])/dt`，没有说明缺少下一帧时的边界策略。
因此不能宣称论文明确指定末帧必须为 90 或必须为 0；应先统一边界约定，再修正错误间隔/短序列长度问题并加入相应 mask。

## B. 高影响未决设计，尚不能证明与原实现一致

### B1 · 六个解码器 masked-attention 分支只看历史，不看生成内容

位置：`robotmdar/partigen/networks.py:66–68`、`:83–85`、`:137–139`。

解码器 raw 分支固定为 `[零 latent 占位，两帧真实历史，全部零未来占位]`，各层重复读取同一个 raw 张量。
结果是六个 masked-attention 分支的未来 query 完全相同，其输出不依赖 latent、当前 hidden 或未来帧位置。

实测默认 9 层、512 hidden 模型（仅诊断时关闭 dropout）：

- decoder masked 层 `[0,1,3,4,6,7]` 的各未来位置输出最大差异全部为 **0**。
- 改变 latent 后，这六个 attention 分支的输出变化全部为 **0**。
- 最终生成动作仍发生变化，最大差异约 **0.1437317**；latent 经三个标准注意力层传播。

不能错误表述为“整个 decoder 不使用 latent”。但目前六个 masked 分支只提供历史相关的更新，无法对未来生成内容进行部位相关的注意力处理。
S2.1/S2.2 规定 decoder 使用 57 维 masked-feature 分支，却没有给出 decoder 原始帧特征从哪里取得。
这是当前重建约定的实证后果，需核对原代码/作者设计；不能由论文自动推导出唯一修复。

### B2 · 注释末帧被人为赋予零运动增量

位置：`robotmdar/partigen/eos.py:20–22`、`robotmdar/partigen/data.py:46–54`；特征计算见 `robotmdar/dtype/motion.py:414–431`。

`primitive_targets(10,18,0,8)` 返回 raw 索引 `[8,9,10,11,12,13,14,15,16,17,17]`。
即使原动作中第 18 帧真实存在，annotation 最后一帧的 look-ahead 也会被替换为第 17 帧。
对于 `q[t]=t`，最后一帧的真实 `Δq=1` 被改为 **0**；根位移和 yaw 增量亦有类似问题。

这把语义结束与合成的零增量边界绑定，可能改变生成模型和 EOS 的学习目标。
论文 Table S3 定义相邻帧增量，但没有规定 annotation 末尾如何构造 look-ahead；README 已将重复末帧列为重建约定。
应核对端点语义与旧数据处理，再决定是否保留真实 look-ahead、只在物理序列末尾复制，或显式屏蔽无定义的增量通道。

### B3 · EOS 输入、标签和文本掩码尚待确认

位置：`robotmdar/partigen/networks.py:153–165`、`:189–206`；`robotmdar/partigen/training.py:91–97`。

当前 EOS 输入是 `[解码帧, 共享 latent, CLIP 文本, 归一化绝对帧位置]`；仅 annotation 最后一帧为正例。
这些具体输入、标注、秒转帧四舍五入及采样分布，不能从论文的逐帧 BCE 公式唯一确定。

此外，文本随机置零仅发生在 denoiser 内部，EOS 始终收到原始全文本。
实测将 denoiser 文本丢弃概率设为 1：去噪分支文本范数为 0，EOS 首层的 512 维文本仍完整保留。
论文没有说明 EOS 自身的输入及 mask 范围，因此这是需核验的设计差异，不能直接断言一定错误。

### B4 · 新旧 backbone 并不相同，旧 checkpoint 不能视为已兼容

新 denoiser 使用 TransformerDecoder 和历史 cross-attention，旧 `model/mld_denoiser.py` 用拼接 token 的 TransformerEncoder。
论文正文 3.4 明确写了历史 cross-attention，因此**不能仅凭新类是 TransformerDecoder 就判它违反论文**。

其它变化包括 VAE 跨层长 skip、token/position 初始化、timestep 的 learned embedding 与旧 sinusoidal+MLP 的差别。
论文未充分定义这些细节，结构大尺寸相同不能证明权重拓扑相同。
当前严格加载 checkpoint 的做法合理；服务器旧 checkpoint 的实际 key/shape 尚未读取，兼容性无法确认。

## C. 明确的复现范围缺口

1. `training.py:206–215` 只报告随机验证原语的损失，不是论文 §4/S3.2 的完整测试集评估。
   本目录未找到冻结 G1 GRU/CLIP evaluator、R-Precision/FID/Diversity/mm-dist/transition 指标、完整 320 帧生成后的时长误差，以及三训练 seed 的统计流程。
   因而无法验证论文的参考质量和时长指标。
2. 本目录没有恢复 S6 的 23→29 DoF、30→50 Hz、初始过渡与冻结 GRIT 跟踪评估/部署系统。
   README 已披露这些范围；它们可能在服务器其它仓库中，但未在本次审查中访问或验证。
3. 实际数据 split 是否 motion-disjoint、是否对应论文的训练/测试集合、缓存统计是否来自训练集、CLIP cache 来源以及服务器权重对应版本，都需要服务器资产确认。

## D. 旧入口仍与论文不一致

新增 `train_partigen_mvae` / `train_partigen_dar` 路由不使用旧训练参数；旧配置是按之前指示保留的历史代码。
若使用旧入口，仍可能运行另一套模型与目标：

- `config/train_dar_humanml3d_23dof_bodypart.yaml:21` 名称是 BodyPart，却选择 `vae: def`。
- `train/train_mvae.py` 使用 Adam；旧 `train/manager.py` 采用线性退火，旧 `config/train/mvae.yaml` 的步数总和也不同于论文。
- `config/data/humanml3d_30fps.yaml:4` 关于 57 维特征的注释错误：不是 6D root rotation/绝对 root translation/关节速度布局，真实 FeatureVersion 3 与 Table S3 相符。
- 旧 DAR 路径没有新增的帧级 EOS。

开源前应清楚区分论文入口和遗留实验，避免仅凭 BodyPart/DAR 文件名选择配置。
本次没有擅自修改或删除旧配置。

## E. 已核对一致的部分

| 项目 | 当前检查结论 |
| --- | --- |
| 57 维布局 | FeatureVersion 3 与 Table S3 对齐 |
| 解剖特征掩码 | 仅限制关节角 11–33；0–10、34–56 对所有头共享 |
| 头与投影 | 四组 DoF 对应四头，独立 57→128 Q/K/V，再拼接投影 |
| VAE 层排列 | 编解码各九层，`masked, masked, standard` 重复三次 |
| 共享 latent | 每原语一个共享 128 维向量，无独立部位 latent |
| 去噪网络 | 8 层、512 hidden、1024 FFN、4 头、GELU、历史 cross-attention |
| 扩散目标 | clean latent/START_X，cosine beta，固定后验方差，无 timestep rescale |
| 训练采样 | 每例均匀采样一个噪声时刻，单次去噪预测，真实历史 teacher forcing |
| CFG | 同噪声、同 timestep、同历史，文本有/无两个 clean-latent 预测 |
| EOS BCE | 正类系数与外层损失系数分开，有效帧均值 |
| EOS 推理 | 先完整生成，再判断；严格越阈值，保留命中帧，无命中保留全部 |
| 候选时长 | 默认候选 320 帧；8/16 帧原语对应 40/20 段 |
| DDPM | 全部反向更新，不裁剪 clean latent，最后一步不加噪声 |
| 损失加权 | Huber δ=1，逐项加权求和；VAE 外层几何系数；DAR 无 KL |
| foot contact | GT contact mask，非接触零元素仍在有效帧均值分母中 |
| smooth | 对关节角使用二阶差分 |
| 四元数组件损失 | 与论文披露的组件 Huber 一致；不能为“修正”而直接换成测地线损失 |

## F. 配置与验证状态

新 MVAE 和 DAR 模板分别有 43、35 个必填字段。按用户要求，训练超参数没有补默认值。
因此表 E 说明的是实现机制；实际训练步数、学习率、系数、dropout、扩散步数、阶段激活和随机 seed，仍需对最终服务器实验配置单独核验。

已运行：现有 13 项 CPU 测试（全部通过）、Hydra 配置组合/必填字段检查、真实模型 decoder hook 诊断、EOS look-ahead 和文本 mask 诊断、旋转数值诊断、调用实际损失函数的 root-only 反例。
没有进行真实 G1 数据训练、真实机器人 XML FK 数值验证、服务器 checkpoint 加载或论文指标复现。

建议处理顺序：先修复 A1/A2，再明确 A3 的终端约定；核对 B1/B2/B3 后固定模型和标签语义；最后连接服务器 evaluator 和实验配置验证。
