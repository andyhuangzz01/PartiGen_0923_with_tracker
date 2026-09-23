"""
Adaptive Length DAR 训练脚本

训练流程:
1. 加载预训练的 VAE (冻结)
2. 加载预训练的 DAR 权重 (可选)
3. 加载预训练的 Length Predictor 权重 (可选)
4. 训练 Adaptive Length DAR

特点:
- 模型自动预测动作时长
- 支持从零训练或微调
- 支持联合训练或冻结 Length Predictor
"""

import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from hydra.utils import instantiate
import wandb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from robotmdar.dtype import seed, logger
from robotmdar.dtype.abc import VAE, Dataset, Denoiser
from robotmdar.model.adaptive_length_denoiser import AdaptiveLengthDenoiser
from robotmdar.utils.train_utils import ManagerDenoiser


# ============== 配置 ==============
class Config:
    # 实验设置
    exp_name = "HumanML3D-G1-23DOF-Adaptive-DAR"
    seed = 42
    device = "cuda"
    gpu_id = 0
    
    # 数据集
    data_dir = "dataset/HumanML3D-G1-23DOF-30fps"
    batch_size = 512
    num_workers = 4
    
    # 模型权重路径
    vae_ckpt = "logs/RobotMDAR/HumanML3D-G1-23DOF-VAE/train-mvae-20251231_060449/ckpt_100000.pth"
    pretrained_dar_ckpt = None  # 从零训练 Adaptive DAR
    length_predictor_ckpt = "logs/length_predictor/best_model_v3.pth"  # 如果没有下载,设为 None
    
    # 训练策略
    train_from_scratch = True  # True: 从零训练 DAR (推荐)
    freeze_length_predictor = False  # False: 联合训练 Length Predictor
    use_length_loss = True  # True: 添加 length 预测损失
    length_loss_weight = 0.1  # Length 损失权重 (可以调整为 0.05-0.2)
    
    # 训练超参数 (遵循原始 DAR 配置)
    max_steps = 300000  # 300k steps (分阶段训练)
    learning_rate = 1e-4
    weight_decay = 0.0
    gradient_clip = 1.0
    
    # 学习率调度 (与原始 DAR 一致)
    lr_scheduler = "cosine"  # cosine 或 constant
    warmup_steps = 5000
    
    # 保存和验证
    save_every = 10000  # 每 10k 步保存一次
    log_every = 100  # 每 100 步记录一次
    eval_every = 5000  # 每 5k 步验证一次
    
    # WandB
    use_wandb = True
    wandb_project = "TextOp-RobotMDAR"
    wandb_entity = None  # 你的 wandb username


def setup_experiment(cfg):
    """设置实验环境"""
    # 设置随机种子
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    
    # 创建输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("logs/RobotMDAR") / cfg.exp_name / f"train-dar-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Experiment: {cfg.exp_name}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*80}\n")
    
    return output_dir


def load_config():
    """加载模型配置"""
    print("Loading configuration files...")
    
    base_cfg = OmegaConf.load("robotmdar/config/base.yaml")
    skeleton_cfg = OmegaConf.load("robotmdar/config/skeleton/g1.yaml")
    data_cfg = OmegaConf.load("robotmdar/config/data/humanml3d_30fps.yaml")
    vae_cfg = OmegaConf.load("robotmdar/config/vae/skip_vae.yaml")
    denoiser_cfg = OmegaConf.load("robotmdar/config/denoiser/adaptive_length.yaml")
    diffusion_cfg = OmegaConf.load("robotmdar/config/diffusion/def.yaml")
    
    cfg = OmegaConf.merge(
        base_cfg,
        {"skeleton": skeleton_cfg},
        {"data": data_cfg},
        {"vae": vae_cfg},
        {"denoiser": denoiser_cfg},
        {"diffusion": diffusion_cfg}
    )
    
    OmegaConf.resolve(cfg)
    return cfg


def load_vae(cfg, model_cfg, device):
    """加载预训练的 VAE (冻结)"""
    print(f"\nLoading VAE from {cfg.vae_ckpt}")
    
    vae_ckpt = torch.load(cfg.vae_ckpt, map_location=device)
    model_cfg.vae.nfeats = model_cfg.data.motion.nfeats
    
    vae = instantiate(model_cfg.vae, _recursive_=False)
    vae.load_state_dict(vae_ckpt['model'])
    vae.to(device)
    vae.eval()
    
    # 冻结 VAE
    for param in vae.parameters():
        param.requires_grad = False
    
    print(f"  VAE loaded and frozen (epoch {vae_ckpt.get('epoch', 'unknown')})")
    return vae


def load_denoiser(cfg, model_cfg, vae, device):
    """创建并初始化 Denoiser"""
    print(f"\nInitializing Adaptive Length Denoiser...")
    
    nfeats = model_cfg.data.motion.nfeats
    model_cfg.denoiser.history_shape = [2, nfeats]
    model_cfg.denoiser.noise_shape = [1, vae.latent_dim]
    
    denoiser = instantiate(model_cfg.denoiser, _recursive_=False)
    denoiser.to(device)
    
    print(f"  Architecture: h_dim={denoiser.h_dim}, layers={denoiser.num_layers}")
    print(f"  Max frames: {denoiser.max_frames}")
    print(f"  Use length predictor: {denoiser.use_length_predictor}")
    
    # 加载预训练 DAR 权重 (可选)
    if cfg.pretrained_dar_ckpt and not cfg.train_from_scratch:
        print(f"\n  Loading pretrained DAR weights from {cfg.pretrained_dar_ckpt}")
        dar_ckpt = torch.load(cfg.pretrained_dar_ckpt, map_location=device)
        
        # 尝试加载，strict=False 因为可能缺少 length predictor 权重
        missing, unexpected = denoiser.load_state_dict(dar_ckpt['model'], strict=False)
        print(f"  Loaded DAR weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
        
        if missing:
            print(f"  Missing keys (will be randomly initialized): {missing[:5]}...")
    
    # 加载预训练 Length Predictor 权重 (可选)
    if cfg.length_predictor_ckpt and denoiser.use_length_predictor:
        print(f"\n  Loading pretrained Length Predictor from {cfg.length_predictor_ckpt}")
        try:
            denoiser.load_length_predictor_weights(cfg.length_predictor_ckpt)
        except Exception as e:
            print(f"  [Warning] Failed to load: {e}")
    
    # 冻结 Length Predictor (可选)
    if cfg.freeze_length_predictor and denoiser.use_length_predictor:
        print(f"\n  Freezing Length Predictor weights")
        for param in denoiser.length_predictor.parameters():
            param.requires_grad = False
    
    return denoiser


def create_dataloaders(cfg, model_cfg):
    """创建数据加载器 (使用 Hydra 配置实例化)"""
    print(f"\nCreating dataloaders from {cfg.data_dir}")
    
    # 使用 Hydra 实例化数据集 (与原始 DAR 训练一致)
    train_dataset: Dataset = instantiate(model_cfg.data.train)
    val_dataset: Dataset = instantiate(model_cfg.data.val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


def compute_loss(denoiser, diffusion, vae, batch, cfg):
    """
    计算训练损失
    
    包括:
    1. Diffusion 损失 (noise prediction)
    2. Length 预测损失 (可选)
    """
    device = next(denoiser.parameters()).device
    
    # 解包数据
    motion = batch['motion'].to(device)  # [B, T, nfeats]
    text_emb = batch['text_emb'].to(device)  # [B, 512]
    history = batch['history'].to(device)  # [B, 2, nfeats]
    motion_length = batch['motion_length'].to(device)  # [B]
    
    batch_size = motion.shape[0]
    
    # VAE 编码 (冻结)
    with torch.no_grad():
        latent = vae.encode(motion)  # [B, 1, latent_dim]
    
    # 采样时间步
    t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
    
    # 采样噪声
    noise = torch.randn_like(latent)
    
    # 前向扩散
    x_t = diffusion.q_sample(latent, t, noise)
    
    # 准备条件
    y = {
        'text': text_emb,
        'history': history,
        'duration': motion_length,  # 提供 GT duration
    }
    
    # 预测噪声
    if cfg.use_length_loss and denoiser.use_length_predictor:
        # 同时返回 noise 和 length 预测
        predicted_noise, predicted_length = denoiser(x_t, t, y, return_length_pred=True)
        
        # Noise 损失
        noise_loss = F.mse_loss(predicted_noise, noise)
        
        # Length 损失
        length_loss = F.mse_loss(predicted_length, motion_length.float())
        
        # 总损失
        total_loss = noise_loss + cfg.length_loss_weight * length_loss
        
        return {
            'total_loss': total_loss,
            'noise_loss': noise_loss.item(),
            'length_loss': length_loss.item(),
        }
    else:
        # 只预测 noise
        predicted_noise = denoiser(x_t, t, y)
        noise_loss = F.mse_loss(predicted_noise, noise)
        
        return {
            'total_loss': noise_loss,
            'noise_loss': noise_loss.item(),
        }


def train_epoch(denoiser, diffusion, vae, train_loader, optimizer, cfg, epoch, global_step, output_dir):
    """训练一个 epoch"""
    denoiser.train()
    epoch_losses = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # 计算损失
        loss_dict = compute_loss(denoiser, diffusion, vae, batch, cfg)
        total_loss = loss_dict['total_loss']
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        if cfg.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), cfg.gradient_clip)
        
        optimizer.step()
        
        # 记录
        epoch_losses.append(loss_dict['noise_loss'])
        global_step += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss_dict['noise_loss']:.4f}",
            'step': global_step
        })
        
        # WandB 记录
        if cfg.use_wandb and global_step % cfg.log_every == 0:
            log_dict = {
                'train/noise_loss': loss_dict['noise_loss'],
                'train/step': global_step,
                'train/epoch': epoch,
            }
            if 'length_loss' in loss_dict:
                log_dict['train/length_loss'] = loss_dict['length_loss']
            wandb.log(log_dict)
        
        # 保存 checkpoint
        if global_step % cfg.save_every == 0:
            save_checkpoint(denoiser, optimizer, epoch, global_step, output_dir)
    
    return global_step, sum(epoch_losses) / len(epoch_losses)


@torch.no_grad()
def validate(denoiser, diffusion, vae, val_loader, cfg):
    """验证"""
    denoiser.eval()
    val_losses = []
    
    for batch in tqdm(val_loader, desc="Validation"):
        loss_dict = compute_loss(denoiser, diffusion, vae, batch, cfg)
        val_losses.append(loss_dict['noise_loss'])
    
    return sum(val_losses) / len(val_losses)


def save_checkpoint(denoiser, optimizer, epoch, global_step, output_dir):
    """保存 checkpoint"""
    ckpt_path = output_dir / f"ckpt_{global_step}.pth"
    
    torch.save({
        'model': denoiser.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
    }, ckpt_path)
    
    print(f"\n  Saved checkpoint to {ckpt_path}")


def main():
    cfg = Config()
    
    # 设置实验
    output_dir = setup_experiment(cfg)
    
    # WandB 初始化
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.exp_name,
            config=vars(cfg),
        )
    
    # 加载模型配置
    model_cfg = load_config()
    
    # 加载模型
    vae = load_vae(cfg, model_cfg, cfg.device)
    denoiser = load_denoiser(cfg, model_cfg, vae, cfg.device)
    
    # 创建 Diffusion
    diffusion = instantiate(model_cfg.diffusion, denoiser=denoiser)
    diffusion.to(cfg.device)
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(cfg, model_cfg)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    
    # 学习率调度器 (与原始 DAR 一致)
    if cfg.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps, eta_min=1e-6)
    else:
        scheduler = None
    
    # 打印可训练参数
    total_params = sum(p.numel() for p in denoiser.parameters())
    trainable_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # 训练循环 (基于 steps 而非 epochs)
    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"Max steps: {cfg.max_steps:,} (分阶段: 50k + 50k + 50k + 150k)")
    print(f"{'='*80}\n")
    
    global_step = 0
    best_val_loss = float('inf')
    epoch = 0
    
    while global_step < cfg.max_steps:
        epoch += 1
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} | Step {global_step}/{cfg.max_steps}")
        print(f"{'='*80}")
        
        # 训练一个 epoch
        denoiser.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 检查是否达到最大步数
            if global_step >= cfg.max_steps:
                break
            
            # Warmup 学习率
            if global_step < cfg.warmup_steps:
                lr = cfg.learning_rate * (global_step + 1) / cfg.warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # 计算损失
            loss_dict = compute_loss(denoiser, diffusion, vae, batch, cfg)
            total_loss = loss_dict['total_loss']
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), cfg.gradient_clip)
            
            optimizer.step()
            
            # 学习率调度
            if scheduler and global_step >= cfg.warmup_steps:
                scheduler.step()
            
            # 记录
            epoch_losses.append(loss_dict['noise_loss'])
            global_step += 1
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss_dict['noise_loss']:.4f}",
                'step': f"{global_step}/{cfg.max_steps}",
                'lr': f"{current_lr:.2e}"
            })
            
            # WandB 记录
            if cfg.use_wandb and global_step % cfg.log_every == 0:
                log_dict = {
                    'train/noise_loss': loss_dict['noise_loss'],
                    'train/step': global_step,
                    'train/epoch': epoch,
                    'train/lr': current_lr,
                }
                if 'length_loss' in loss_dict:
                    log_dict['train/length_loss'] = loss_dict['length_loss']
                wandb.log(log_dict)
            
            # 保存 checkpoint
            if global_step % cfg.save_every == 0:
                save_checkpoint(denoiser, optimizer, epoch, global_step, output_dir)
            
            # 验证
            if global_step % cfg.eval_every == 0:
                val_loss = validate(denoiser, diffusion, vae, val_loader, cfg)
                print(f"\n  Step {global_step} | Val Loss: {val_loss:.4f}")
                
                if cfg.use_wandb:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/step': global_step,
                    })
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt_path = output_dir / "ckpt_best.pth"
                    torch.save({
                        'model': denoiser.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'val_loss': val_loss,
                    }, best_ckpt_path)
                    print(f"  ✅ Saved best model (val_loss={val_loss:.4f})")
                
                denoiser.train()  # 恢复训练模式
        
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\n  Epoch {epoch} | Train Loss: {avg_loss:.4f}")
    
    # 保存最终模型
    final_ckpt_path = output_dir / f"ckpt_{global_step}.pth"
    torch.save({
        'model': denoiser.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
    }, final_ckpt_path)
    
    print(f"\n{'='*80}")
    print(f"✅ Training completed!")
    print(f"Total steps: {global_step:,}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
