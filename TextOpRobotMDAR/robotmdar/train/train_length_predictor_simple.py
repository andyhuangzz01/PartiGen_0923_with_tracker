"""
Simple Training Script for Length Predictor

This script trains a simple MLP to predict motion duration from text embeddings.
Designed to work with the existing HumanML3D-G1 dataset format.

Usage:
    python train_length_predictor_simple.py --datadir ./dataset/HumanML3D-G1-23DOF-30fps --project len_predict

Author: TextOp Team
Date: 2026-01-02
"""

import os
import sys
import argparse
import joblib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not installed, logging disabled")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class LengthPredictor(nn.Module):
    """MLP for predicting motion duration from text embeddings.
    
    改进版本：
    1. 直接回归帧数（不用Sigmoid归一化）
    2. 使用残差连接
    3. 更深的网络 + 更宽的隐藏层
    """
    
    def __init__(
        self,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_frames: int = 196,
        mean_frames: float = 150.0,  # 用于初始化bias
    ):
        super().__init__()
        
        self.max_frames = max_frames
        self.mean_frames = mean_frames
        
        # Input projection
        self.input_proj = nn.Linear(text_dim, hidden_dim)
        
        # Residual MLP blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ))
        
        # Output head - 直接输出帧数
        self.output_head = nn.Linear(hidden_dim, 1)
        
        # 初始化output bias为均值，让模型从均值开始学习
        nn.init.zeros_(self.output_head.weight)
        nn.init.constant_(self.output_head.bias, mean_frames)
    
    def forward(self, text_emb):
        """
        Args:
            text_emb: [B, text_dim]
        Returns:
            duration_frames: [B, 1] predicted frame count (可以是任意正数)
        """
        x = self.input_proj(text_emb)
        
        for block in self.blocks:
            x = x + block(x)  # 残差连接
        
        out = self.output_head(x)
        # 用ReLU确保输出为正，加一个小的最小值
        out = torch.relu(out) + 30.0  # 至少30帧（1秒）
        return out


class LengthDataset(Dataset):
    """Dataset for length prediction training."""
    
    def __init__(self, datadir, split='train', fps=30.0, use_clip=True):
        self.datadir = Path(datadir)
        self.split = split
        self.fps = fps
        self.use_clip = use_clip
        
        # Load raw data
        data_file = self.datadir / f'{split}.pkl'
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        print(f"Loading data from {data_file}...")
        raw_data = joblib.load(data_file)
        print(f"Loaded {len(raw_data)} samples")
        
        # Try to load precomputed text embeddings
        text_emb_file = self.datadir / f'{split}_text_embed.pkl'
        self.text_embeddings = None
        if text_emb_file.exists():
            print(f"Loading text embeddings from {text_emb_file}...")
            self.text_embeddings = torch.load(text_emb_file, map_location='cpu')
            # 检查是否为空
            if isinstance(self.text_embeddings, dict) and len(self.text_embeddings) > 0:
                first_val = list(self.text_embeddings.values())[0]
                if hasattr(first_val, 'sum') and first_val.sum().item() == 0:
                    print("WARNING: Text embeddings are all zeros! Will compute on-the-fly.")
                    self.text_embeddings = None
        
        # If no valid embeddings, load CLIP model
        self.clip_model = None
        self.clip_preprocess = None
        if self.text_embeddings is None and use_clip:
            print("Loading CLIP model for on-the-fly embedding...")
            try:
                import clip
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
                self.clip_model.eval()
                print("CLIP model loaded successfully")
            except Exception as e:
                print(f"Failed to load CLIP: {e}")
                print("Will use random embeddings (for testing only)")
        
        # Extract samples with text and duration
        self.samples = []
        duration_list = []
        
        for item in raw_data:
            # 从 frame_ann 获取文本
            text = None
            if 'frame_ann' in item and len(item['frame_ann']) > 0:
                # frame_ann 格式: [(start, end, text, labels), ...]
                ann = item['frame_ann'][0]  # 取第一个标注
                if len(ann) >= 3:
                    text = ann[2]  # 第三个元素是文本
            elif 'text' in item:
                text = item.get('text', '')
            
            if not text:
                continue
            
            # Get duration
            if 'length' in item:
                duration = int(item['length'])
            elif 'motion' in item and 'motion_len' in item['motion']:
                duration = int(item['motion']['motion_len'])
            else:
                continue
            
            self.samples.append({
                'text': text,
                'duration_frames': duration,
            })
            duration_list.append(duration)
        
        print(f"Found {len(self.samples)} valid samples")
        
        # Statistics
        if len(duration_list) > 0:
            self.duration_mean = np.mean(duration_list)
            self.duration_std = np.std(duration_list)
            self.duration_max = max(duration_list)
            self.duration_min = min(duration_list)
            print(f"Duration stats: mean={self.duration_mean:.1f}, std={self.duration_std:.1f}, "
                  f"min={self.duration_min}, max={self.duration_max}")
    
    def __len__(self):
        return len(self.samples)
    
    def _get_clip_embedding(self, text):
        """Get CLIP embedding for text."""
        if self.text_embeddings is not None and text in self.text_embeddings:
            emb = self.text_embeddings[text]
            if isinstance(emb, np.ndarray):
                return torch.from_numpy(emb).float()
            return emb.float()
        
        if self.clip_model is not None:
            import clip
            with torch.no_grad():
                tokens = clip.tokenize([text], truncate=True)
                emb = self.clip_model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)  # Normalize
                return emb.squeeze(0).float()
        
        # Fallback: random embedding (for testing)
        return torch.randn(512)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        text_emb = self._get_clip_embedding(text)
        
        return {
            'text_embedding': text_emb.float(),
            'duration_frames': torch.tensor(sample['duration_frames'], dtype=torch.float32),
        }


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize WandB
    use_wandb = WANDB_AVAILABLE and args.project
    if use_wandb:
        try:
            wandb.init(
                project=args.project,
                name=args.run_name or f"length_predictor_{args.hidden_dim}d_{args.num_layers}L",
                config=vars(args),
            )
            print(f"WandB initialized: project={args.project}")
        except Exception as e:
            print(f"WandB init failed: {e}, continuing without logging")
            use_wandb = False
    
    # Load datasets
    train_dataset = LengthDataset(args.datadir, split='train', fps=args.fps)
    val_dataset = LengthDataset(args.datadir, split='val', fps=args.fps)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Get dimensions
    sample = train_dataset[0]
    text_dim = sample['text_embedding'].shape[-1]
    max_frames = int(train_dataset.duration_max * 1.1)  # Add 10% margin
    mean_frames = train_dataset.duration_mean
    
    print(f"Text dim: {text_dim}, Max frames: {max_frames}, Mean frames: {mean_frames:.1f}")
    
    # Create model
    model = LengthPredictor(
        text_dim=text_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_frames=max_frames,
        mean_frames=mean_frames,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss
    criterion = nn.MSELoss()
    
    # Training
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_mae = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_mae = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            text_emb = batch['text_embedding'].to(device)
            target = batch['duration_frames'].unsqueeze(-1).to(device)
            
            pred = model(text_emb)
            loss = criterion(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.abs(pred - target).mean().item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_mae = 0
        
        with torch.no_grad():
            for batch in val_loader:
                text_emb = batch['text_embedding'].to(device)
                target = batch['duration_frames'].unsqueeze(-1).to(device)
                
                pred = model(text_emb)
                loss = criterion(pred, target)
                
                val_loss += loss.item()
                val_mae += torch.abs(pred - target).mean().item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, MAE={train_mae:.2f} | "
              f"Val Loss={val_loss:.4f}, MAE={val_mae:.2f} frames")
        
        # Log to WandB
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/mae_frames': train_mae,
                'train/mae_seconds': train_mae / args.fps,
                'val/loss': val_loss,
                'val/mae_frames': val_mae,
                'val/mae_seconds': val_mae / args.fps,
                'lr': scheduler.get_last_lr()[0],
            })
        
        # Save best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'config': {
                    'text_dim': text_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'max_frames': max_frames,
                    'mean_frames': mean_frames,
                },
            }, save_dir / 'length_predictor_best.pth')
            print(f"  -> Saved best model (MAE={val_mae:.2f} frames)")
    
    # Finish WandB
    if use_wandb:
        wandb.log({'best_val_mae_frames': best_val_mae, 'best_val_mae_seconds': best_val_mae / args.fps})
        wandb.finish()
    
    print(f"\nTraining complete! Best MAE: {best_val_mae:.2f} frames")
    print(f"At {args.fps} fps, this is {best_val_mae/args.fps:.2f} seconds error")


def test_inference(args):
    """Test the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    ckpt_path = Path(args.save_dir) / 'length_predictor_best.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    
    config = ckpt['config']
    model = LengthPredictor(**config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {ckpt_path}")
    print(f"Validation MAE: {ckpt['val_mae']:.2f} frames")
    
    # Load some test samples
    val_dataset = LengthDataset(args.datadir, split='val')
    
    # 诊断：检查embedding是否有差异
    print("\n=== Embedding Diagnosis ===")
    embeddings = []
    for i in range(min(100, len(val_dataset))):
        emb = val_dataset[i]['text_embedding']
        embeddings.append(emb)
    embeddings = torch.stack(embeddings)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
    print(f"Embedding min: {embeddings.min():.4f}, max: {embeddings.max():.4f}")
    
    # 检查不同样本的embedding是否相同
    emb_diff = (embeddings[0] - embeddings[1]).abs().sum().item()
    print(f"Diff between sample 0 and 1: {emb_diff:.4f}")
    emb_diff2 = (embeddings[0] - embeddings[10]).abs().sum().item()
    print(f"Diff between sample 0 and 10: {emb_diff2:.4f}")
    
    print("\nSample predictions:")
    for i in range(min(10, len(val_dataset))):
        sample = val_dataset[i]
        text_emb = sample['text_embedding'].unsqueeze(0).to(device)
        target = sample['duration_frames'].item()
        
        with torch.no_grad():
            pred = model(text_emb).item()
        
        text = val_dataset.samples[i]['text'][:50] + "..."
        print(f"  {text}")
        print(f"    Target: {target:.0f} frames ({target/30:.1f}s), Pred: {pred:.0f} frames ({pred/30:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./dataset/HumanML3D-G1-23DOF-30fps')
    parser.add_argument('--save_dir', type=str, default='./logs/length_predictor')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--test', action='store_true', help='Run inference test')
    # WandB arguments
    parser.add_argument('--project', type=str, default='', help='WandB project name')
    parser.add_argument('--run_name', type=str, default='', help='WandB run name')
    
    args = parser.parse_args()
    
    if args.test:
        test_inference(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
