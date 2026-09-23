"""
Improved Length Predictor Training Script (v2)

Improvements over v1:
1. Multi-annotation: Use ALL text annotations per motion (2.88x more data)
2. Better model: Transformer encoder instead of MLP
3. Better loss: MSE + relative error penalty
4. Data augmentation: Random text selection during training

Usage:
    python train_length_predictor_v2.py \
        --datadir ./dataset/HumanML3D-G1-23DOF-30fps \
        --save_dir ./logs/length_predictor_v2 \
        --project len_predict

Author: TextOp Team
Date: 2026-01-03
"""

import os
import sys
import argparse
import joblib
from pathlib import Path
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
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


# =============================================================================
# Model Definitions
# =============================================================================

class LengthPredictorMLP(nn.Module):
    """Improved MLP with residual connections."""
    
    def __init__(
        self,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        mean_frames: float = 150.0,
    ):
        super().__init__()
        self.mean_frames = mean_frames
        
        self.input_proj = nn.Linear(text_dim, hidden_dim)
        
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
        
        self.output_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.constant_(self.output_head.bias, mean_frames)
    
    def forward(self, text_emb):
        x = self.input_proj(text_emb)
        for block in self.blocks:
            x = x + block(x)
        out = self.output_head(x)
        out = F.relu(out) + 30.0
        return out


class LengthPredictorTransformer(nn.Module):
    """Transformer-based length predictor for better text understanding."""
    
    def __init__(
        self,
        text_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        mean_frames: float = 150.0,
    ):
        super().__init__()
        self.mean_frames = mean_frames
        
        # Project text embedding to hidden dim
        self.input_proj = nn.Linear(text_dim, hidden_dim)
        
        # Learnable query token
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer decoder layers (query attends to text)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize output bias to mean
        nn.init.constant_(self.output_head[-1].bias, mean_frames)
    
    def forward(self, text_emb):
        """
        Args:
            text_emb: [B, text_dim] or [B, num_texts, text_dim]
        """
        # Handle both single and multiple text embeddings
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)  # [B, 1, text_dim]
        
        B = text_emb.shape[0]
        
        # Project to hidden dim
        memory = self.input_proj(text_emb)  # [B, num_texts, hidden_dim]
        
        # Expand query token for batch
        query = self.query_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        
        # Transformer decoding
        out = self.transformer(query, memory)  # [B, 1, hidden_dim]
        
        # Predict duration
        out = self.output_head(out.squeeze(1))  # [B, 1]
        out = F.relu(out) + 30.0
        return out


# =============================================================================
# Dataset
# =============================================================================

class MultiAnnotationDataset(Dataset):
    """Dataset that uses ALL text annotations per motion for training."""
    
    def __init__(self, datadir, split='train', fps=30.0, expand_annotations=True):
        self.datadir = Path(datadir)
        self.split = split
        self.fps = fps
        self.expand_annotations = expand_annotations
        
        # Load raw data
        data_file = self.datadir / f'{split}.pkl'
        print(f"Loading data from {data_file}...")
        raw_data = joblib.load(data_file)
        print(f"Loaded {len(raw_data)} motion samples")
        
        # Load CLIP model on GPU for faster encoding
        print("Loading CLIP model...")
        try:
            import clip
            self.clip_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.clip_model, _ = clip.load("ViT-B/32", device=self.clip_device)
            self.clip_model.eval()
            self.clip = clip
            print(f"CLIP model loaded on {self.clip_device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP: {e}")
        
        # Extract samples - expand each motion to multiple (text, duration) pairs
        self.samples = []
        duration_list = []
        
        for item in raw_data:
            if 'frame_ann' not in item or len(item['frame_ann']) == 0:
                continue
            
            duration = int(item.get('length', item['motion'].get('motion_len', 0)))
            if duration == 0:
                continue
            
            if expand_annotations:
                # Create one sample per annotation
                for ann in item['frame_ann']:
                    if len(ann) >= 3 and ann[2]:
                        self.samples.append({
                            'text': ann[2],
                            'duration_frames': duration,
                        })
                        duration_list.append(duration)
            else:
                # Only use first annotation
                text = item['frame_ann'][0][2] if len(item['frame_ann'][0]) >= 3 else None
                if text:
                    self.samples.append({
                        'text': text,
                        'duration_frames': duration,
                    })
                    duration_list.append(duration)
        
        print(f"Created {len(self.samples)} (text, duration) pairs")
        
        # Precompute all embeddings for efficiency
        print("Precomputing CLIP embeddings...")
        self._precompute_embeddings()
        
        # Statistics
        self.duration_mean = np.mean(duration_list)
        self.duration_std = np.std(duration_list)
        self.duration_max = max(duration_list)
        self.duration_min = min(duration_list)
        print(f"Duration stats: mean={self.duration_mean:.1f}, std={self.duration_std:.1f}, "
              f"min={self.duration_min}, max={self.duration_max}")
    
    def _precompute_embeddings(self):
        """Precompute CLIP embeddings for all unique texts (on GPU)."""
        unique_texts = list(set(s['text'] for s in self.samples))
        print(f"Computing embeddings for {len(unique_texts)} unique texts on {self.clip_device}...")
        
        self.text_to_embedding = {}
        batch_size = 512  # Larger batch size for GPU
        
        with torch.no_grad():
            for i in tqdm(range(0, len(unique_texts), batch_size), desc="CLIP encoding"):
                batch_texts = unique_texts[i:i+batch_size]
                tokens = self.clip.tokenize(batch_texts, truncate=True).to(self.clip_device)
                embeddings = self.clip_model.encode_text(tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu()  # Move back to CPU for storage
                
                for text, emb in zip(batch_texts, embeddings):
                    self.text_to_embedding[text] = emb.float()
        
        # Free GPU memory
        del self.clip_model
        torch.cuda.empty_cache()
        print(f"Cached {len(self.text_to_embedding)} embeddings")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text_emb = self.text_to_embedding[sample['text']]
        
        return {
            'text_embedding': text_emb,
            'duration_frames': torch.tensor(sample['duration_frames'], dtype=torch.float32),
        }


# =============================================================================
# Loss Functions
# =============================================================================

class CombinedLoss(nn.Module):
    """Combined MSE + relative error loss."""
    
    def __init__(self, mse_weight=1.0, rel_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.rel_weight = rel_weight
    
    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        
        # Relative error: penalize percentage error
        rel_error = torch.abs(pred - target) / (target + 1e-6)
        rel_loss = rel_error.mean()
        
        return self.mse_weight * mse_loss + self.rel_weight * rel_loss


# =============================================================================
# Training
# =============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize WandB
    use_wandb = WANDB_AVAILABLE and args.project
    if use_wandb:
        try:
            wandb.init(
                project=args.project,
                name=args.run_name or f"len_pred_v2_{args.model_type}",
                config=vars(args),
            )
            print(f"WandB initialized: project={args.project}")
        except Exception as e:
            print(f"WandB init failed: {e}")
            use_wandb = False
    
    # Load datasets
    train_dataset = MultiAnnotationDataset(
        args.datadir, split='train', fps=args.fps, 
        expand_annotations=args.expand_annotations
    )
    val_dataset = MultiAnnotationDataset(
        args.datadir, split='val', fps=args.fps,
        expand_annotations=False  # Don't expand for validation
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    # Create model
    text_dim = train_dataset.text_to_embedding[list(train_dataset.text_to_embedding.keys())[0]].shape[-1]
    mean_frames = train_dataset.duration_mean
    
    if args.model_type == 'mlp':
        model = LengthPredictorMLP(
            text_dim=text_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            mean_frames=mean_frames,
        )
    else:
        model = LengthPredictorTransformer(
            text_dim=text_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            mean_frames=mean_frames,
        )
    
    model = model.to(device)
    print(f"Model: {args.model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Loss
    criterion = CombinedLoss(mse_weight=1.0, rel_weight=args.rel_weight)
    
    # Training
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_mae = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_mae = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False):
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
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                text_emb = batch['text_embedding'].to(device)
                target = batch['duration_frames'].unsqueeze(-1).to(device)
                
                pred = model(text_emb)
                loss = criterion(pred, target)
                
                val_loss += loss.item()
                val_mae += torch.abs(pred - target).mean().item()
                val_preds.extend(pred.cpu().numpy().flatten())
                val_targets.extend(target.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        # Compute additional metrics
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        rel_error = np.abs(val_preds - val_targets) / (val_targets + 1e-6)
        val_mape = rel_error.mean() * 100  # Mean Absolute Percentage Error
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, MAE={train_mae:.2f} | "
              f"Val Loss={val_loss:.4f}, MAE={val_mae:.2f} frames, MAPE={val_mape:.1f}%")
        
        # Log to WandB
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/mae_frames': train_mae,
                'val/loss': val_loss,
                'val/mae_frames': val_mae,
                'val/mae_seconds': val_mae / args.fps,
                'val/mape': val_mape,
                'lr': scheduler.get_last_lr()[0],
            })
        
        # Save best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'val_mape': val_mape,
                'config': {
                    'model_type': args.model_type,
                    'text_dim': text_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_heads': args.num_heads,
                    'dropout': args.dropout,
                    'mean_frames': mean_frames,
                },
            }, save_dir / 'length_predictor_best.pth')
            print(f"  -> Saved best model (MAE={val_mae:.2f} frames, MAPE={val_mape:.1f}%)")
    
    # Finish WandB
    if use_wandb:
        wandb.log({'best_val_mae_frames': best_val_mae})
        wandb.finish()
    
    print(f"\nTraining complete! Best MAE: {best_val_mae:.2f} frames ({best_val_mae/args.fps:.2f}s)")


def test_inference(args):
    """Test the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    ckpt_path = Path(args.save_dir) / 'length_predictor_best.pth'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    config = ckpt['config']
    if config.get('model_type', 'mlp') == 'transformer':
        model = LengthPredictorTransformer(
            text_dim=config['text_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config.get('num_heads', 4),
            dropout=config['dropout'],
            mean_frames=config['mean_frames'],
        )
    else:
        model = LengthPredictorMLP(
            text_dim=config['text_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            mean_frames=config['mean_frames'],
        )
    
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {ckpt_path}")
    print(f"Validation MAE: {ckpt['val_mae']:.2f} frames ({ckpt['val_mae']/30:.2f}s)")
    print(f"Validation MAPE: {ckpt.get('val_mape', 'N/A')}")
    
    # Load validation data
    val_dataset = MultiAnnotationDataset(args.datadir, split='val', expand_annotations=False)
    
    print("\nSample predictions:")
    for i in range(min(15, len(val_dataset))):
        sample = val_dataset[i]
        text_emb = sample['text_embedding'].unsqueeze(0).to(device)
        target = sample['duration_frames'].item()
        
        with torch.no_grad():
            pred = model(text_emb).item()
        
        text = val_dataset.samples[i]['text'][:60]
        error = abs(pred - target)
        rel_error = error / target * 100
        
        status = "✓" if error < 15 else "✗"
        print(f"  {status} {text}...")
        print(f"    Target: {target:.0f} ({target/30:.1f}s), Pred: {pred:.0f} ({pred/30:.1f}s), "
              f"Error: {error:.0f} frames ({rel_error:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./dataset/HumanML3D-G1-23DOF-30fps')
    parser.add_argument('--save_dir', type=str, default='./logs/length_predictor_v2')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--rel_weight', type=float, default=0.1, help='Weight for relative error loss')
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'transformer'])
    parser.add_argument('--expand_annotations', action='store_true', default=True,
                        help='Use all annotations per motion (data augmentation)')
    parser.add_argument('--no_expand', action='store_false', dest='expand_annotations')
    parser.add_argument('--test', action='store_true', help='Run inference test')
    # WandB
    parser.add_argument('--project', type=str, default='', help='WandB project name')
    parser.add_argument('--run_name', type=str, default='', help='WandB run name')
    
    args = parser.parse_args()
    
    if args.test:
        test_inference(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
