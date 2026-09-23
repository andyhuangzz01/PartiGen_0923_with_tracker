"""
Length Predictor Module

A lightweight MLP-based model that predicts motion duration from text embeddings.
This implements "Scheme 1: Predict-then-Generate" for variable-length motion generation.

Author: TextOp Team
Date: 2026-01-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LengthPredictor(nn.Module):
    """
    MLP-based length predictor that maps text embeddings to motion duration.
    
    Args:
        text_dim: Dimension of input text embeddings (default: 512 for CLIP)
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        dropout: Dropout rate
        output_type: 'frames' for frame count, 'seconds' for duration in seconds
        max_frames: Maximum number of frames (for normalization)
        fps: Frames per second (for seconds output)
    """
    
    def __init__(
        self,
        text_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_type: str = 'frames',  # 'frames' or 'seconds'
        max_frames: int = 196,  # HumanML3D max length
        fps: float = 30.0,
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.output_type = output_type
        self.max_frames = max_frames
        self.fps = fps
        
        # Build MLP layers
        layers = []
        in_dim = text_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer - predicts normalized duration [0, 1]
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # Output in [0, 1] range
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict motion duration from text embedding.
        
        Args:
            text_emb: Text embeddings [B, text_dim] or [B, seq_len, text_dim]
        
        Returns:
            Predicted duration [B, 1]
            - If output_type='frames': frame count (denormalized)
            - If output_type='seconds': duration in seconds
        """
        # Handle sequence input - take mean or first token
        if text_emb.dim() == 3:
            text_emb = text_emb.mean(dim=1)  # [B, text_dim]
        
        # Predict normalized duration [0, 1]
        normalized_duration = self.mlp(text_emb)  # [B, 1]
        
        # Denormalize to actual frames/seconds
        if self.output_type == 'frames':
            duration = normalized_duration * self.max_frames
        else:  # seconds
            duration = normalized_duration * (self.max_frames / self.fps)
        
        return duration
    
    def predict_primitives(
        self, 
        text_emb: torch.Tensor, 
        primitive_frames: int = 8,
        min_primitives: int = 1,
        max_primitives: int = 25,
    ) -> torch.Tensor:
        """
        Predict the number of primitives to generate.
        
        Args:
            text_emb: Text embeddings [B, text_dim]
            primitive_frames: Number of frames per primitive
            min_primitives: Minimum number of primitives
            max_primitives: Maximum number of primitives
        
        Returns:
            Number of primitives [B] (integer tensor)
        """
        frames = self.forward(text_emb)  # [B, 1]
        
        # Convert to primitives
        num_primitives = torch.ceil(frames / primitive_frames).squeeze(-1)
        
        # Clamp to valid range
        num_primitives = torch.clamp(num_primitives, min_primitives, max_primitives)
        
        return num_primitives.long()


class LengthPredictorWithUncertainty(LengthPredictor):
    """
    Length predictor that also outputs uncertainty estimation.
    Useful for deciding when to use default duration.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace final layer to output mean and variance
        # Remove the last two layers (Linear + Sigmoid)
        self.mlp = self.mlp[:-2]
        
        # Separate heads for mean and log_variance
        self.mean_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        self.logvar_head = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict motion duration with uncertainty.
        
        Args:
            text_emb: Text embeddings [B, text_dim]
        
        Returns:
            mean: Predicted duration [B, 1]
            std: Uncertainty (standard deviation) [B, 1]
        """
        if text_emb.dim() == 3:
            text_emb = text_emb.mean(dim=1)
        
        features = self.mlp(text_emb)  # [B, hidden_dim]
        
        # Predict mean (normalized)
        normalized_mean = self.mean_head(features)  # [B, 1]
        
        # Predict log variance
        logvar = self.logvar_head(features)  # [B, 1]
        std = torch.exp(0.5 * logvar)  # [B, 1]
        
        # Denormalize
        if self.output_type == 'frames':
            mean = normalized_mean * self.max_frames
            std = std * self.max_frames
        else:
            mean = normalized_mean * (self.max_frames / self.fps)
            std = std * (self.max_frames / self.fps)
        
        return mean, std
    
    def sample(self, text_emb: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample duration with uncertainty.
        
        Args:
            text_emb: Text embeddings [B, text_dim]
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Sampled duration [B, 1]
        """
        mean, std = self.forward(text_emb)
        
        if self.training or temperature > 0:
            # Sample from Gaussian
            eps = torch.randn_like(mean)
            duration = mean + temperature * std * eps
        else:
            duration = mean
        
        # Ensure positive
        duration = F.relu(duration) + 1e-6
        
        return duration


class LengthPredictorLoss(nn.Module):
    """
    Loss function for training length predictor.
    
    Supports:
    - MSE loss (default)
    - Smooth L1 loss
    - Log-space loss (better for wide range of durations)
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        log_space: bool = False,
        relative_weight: float = 0.0,
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.log_space = log_space
        self.relative_weight = relative_weight
        
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss()
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            pred: Predicted duration [B, 1]
            target: Ground truth duration [B, 1]
        
        Returns:
            Scalar loss value
        """
        if self.log_space:
            # Log-space loss (better for wide range)
            pred = torch.log(pred + 1)
            target = torch.log(target + 1)
        
        loss = self.base_loss(pred, target)
        
        # Optional: Add relative error term
        if self.relative_weight > 0:
            relative_error = torch.abs(pred - target) / (target + 1e-6)
            loss = loss + self.relative_weight * relative_error.mean()
        
        return loss


# Utility functions

def compute_duration_statistics(dataloader) -> dict:
    """
    Compute duration statistics from dataset.
    
    Returns:
        Dictionary with mean, std, min, max, percentiles
    """
    durations = []
    
    for batch in dataloader:
        if 'duration' in batch:
            durations.append(batch['duration'])
        elif 'motion' in batch:
            # Infer from motion length
            motion = batch['motion']
            duration = motion.shape[1]  # Assuming [B, T, D]
            durations.append(torch.tensor([duration] * motion.shape[0]))
    
    durations = torch.cat(durations)
    
    return {
        'mean': durations.float().mean().item(),
        'std': durations.float().std().item(),
        'min': durations.min().item(),
        'max': durations.max().item(),
        'p25': torch.quantile(durations.float(), 0.25).item(),
        'p50': torch.quantile(durations.float(), 0.50).item(),
        'p75': torch.quantile(durations.float(), 0.75).item(),
        'p95': torch.quantile(durations.float(), 0.95).item(),
    }


if __name__ == "__main__":
    # Quick test
    model = LengthPredictor(text_dim=512, hidden_dim=256, num_layers=3)
    
    # Dummy input
    text_emb = torch.randn(4, 512)
    
    # Forward pass
    duration = model(text_emb)
    print(f"Input shape: {text_emb.shape}")
    print(f"Output shape: {duration.shape}")
    print(f"Predicted durations: {duration.squeeze().tolist()}")
    
    # Test primitive prediction
    num_prims = model.predict_primitives(text_emb, primitive_frames=8)
    print(f"Predicted primitives: {num_prims.tolist()}")
    
    # Test with uncertainty
    model_unc = LengthPredictorWithUncertainty(text_dim=512)
    mean, std = model_unc(text_emb)
    print(f"Mean duration: {mean.squeeze().tolist()}")
    print(f"Uncertainty (std): {std.squeeze().tolist()}")
