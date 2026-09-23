"""
Adaptive Length Denoiser - DAR with Integrated Length Prediction

This denoiser automatically predicts the motion length from text embeddings,
eliminating the need for external duration input.

Two modes:
1. Integrated Length Predictor: Uses a built-in MLP to predict length
2. EOS Token: Predicts when to stop generating (future work)

Author: TextOp Team
Date: 2026-01-05
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LengthPredictorHead(nn.Module):
    """
    Internal length predictor head.
    Can be initialized from pretrained Length Predictor weights.
    """
    def __init__(self, input_dim=512, hidden_dim=256, max_length=300):
        super().__init__()
        self.max_length = max_length
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, text_emb):
        """
        Args:
            text_emb: [B, 512] text embeddings
        Returns:
            predicted_length: [B] integer frame numbers
        """
        # Predict continuous value
        length_raw = self.net(text_emb).squeeze(-1)  # [B]
        
        # Clamp and round to integer
        length_clamped = torch.clamp(length_raw, min=10, max=self.max_length)
        length_int = length_clamped.round().long()
        
        return length_int, length_raw  # Return both for potential loss computation


class DurationEmbedder(nn.Module):
    """Duration embedding layer (same as duration_denoiser)"""
    def __init__(self, max_duration=300, h_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(max_duration + 1, h_dim)
    
    def forward(self, durations):
        return self.embedding(durations)


class AdaptiveLengthDenoiser(nn.Module):
    """
    Denoiser with integrated length prediction.
    
    Usage:
        # Training: provide ground truth duration
        output = model(x, t, y={'text': text_emb, 'duration': gt_duration})
        
        # Inference: length is predicted automatically
        output = model(x, t, y={'text': text_emb})  # duration auto-predicted
    """
    
    def __init__(self,
                 h_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 clip_dim: int = 512,
                 history_shape: tuple = (2, 57),
                 noise_shape: tuple = (1, 128),
                 max_frames: int = 300,
                 fps: float = 30.0,
                 cond_mask_prob: float = 0.1,
                 use_length_predictor: bool = True,
                 length_predictor_hidden: int = 256,
                 **kwargs):
        super().__init__()
        
        self.h_dim = h_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.clip_dim = clip_dim
        self.history_shape = history_shape
        self.noise_shape = noise_shape
        self.max_frames = max_frames
        self.fps = fps
        self.cond_mask_prob = cond_mask_prob
        self.use_length_predictor = use_length_predictor
        
        # ========== Length Predictor (Optional) ==========
        if use_length_predictor:
            self.length_predictor = LengthPredictorHead(
                input_dim=clip_dim,
                hidden_dim=length_predictor_hidden,
                max_length=max_frames
            )
        else:
            self.length_predictor = None
        
        # ========== Duration Embedder ==========
        self.duration_embedder = DurationEmbedder(max_frames, h_dim)
        
        # ========== Input Embedders ==========
        # Time step embedding
        self.time_embedder = nn.Sequential(
            nn.Linear(1, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
        
        # Text embedding projection
        self.text_embedder = nn.Linear(clip_dim, h_dim)
        
        # History motion embedding
        history_dim = history_shape[0] * history_shape[1]
        self.history_embedder = nn.Linear(history_dim, h_dim)
        
        # Noise embedding
        noise_dim = noise_shape[0] * noise_shape[1]
        self.noise_embedder = nn.Linear(noise_dim, h_dim)
        
        # ========== Transformer ==========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # ========== Output Head ==========
        self.output_head = nn.Linear(h_dim, noise_dim)
        
        # ========== Null Embeddings for Classifier-Free Guidance ==========
        self.null_text_emb = nn.Parameter(torch.randn(1, h_dim))
        self.null_duration_emb = nn.Parameter(torch.randn(1, h_dim))
        
        print(f"[AdaptiveLengthDenoiser] Initialized with:")
        print(f"  - h_dim={h_dim}, layers={num_layers}, heads={num_heads}")
        print(f"  - max_frames={max_frames}, fps={fps}")
        print(f"  - use_length_predictor={use_length_predictor}")
    
    def load_length_predictor_weights(self, pretrained_path):
        """
        Load pretrained Length Predictor weights.
        
        Args:
            pretrained_path: Path to saved Length Predictor checkpoint
        """
        if not self.use_length_predictor:
            print("[Warning] Length predictor is disabled, skipping weight loading")
            return
        
        print(f"[AdaptiveLengthDenoiser] Loading length predictor from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location='cpu')
        
        # Load state dict
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        
        # Map keys (pretrained keys might have different prefix)
        mapped_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'net.' prefix if exists
            new_k = k.replace('net.', '')
            mapped_state_dict[f'net.{new_k}'] = v
        
        self.length_predictor.load_state_dict(mapped_state_dict, strict=True)
        print(f"[AdaptiveLengthDenoiser] Length predictor loaded successfully")
        
        # Optionally freeze length predictor
        # for param in self.length_predictor.parameters():
        #     param.requires_grad = False
    
    def predict_length(self, text_emb):
        """
        Predict motion length from text embedding.
        
        Args:
            text_emb: [B, clip_dim]
        Returns:
            predicted_length: [B] integer frames
        """
        if not self.use_length_predictor:
            # Fallback to default length
            batch_size = text_emb.shape[0]
            return torch.full((batch_size,), 120, dtype=torch.long, device=text_emb.device)
        
        with torch.no_grad():
            length_int, _ = self.length_predictor(text_emb)
        return length_int
    
    def forward(self, x, timesteps, y):
        """
        Forward pass with automatic length prediction.
        
        Args:
            x: [B, 1, latent_dim] noisy latent
            timesteps: [B] diffusion timesteps
            y: dict with keys:
                - 'text': [B, clip_dim] text embeddings
                - 'history': [B, 2, nfeats] motion history
                - 'duration': [B] (optional) ground truth duration
                - 'uncond': bool (optional) for classifier-free guidance
        
        Returns:
            predicted_noise: [B, 1, latent_dim]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Extract inputs
        text_emb = y['text']  # [B, 512]
        history = y['history']  # [B, 2, nfeats]
        uncond = y.get('uncond', False)
        
        # ========== Length Prediction ==========
        if 'duration' in y and y['duration'] is not None:
            # Training mode: use ground truth duration
            duration = y['duration']  # [B]
        else:
            # Inference mode: predict duration
            if self.use_length_predictor:
                duration, _ = self.length_predictor(text_emb)  # [B]
            else:
                # Fallback
                duration = torch.full((batch_size,), 120, dtype=torch.long, device=device)
        
        # ========== Embed All Inputs ==========
        # Time embedding
        t_norm = timesteps.float().unsqueeze(-1) / 1000.0  # Normalize to [0, 1]
        time_emb = self.time_embedder(t_norm)  # [B, h_dim]
        
        # Text embedding (with CFG)
        if uncond or (self.training and torch.rand(1).item() < self.cond_mask_prob):
            text_emb_h = self.null_text_emb.expand(batch_size, -1)
        else:
            text_emb_h = self.text_embedder(text_emb)  # [B, h_dim]
        
        # Duration embedding (with CFG)
        if uncond:
            duration_emb = self.null_duration_emb.expand(batch_size, -1)
        else:
            duration_emb = self.duration_embedder(duration)  # [B, h_dim]
        
        # History embedding
        history_flat = history.reshape(batch_size, -1)
        history_emb = self.history_embedder(history_flat)  # [B, h_dim]
        
        # Noise embedding
        x_flat = x.reshape(batch_size, -1)
        noise_emb = self.noise_embedder(x_flat)  # [B, h_dim]
        
        # ========== Build Transformer Sequence ==========
        # Sequence: [time, text, duration, history, noise]
        sequence = torch.stack([
            time_emb,
            text_emb_h,
            duration_emb,
            history_emb,
            noise_emb
        ], dim=1)  # [B, 5, h_dim]
        
        # ========== Transformer Forward ==========
        transformer_out = self.transformer(sequence)  # [B, 5, h_dim]
        
        # Take noise token output
        noise_out = transformer_out[:, -1, :]  # [B, h_dim]
        
        # ========== Output Projection ==========
        predicted_noise = self.output_head(noise_out)  # [B, noise_dim]
        predicted_noise = predicted_noise.reshape(x.shape)  # [B, 1, latent_dim]
        
        return predicted_noise
    
    @property
    def device(self):
        return next(self.parameters()).device


class AdaptiveLengthDARWithLengthLoss(AdaptiveLengthDenoiser):
    """
    Extended version that also trains the length predictor jointly.
    
    Usage:
        model = AdaptiveLengthDARWithLengthLoss(...)
        
        # Forward returns both noise prediction and length prediction
        noise_pred, length_pred = model(x, t, y)
        
        # Compute losses
        noise_loss = mse_loss(noise_pred, target_noise)
        length_loss = mse_loss(length_pred, y['duration'].float())
        total_loss = noise_loss + 0.1 * length_loss
    """
    
    def __init__(self, length_loss_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.length_loss_weight = length_loss_weight
    
    def forward(self, x, timesteps, y, return_length_pred=False):
        """
        Extended forward that optionally returns length prediction.
        
        Args:
            return_length_pred: If True, also return length prediction for loss
        
        Returns:
            predicted_noise: [B, 1, latent_dim]
            length_pred: [B] (only if return_length_pred=True)
        """
        # Get length prediction if needed
        if return_length_pred and self.use_length_predictor:
            text_emb = y['text']
            length_int, length_raw = self.length_predictor(text_emb)
            
            # Temporarily store predicted length for use in forward
            y['duration'] = length_int
        
        # Regular forward
        predicted_noise = super().forward(x, timesteps, y)
        
        if return_length_pred and self.use_length_predictor:
            return predicted_noise, length_raw
        else:
            return predicted_noise


if __name__ == "__main__":
    # Test the model
    print("=" * 80)
    print("Testing AdaptiveLengthDenoiser")
    print("=" * 80)
    
    # Create model
    model = AdaptiveLengthDenoiser(
        h_dim=512,
        num_layers=8,
        use_length_predictor=True,
    )
    
    # Test inputs
    batch_size = 4
    x = torch.randn(batch_size, 1, 128)
    timesteps = torch.randint(0, 1000, (batch_size,))
    y = {
        'text': torch.randn(batch_size, 512),
        'history': torch.randn(batch_size, 2, 57),
    }
    
    # Test forward (inference mode - auto predict length)
    print("\n[Test 1] Inference mode (auto length prediction)")
    output = model(x, timesteps, y)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test forward (training mode - with GT length)
    print("\n[Test 2] Training mode (with GT duration)")
    y['duration'] = torch.randint(60, 180, (batch_size,))
    output = model(x, timesteps, y)
    print(f"GT duration: {y['duration']}")
    print(f"Output shape: {output.shape}")
    
    # Test length prediction
    print("\n[Test 3] Length prediction")
    predicted_lengths = model.predict_length(y['text'])
    print(f"Predicted lengths: {predicted_lengths}")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
