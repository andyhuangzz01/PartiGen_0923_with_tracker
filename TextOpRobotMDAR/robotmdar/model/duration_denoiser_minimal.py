"""
Duration-Conditioned Denoiser (Minimal Version)

This is a minimal modification of the original DenoiserTransformer that adds
duration conditioning. It's designed to:
1. Be compatible with pretrained DAR weights (can load and fine-tune)
2. Add minimal new parameters (only duration embedding)
3. Not break existing functionality

The key addition is a duration embedding that gets added to the transformer sequence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationConditionedDenoiserTransformerMinimal(nn.Module):
    """
    Minimal duration-conditioned denoiser based on DenoiserTransformer.
    
    Changes from original:
    1. Added DurationEmbedder
    2. Added null_duration_emb for CFG
    3. Modified forward() to accept duration in y dict
    4. Sequence is now: [time, text, duration, history..., noise]
    
    Can load pretrained weights from original DenoiserTransformer.
    """
    
    def __init__(self,
                 h_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 clip_dim: int = 512,
                 history_shape: tuple = (2, 57),
                 noise_shape: tuple = (1, 128),
                 max_frames: int = 300,
                 fps: float = 30.0,
                 duration_cond_mask_prob: float = 0.1,
                 use_vae: bool = True,
                 **kargs):
        super().__init__()
        self.h_dim = h_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.history_shape = history_shape
        self.noise_shape = noise_shape
        self.clip_dim = clip_dim
        self.max_frames = max_frames
        self.fps = fps

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.duration_cond_mask_prob = duration_cond_mask_prob
        
        print(f'[DurationConditionedDenoiserMinimal] h_dim={h_dim}, num_layers={num_layers}')
        print(f'  cond_mask_prob={self.cond_mask_prob}, duration_mask_prob={duration_cond_mask_prob}')

        # === Original DenoiserTransformer components ===
        self.sequence_pos_encoder = PositionalEncoding(self.h_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.h_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(self.clip_dim, self.h_dim)
        self.embed_history = nn.Linear(self.history_shape[-1], self.h_dim)
        self.embed_noise = nn.Linear(self.noise_shape[-1], self.h_dim)

        print("TRANS_ENC init (Duration-Conditioned Minimal)")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.h_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers)

        self.output_process = nn.Linear(self.h_dim, self.noise_shape[-1])

        # === NEW: Duration conditioning components ===
        self.embed_duration = DurationEmbedder(self.h_dim, max_frames, fps)
        self.null_duration_emb = nn.Parameter(torch.zeros(1, self.h_dim))
        nn.init.normal_(self.null_duration_emb, std=0.02)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond: torch.Tensor, force_mask: bool = False) -> torch.Tensor:
        """Mask text condition for classifier-free guidance."""
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond
    
    def mask_duration(self, duration_emb: torch.Tensor, force_mask: bool = False) -> torch.Tensor:
        """Mask duration condition for classifier-free guidance."""
        bs = duration_emb.shape[0]
        if force_mask:
            return self.null_duration_emb.expand(bs, -1)
        elif self.training and self.duration_cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=duration_emb.device) * self.duration_cond_mask_prob).view(bs, 1)
            return duration_emb * (1. - mask) + self.null_duration_emb * mask
        else:
            return duration_emb

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, y: dict = None) -> torch.Tensor:
        """
        Forward pass with optional duration conditioning.
        
        Args:
            x_t: [B, T=1, D] noisy latent
            timesteps: [B] diffusion timesteps
            y: condition dict containing:
                - 'history_motion_normalized': [B, History, nfeats]
                - 'text_embedding': [B, clip_dim]
                - 'duration': [B] target duration in frames (optional)
                - 'duration_type': str, 'frames'/'seconds'/'normalized' (default: 'frames')
                - 'uncond': bool, force unconditional generation
                - 'uncond_duration': bool, force unconditional duration
        """
        # Timestep embedding
        emb_time = self.embed_timestep(timesteps)  # [1, bs, d]
        
        # History embedding
        emb_history = self.embed_history(y['history_motion_normalized']).permute(1, 0, 2)  # [H, bs, d]
        
        # Text embedding with masking
        force_mask = y.get('uncond', False)
        emb_text = self.embed_text(self.mask_cond(y['text_embedding'], force_mask=force_mask)).unsqueeze(0)  # [1, bs, d]
        
        # Noise embedding
        emb_noise = self.embed_noise(x_t).permute(1, 0, 2)  # [1, bs, d]
        
        # Duration embedding (NEW)
        batch_size = x_t.shape[0]
        if 'duration' in y and y['duration'] is not None:
            duration_type = y.get('duration_type', 'frames')
            emb_duration = self.embed_duration(y['duration'], input_type=duration_type)
        else:
            # No duration provided - use null embedding
            emb_duration = self.null_duration_emb.expand(batch_size, -1)
        
        force_mask_duration = y.get('uncond_duration', False) or force_mask
        emb_duration = self.mask_duration(emb_duration, force_mask=force_mask_duration)
        emb_duration = emb_duration.unsqueeze(0)  # [1, bs, d]
        
        # Concatenate: [time, text, duration, history..., noise]
        xseq = torch.cat((emb_time, emb_text, emb_duration, emb_history, emb_noise), dim=0)
        
        # Transformer
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq)[-self.noise_shape[0]:]  # [1, bs, h_dim]
        
        # Project to output
        output = self.output_process(output)  # [1, B, noise_shape[-1]]
        output = output.permute(1, 0, 2)  # [B, 1, noise_shape[-1]]
        
        return output
    
    def load_pretrained_denoiser(self, state_dict: dict, strict: bool = False):
        """
        Load weights from a pretrained DenoiserTransformer.
        
        Duration-specific layers (embed_duration, null_duration_emb) will be
        randomly initialized and need fine-tuning.
        """
        # Keys that are new in this model
        new_keys = ['embed_duration', 'null_duration_emb']
        
        # Filter out new keys from expected
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
        new_missing = [k for k in missing if any(nk in k for nk in new_keys)]
        other_missing = [k for k in missing if not any(nk in k for nk in new_keys)]
        
        print(f"[load_pretrained_denoiser] Loaded weights from pretrained model")
        print(f"  New layers (will be trained): {len(new_missing)} keys")
        if other_missing:
            print(f"  WARNING: Other missing keys: {other_missing}")
        if unexpected:
            print(f"  WARNING: Unexpected keys: {unexpected}")
        
        return missing, unexpected


class DurationEmbedder(nn.Module):
    """Embed duration into hidden space using sinusoidal encoding."""
    
    def __init__(self, h_dim: int, max_frames: int = 300, fps: float = 30.0):
        super().__init__()
        self.h_dim = h_dim
        self.max_frames = max_frames
        self.fps = fps
        
        self.duration_encoder = SinusoidalPositionEmbeddings(h_dim)
        self.duration_embed = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
    
    def forward(self, duration: torch.Tensor, input_type: str = 'frames') -> torch.Tensor:
        if input_type == 'frames':
            normalized = duration.float() / self.max_frames
        elif input_type == 'seconds':
            normalized = (duration * self.fps) / self.max_frames
        elif input_type == 'normalized':
            normalized = duration
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        scaled = normalized * 1000
        emb = self.duration_encoder(scaled)
        return self.duration_embed(emb)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for continuous values."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """Embed diffusion timesteps."""
    
    def __init__(self, h_dim: int, sequence_pos_encoder: PositionalEncoding):
        super().__init__()
        self.h_dim = h_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        self.time_embed = nn.Sequential(
            nn.Linear(self.h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
