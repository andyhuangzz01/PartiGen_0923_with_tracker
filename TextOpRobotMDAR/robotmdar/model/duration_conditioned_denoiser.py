"""
Duration-Conditioned Denoiser for Variable-Length Motion Generation

This module implements Scheme 4 from duration_predict.md:
- Duration Condition: Add duration/length as an additional condition to the denoiser
- The model learns to generate motion tokens appropriate for the target duration
- At inference time, duration can come from:
  1. User specification (explicit control)
  2. Length Predictor (automatic prediction)
  3. LLM planning (semantic understanding)

Key Design:
- Duration embedding is added as a separate token in the transformer sequence
- Compatible with both training (ground truth duration) and inference (predicted duration)
- Supports both frame count and normalized duration inputs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationEmbedder(nn.Module):
    """
    Embed duration/length information into a learnable representation.
    
    Supports multiple input formats:
    - Frame count (integer): e.g., 120 frames
    - Normalized duration (float 0-1): e.g., 0.5 for medium length
    - Seconds (float): e.g., 4.0 seconds
    
    Uses sinusoidal encoding similar to timestep embedding for smooth interpolation.
    """
    
    def __init__(self, h_dim: int, max_frames: int = 300, fps: float = 30.0):
        super().__init__()
        self.h_dim = h_dim
        self.max_frames = max_frames
        self.fps = fps
        
        # Sinusoidal encoding for duration
        self.duration_encoder = SinusoidalPositionEmbeddings(h_dim)
        
        # Project to hidden dimension
        self.duration_embed = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
    
    def forward(self, duration: torch.Tensor, input_type: str = 'frames') -> torch.Tensor:
        """
        Args:
            duration: [B] tensor of duration values
            input_type: 'frames', 'normalized', or 'seconds'
            
        Returns:
            [B, h_dim] duration embedding
        """
        # Normalize to [0, 1] range based on max_frames
        if input_type == 'frames':
            normalized = duration.float() / self.max_frames
        elif input_type == 'seconds':
            normalized = (duration * self.fps) / self.max_frames
        elif input_type == 'normalized':
            normalized = duration
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        # Scale to reasonable range for sinusoidal encoding (0-1000)
        scaled = normalized * 1000
        
        # Get sinusoidal encoding
        emb = self.duration_encoder(scaled)
        
        # Project through MLP
        return self.duration_embed(emb)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for continuous values (like timesteps or durations)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B] tensor of values
            
        Returns:
            [B, dim] sinusoidal embeddings
        """
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DurationConditionedDenoiserTransformer(nn.Module):
    """
    Transformer-based denoiser with duration conditioning.
    
    This extends the base DenoiserTransformer by adding:
    1. Duration embedding as an additional condition token
    2. Optional duration dropout for classifier-free guidance
    
    The forward pass sequence becomes:
    [time_emb, text_emb, duration_emb, history_emb, noise_emb]
    
    During training:
    - Use ground truth duration (number of frames in the motion)
    - Optional duration masking for unconditional generation capability
    
    During inference:
    - Use predicted duration from LengthPredictor
    - Or user-specified duration for explicit control
    """
    
    def __init__(self,
                 h_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 clip_dim: int = 512,
                 history_shape: tuple = (2, 276),
                 noise_shape: tuple = (1, 128),
                 max_frames: int = 300,
                 fps: float = 30.0,
                 duration_cond_mask_prob: float = 0.0,
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

        # Probability of masking conditions for classifier-free guidance
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.duration_cond_mask_prob = duration_cond_mask_prob
        
        print(f'[DurationConditionedDenoiser] cond_mask_prob: {self.cond_mask_prob}, '
              f'duration_cond_mask_prob: {self.duration_cond_mask_prob}')

        # Positional encoding
        self.sequence_pos_encoder = PositionalEncoding(self.h_dim, self.dropout)
        
        # Input embeddings
        self.embed_timestep = TimestepEmbedder(self.h_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(self.clip_dim, self.h_dim)
        self.embed_history = nn.Linear(self.history_shape[-1], self.h_dim)
        self.embed_noise = nn.Linear(self.noise_shape[-1], self.h_dim)
        
        # NEW: Duration embedding
        self.embed_duration = DurationEmbedder(self.h_dim, max_frames, fps)
        
        # Null duration embedding for masking (learnable)
        self.null_duration_emb = nn.Parameter(torch.zeros(1, self.h_dim))
        nn.init.normal_(self.null_duration_emb, std=0.02)

        # Transformer encoder layers
        print("TRANS_ENC init (Duration-Conditioned)")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.h_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers)

        # Output projection
        self.output_process = nn.Linear(self.h_dim, self.noise_shape[-1])

    def parameters_wo_clip(self):
        return [
            p for name, p in self.named_parameters()
            if not name.startswith('clip_model.')
        ]

    def mask_cond(self, cond: torch.Tensor, force_mask: bool = False) -> torch.Tensor:
        """Mask text condition for classifier-free guidance."""
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond
    
    def mask_duration(self, duration_emb: torch.Tensor, force_mask: bool = False) -> torch.Tensor:
        """Mask duration condition for classifier-free guidance."""
        bs = duration_emb.shape[0]
        if force_mask:
            return self.null_duration_emb.expand(bs, -1)
        elif self.training and self.duration_cond_mask_prob > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=duration_emb.device) * self.duration_cond_mask_prob
            ).view(bs, 1)
            return duration_emb * (1. - mask) + self.null_duration_emb * mask
        else:
            return duration_emb

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, y: dict = None) -> torch.Tensor:
        """
        Forward pass with duration conditioning.
        
        Args:
            x_t: [B, T=1, D] noisy latent
            timesteps: [B] diffusion timesteps
            y: condition dict containing:
                - 'history_motion_normalized': [B, History, nfeats]
                - 'text_embedding': [B, clip_dim]
                - 'duration': [B] target duration in frames (optional)
                - 'duration_type': str, one of 'frames', 'seconds', 'normalized' (default: 'frames')
                - 'uncond': bool, force unconditional generation
                - 'uncond_duration': bool, force unconditional duration
                
        Returns:
            [B, 1, noise_dim] predicted noise
        """
        # Timestep embedding
        emb_time = self.embed_timestep(timesteps)  # [1, bs, d]
        
        # History embedding
        emb_history = self.embed_history(
            y['history_motion_normalized']
        ).permute(1, 0, 2)  # [History, bs, d]
        
        # Text embedding with masking
        force_mask = y.get('uncond', False)
        emb_text = self.embed_text(
            self.mask_cond(y['text_embedding'], force_mask=force_mask)
        ).unsqueeze(0)  # [1, bs, d]
        
        # Duration embedding with masking
        if 'duration' in y:
            duration_type = y.get('duration_type', 'frames')
            emb_duration = self.embed_duration(y['duration'], input_type=duration_type)
        else:
            # If no duration provided, use null embedding
            batch_size = x_t.shape[0]
            emb_duration = self.null_duration_emb.expand(batch_size, -1)
        
        force_mask_duration = y.get('uncond_duration', False) or force_mask
        emb_duration = self.mask_duration(emb_duration, force_mask=force_mask_duration)
        emb_duration = emb_duration.unsqueeze(0)  # [1, bs, d]
        
        # Noise embedding
        emb_noise = self.embed_noise(x_t).permute(1, 0, 2)  # [1, bs, d]
        
        # Concatenate all embeddings
        # Sequence: [time, text, duration, history..., noise]
        xseq = torch.cat((emb_time, emb_text, emb_duration, emb_history, emb_noise), dim=0)
        
        # Add positional encoding and run transformer
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq)[-self.noise_shape[0]:]  # [1, bs, h_dim]
        
        # Project to output
        output = self.output_process(output)  # [1, B, noise_shape[-1]]
        output = output.permute(1, 0, 2)  # [B, 1, noise_shape[-1]]
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
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

        time_embed_dim = self.h_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.h_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_embed(
            self.sequence_pos_encoder.pe[timesteps]
        ).permute(1, 0, 2)


# =============================================================================
# Utility functions for training and inference
# =============================================================================

def create_duration_conditioned_denoiser(base_config: dict, **override_config) -> DurationConditionedDenoiserTransformer:
    """
    Create a duration-conditioned denoiser from a base config.
    
    Args:
        base_config: Base denoiser configuration
        override_config: Additional config for duration conditioning
        
    Returns:
        DurationConditionedDenoiserTransformer instance
    """
    config = {**base_config, **override_config}
    return DurationConditionedDenoiserTransformer(**config)


def load_pretrained_weights_into_duration_denoiser(
    duration_denoiser: DurationConditionedDenoiserTransformer,
    pretrained_state_dict: dict,
    strict: bool = False
) -> tuple:
    """
    Load weights from a pretrained base denoiser into duration-conditioned denoiser.
    
    The duration-specific layers (embed_duration, null_duration_emb) will be initialized
    randomly and need to be fine-tuned.
    
    Args:
        duration_denoiser: Target duration-conditioned denoiser
        pretrained_state_dict: State dict from pretrained base denoiser
        strict: Whether to require exact key matching
        
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    # Filter out duration-related keys from target
    duration_keys = ['embed_duration', 'null_duration_emb']
    
    # Load compatible weights
    missing_keys, unexpected_keys = duration_denoiser.load_state_dict(
        pretrained_state_dict, strict=False
    )
    
    # Report what was loaded
    duration_missing = [k for k in missing_keys if any(dk in k for dk in duration_keys)]
    other_missing = [k for k in missing_keys if not any(dk in k for dk in duration_keys)]
    
    print(f"[load_pretrained_weights] Loaded {len(pretrained_state_dict) - len(unexpected_keys)} weights")
    print(f"[load_pretrained_weights] Duration-specific layers (need training): {duration_missing}")
    if other_missing:
        print(f"[load_pretrained_weights] WARNING: Other missing keys: {other_missing}")
    if unexpected_keys:
        print(f"[load_pretrained_weights] WARNING: Unexpected keys: {unexpected_keys}")
    
    return missing_keys, unexpected_keys


# =============================================================================
# Integration with DART sampling pipeline
# =============================================================================

class DurationConditionedDARTSampler:
    """
    Wrapper for DART sampling with duration conditioning.
    
    Usage:
        sampler = DurationConditionedDARTSampler(denoiser, scheduler, vae, length_predictor)
        
        # Auto-predict duration
        motion = sampler.sample(text_embedding, history_motion, predict_duration=True)
        
        # User-specified duration
        motion = sampler.sample(text_embedding, history_motion, target_duration=120)
    """
    
    def __init__(self, 
                 denoiser: DurationConditionedDenoiserTransformer,
                 scheduler,  # DDPMScheduler or similar
                 vae,  # VAE for latent decoding
                 length_predictor=None):  # Optional LengthPredictor
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.vae = vae
        self.length_predictor = length_predictor
    
    def predict_duration(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """Predict duration from text embedding using LengthPredictor."""
        if self.length_predictor is None:
            raise ValueError("LengthPredictor not provided. Use target_duration instead.")
        
        with torch.no_grad():
            return self.length_predictor(text_embedding)
    
    @torch.no_grad()
    def sample(self,
               text_embedding: torch.Tensor,
               history_motion: torch.Tensor,
               target_duration: torch.Tensor = None,
               predict_duration: bool = False,
               guidance_scale: float = 7.5,
               num_inference_steps: int = 50) -> torch.Tensor:
        """
        Sample motion with duration conditioning.
        
        Args:
            text_embedding: [B, clip_dim] CLIP text embedding
            history_motion: [B, history_len, nfeats] history motion
            target_duration: [B] target duration in frames (optional)
            predict_duration: Whether to predict duration automatically
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of diffusion steps
            
        Returns:
            [B, target_duration, nfeats] generated motion
        """
        batch_size = text_embedding.shape[0]
        device = text_embedding.device
        
        # Determine duration
        if target_duration is not None:
            duration = target_duration
        elif predict_duration:
            duration = self.predict_duration(text_embedding)
            duration = duration.round().long()  # Convert to integer frames
        else:
            raise ValueError("Either target_duration or predict_duration must be specified")
        
        # Initialize noise
        latent_shape = (batch_size, *self.denoiser.noise_shape)
        latents = torch.randn(latent_shape, device=device)
        
        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Prepare condition dict
        y = {
            'text_embedding': text_embedding,
            'history_motion_normalized': history_motion,
            'duration': duration,
            'duration_type': 'frames'
        }
        
        # Denoising loop with classifier-free guidance
        for t in self.scheduler.timesteps:
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            if guidance_scale > 1.0:
                # Unconditional prediction
                y_uncond = {**y, 'uncond': True}
                noise_uncond = self.denoiser(latents, timesteps, y_uncond)
                
                # Conditional prediction
                noise_cond = self.denoiser(latents, timesteps, y)
                
                # Classifier-free guidance
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.denoiser(latents, timesteps, y)
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to motion
        # Note: This is a simplified version. Actual decoding depends on your VAE architecture
        motion = self.vae.decode(latents)
        
        return motion, duration
