"""
Body Part Aware MLD VAE

This module implements a VAE with body-part-aware attention mechanism,
which is a drop-in replacement for AutoMldVae with enhanced attention
for different body parts of the robot.

Key Features:
1. Uses BodyPartSkipTransformerEncoder instead of standard SkipTransformerEncoder
2. Maintains full compatibility with existing training and inference pipelines
3. Configurable body part definitions via YAML

Author: TextOp Team
Date: 2026-01-02
"""

from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution

from robotmdar.model.operator import PositionalEncoding
from robotmdar.model.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from robotmdar.model.operator.body_part_attention import (
    BodyPartTransformerEncoderLayer,
    BodyPartSkipTransformerEncoder,
    DEFAULT_G1_23DOF_BODY_PARTS,
)
from robotmdar.model.operator.simple_body_part_attention import (
    SimpleBodyPartTransformerEncoderLayer,
    SimpleBodyPartSkipTransformerEncoder,
)
from robotmdar.model.operator.position_encoding import build_position_encoding
from robotmdar.model.modules import Patcher1D, UnPatcher1D


class BodyPartAutoMldVae(nn.Module):
    """
    VAE with Body-Part-Aware Attention Mechanism.
    
    This is an enhanced version of AutoMldVae that uses body-part-aware
    attention in the transformer encoder/decoder. The attention mechanism
    groups attention heads by body parts, allowing the model to learn
    different attention patterns for different body parts.
    
    Args:
        nfeats: Input feature dimension (e.g., 57 for 23-DOF robot)
        latent_dim: Latent space dimension [latent_size, latent_channels]
        h_dim: Hidden dimension for transformer
        ff_size: Feedforward network dimension
        num_layers: Number of transformer layers (must be odd)
        num_heads: Number of attention heads
        dropout: Dropout rate
        arch: Architecture type ("all_encoder" or "encoder_decoder")
        normalize_before: Whether to apply layer norm before attention
        activation: Activation function ("relu" or "gelu")
        position_embedding: Position embedding type ("learned" or "sine")
        use_patcher: Whether to use wavelet patching
        patch_size: Patch size for wavelet transform
        patch_method: Patching method ("haar", etc.)
        num_body_parts: Number of body parts for attention grouping
        body_part_config: Dict mapping body part names to DOF indices
        use_body_part_attention: Whether to use body-part-aware attention
        use_kinematic_mask: Whether to apply kinematic chain constraints
    """

    def __init__(
        self,
        nfeats: int,
        latent_dim: list = [1, 256],
        h_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "all_encoder",
        normalize_before: bool = False,
        activation: str = "gelu",
        position_embedding: str = "learned",
        use_patcher: bool = False,
        patch_size: int = 1,
        patch_method: str = "haar",
        # Body part attention specific parameters
        num_body_parts: int = 4,
        body_part_config: Optional[Dict[str, List[int]]] = None,
        use_body_part_attention: bool = True,
        use_simple_attention: bool = False,  # Use simplified version (no embeddings)
        use_kinematic_mask: bool = False,
    ) -> None:

        super().__init__()

        # Basic parameters
        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.h_dim = h_dim
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = False
        self.pe_type = "mld"

        # Patching parameters
        self.use_patcher = use_patcher
        self.patch_size = patch_size
        self.patch_method = patch_method

        if self.use_patcher:
            self.wavelet_transform = Patcher1D(patch_size, patch_method)
            self.inverse_wavelet_transform = UnPatcher1D(patch_size, patch_method)

        # Body part attention parameters
        self.num_body_parts = num_body_parts
        self.body_part_config = body_part_config or DEFAULT_G1_23DOF_BODY_PARTS
        self.use_body_part_attention = use_body_part_attention
        self.use_simple_attention = use_simple_attention
        self.use_kinematic_mask = use_kinematic_mask

        # Position encoding
        self.query_pos_encoder = build_position_encoding(
            self.h_dim, position_embedding=position_embedding
        )
        self.query_pos_decoder = build_position_encoding(
            self.h_dim, position_embedding=position_embedding
        )

        # Build encoder
        if use_body_part_attention:
            if use_simple_attention:
                # Use simplified body-part-aware encoder (no embeddings)
                encoder_layer = SimpleBodyPartTransformerEncoderLayer(
                    d_model=self.h_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    num_body_parts=num_body_parts,
                )
                encoder_norm = nn.LayerNorm(self.h_dim)
                self.encoder = SimpleBodyPartSkipTransformerEncoder(
                    encoder_layer, num_layers, encoder_norm
                )
            else:
                # Use full body-part-aware encoder layer (with embeddings)
                encoder_layer = BodyPartTransformerEncoderLayer(
                    d_model=self.h_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    num_body_parts=num_body_parts,
                    body_part_config=self.body_part_config,
                    use_kinematic_mask=use_kinematic_mask,
                )
                encoder_norm = nn.LayerNorm(self.h_dim)
                self.encoder = BodyPartSkipTransformerEncoder(
                    encoder_layer, num_layers, encoder_norm
                )
        else:
            # Fall back to standard encoder
            encoder_layer = TransformerEncoderLayer(
                self.h_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            encoder_norm = nn.LayerNorm(self.h_dim)
            self.encoder = SkipTransformerEncoder(
                encoder_layer, num_layers, encoder_norm
            )

        self.encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)

        # Build decoder
        if self.arch == "all_encoder":
            if use_body_part_attention:
                if use_simple_attention:
                    # Simplified version
                    decoder_layer = SimpleBodyPartTransformerEncoderLayer(
                        d_model=self.h_dim,
                        nhead=num_heads,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        normalize_before=normalize_before,
                        num_body_parts=num_body_parts,
                    )
                    decoder_norm = nn.LayerNorm(self.h_dim)
                    self.decoder = SimpleBodyPartSkipTransformerEncoder(
                        decoder_layer, num_layers, decoder_norm
                    )
                else:
                    # Full version
                    decoder_layer = BodyPartTransformerEncoderLayer(
                        d_model=self.h_dim,
                        nhead=num_heads,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        normalize_before=normalize_before,
                        num_body_parts=num_body_parts,
                        body_part_config=self.body_part_config,
                        use_kinematic_mask=use_kinematic_mask,
                    )
                    decoder_norm = nn.LayerNorm(self.h_dim)
                    self.decoder = BodyPartSkipTransformerEncoder(
                        decoder_layer, num_layers, decoder_norm
                    )
            else:
                decoder_norm = nn.LayerNorm(self.h_dim)
                self.decoder = SkipTransformerEncoder(
                    encoder_layer, num_layers, decoder_norm
                )
        elif self.arch == "encoder_decoder":
            # Note: encoder_decoder arch uses standard attention in decoder
            # This could be extended to use body-part-aware cross-attention
            decoder_layer = TransformerDecoderLayer(
                self.h_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerDecoder(
                decoder_layer, num_layers, decoder_norm
            )
        else:
            raise ValueError(f"Not support architecture: {arch}")

        self.decoder_latent_proj = nn.Linear(self.latent_dim, self.h_dim)

        # Global motion token for latent space
        self.global_motion_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.h_dim)
        )

        # Input/output projections
        self.skel_embedding = nn.Linear(input_feats, self.h_dim)
        self.final_layer = nn.Linear(self.h_dim, output_feats)

        # Latent space normalization buffers
        self.register_buffer('latent_mean', torch.tensor(0))
        self.register_buffer('latent_std', torch.tensor(1))

    def encode(
        self,
        future_motion: Tensor,
        history_motion: Tensor,
        scale_latent: bool = False,
    ) -> Tuple[Tensor, Distribution]:
        """
        Encode motion sequence to latent space.
        
        Args:
            future_motion: Future motion frames [batch_size, nfuture, nfeats]
            history_motion: History motion frames [batch_size, nhistory, nfeats]
            scale_latent: Whether to scale latent by learned std
            
        Returns:
            latent: Sampled latent vector [latent_size, batch_size, latent_dim]
            dist: Latent distribution (Normal)
        """
        bs, nfuture, nfeats = future_motion.shape
        nhistory = history_motion.shape[1]

        # Concatenate history and future
        x = torch.cat((history_motion, future_motion), dim=1)  # [bs, H+F, nfeats]

        # Apply wavelet transform if enabled
        if self.use_patcher:
            x = self.wavelet_transform(x)

        # Embed to hidden dimension
        x = self.skel_embedding(x)

        # Transpose for transformer: [bs, seq, h_dim] -> [seq, bs, h_dim]
        x = x.permute(1, 0, 2)

        # Prepare global motion tokens for distribution
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # Concatenate tokens with sequence
        xseq = torch.cat((dist, x), 0)

        # Add positional encoding
        xseq = self.query_pos_encoder(xseq)

        # Encode
        dist = self.encoder(xseq)[:dist.shape[0]]  # [2*latent_size, bs, h_dim]
        dist = self.encoder_latent_proj(dist)  # [2*latent_size, bs, latent_dim]

        # Extract distribution parameters
        mu = dist[0:self.latent_size, ...]
        logvar = dist[self.latent_size:, ...]
        logvar = torch.clamp(logvar, min=-10, max=10)  # Numerical stability

        # Reparameterization
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()

        if scale_latent:
            latent = latent / self.latent_std

        return latent, dist

    def decode(
        self,
        z: Tensor,
        history_motion: Tensor,
        nfuture: int,
        scale_latent: bool = False,
    ) -> Tensor:
        """
        Decode latent vector to motion sequence.
        
        Args:
            z: Latent vector [latent_size, batch_size, latent_dim]
            history_motion: History motion frames [batch_size, nhistory, nfeats]
            nfuture: Number of future frames to generate
            scale_latent: Whether to scale latent by learned std
            
        Returns:
            feats: Reconstructed future motion [batch_size, nfuture, nfeats]
        """
        bs = history_motion.shape[0]
        device = next(self.parameters()).device

        # Scale latent if needed
        if scale_latent:
            z = z * self.latent_std

        # Project latent to hidden dimension
        z = self.decoder_latent_proj(z).to(device)

        # Prepare query tokens for future frames
        queries = torch.zeros(nfuture, bs, self.h_dim, device=z.device).to(device)

        # Embed history motion
        history_embedding = self.skel_embedding(history_motion).permute(1, 0, 2).to(device)

        # Decode based on architecture
        if self.arch == "all_encoder":
            xseq = torch.cat((z, history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq)[-nfuture:]

        elif self.arch == "encoder_decoder":
            xseq = torch.cat((history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(tgt=xseq, memory=z)
            output = output[-nfuture:]

        # Final projection to feature space
        output = self.final_layer(output)

        # Transpose back: [seq, bs, nfeats] -> [bs, seq, nfeats]
        feats = output.permute(1, 0, 2)

        # Apply inverse wavelet transform if needed
        if self.use_patcher:
            feats = self.inverse_wavelet_transform(feats)

        return feats

    def forward(self, z: Tensor, history_motion: Tensor, nfuture: int = 8) -> Tensor:
        """
        Forward pass (decode only, for inference).
        
        Args:
            z: Latent vector
            history_motion: History motion frames
            nfuture: Number of future frames
            
        Returns:
            Decoded motion sequence
        """
        return self.decode(z, history_motion, nfuture)

    def get_body_part_info(self) -> Dict:
        """
        Get information about body part configuration.
        
        Returns:
            Dict containing body part configuration and statistics
        """
        return {
            'num_body_parts': self.num_body_parts,
            'body_part_config': self.body_part_config,
            'use_body_part_attention': self.use_body_part_attention,
            'use_kinematic_mask': self.use_kinematic_mask,
            'total_dofs': sum(len(v) for v in self.body_part_config.values()),
        }
