"""
Simplified Body Part Attention

A minimal implementation that only groups attention heads by body parts,
without adding learnable embeddings or complex masking.

Key differences from complex version:
1. NO body part embeddings - avoids disrupting learned attention patterns
2. Simple head grouping - different head groups naturally focus on different features
3. Optional feature reordering - group features by body parts for locality

Author: TextOp Team
Date: 2026-01-02
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimpleBodyPartAttention(nn.Module):
    """
    Simplified body-part-aware multi-head attention.
    
    This is a minimal implementation that only groups attention heads,
    without adding extra parameters or complex mechanisms.
    
    Args:
        d_model: Model dimension
        nhead: Total number of attention heads (must be divisible by num_body_parts)
        num_body_parts: Number of body parts (default: 4)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_body_parts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_body_parts = num_body_parts
        
        assert nhead % num_body_parts == 0, \
            f"nhead ({nhead}) must be divisible by num_body_parts ({num_body_parts})"
        
        self.heads_per_part = nhead // num_body_parts
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        # Standard attention projections (no extra parameters)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Standard multi-head attention forward pass.
        The only difference is the conceptual grouping of heads.
        
        Args:
            query: [seq_len, batch_size, d_model]
            key: [seq_len, batch_size, d_model]
            value: [seq_len, batch_size, d_model]
            
        Returns:
            output: [seq_len, batch_size, d_model]
        """
        seq_len, batch_size, _ = query.shape
        
        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        # [L, B, d_model] -> [L, B, nhead, head_dim] -> [B, nhead, L, head_dim]
        Q = Q.view(seq_len, batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        K = K.view(seq_len, batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        V = V.view(seq_len, batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        
        # Compute attention scores
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + attn_mask
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len, batch_size, self.d_model)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output, None


class SimpleBodyPartTransformerEncoderLayer(nn.Module):
    """
    Simplified transformer encoder layer with body-part-aware attention.
    Drop-in replacement for standard TransformerEncoderLayer.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        num_body_parts: int = 4,
    ):
        super().__init__()
        
        self.normalize_before = normalize_before
        
        # Use simplified body part attention
        self.self_attn = SimpleBodyPartAttention(
            d_model=d_model,
            nhead=nhead,
            num_body_parts=num_body_parts,
            dropout=dropout,
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass"""
        # Add positional encoding if provided
        q = k = src if pos is None else src + pos
        
        # Self-attention block
        if self.normalize_before:
            src2, _ = self.self_attn(
                self.norm1(q), self.norm1(k), self.norm1(src),
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + self.dropout1(src2)
        else:
            src2, _ = self.self_attn(
                q, k, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = self.norm1(src + self.dropout1(src2))
        
        # Feedforward block
        if self.normalize_before:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(src2))
        
        return src


class SimpleBodyPartSkipTransformerEncoder(nn.Module):
    """
    Simplified skip-connection transformer encoder with body part attention.
    Compatible with existing SkipTransformerEncoder API.
    """
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([
            type(encoder_layer)(
                d_model=encoder_layer.self_attn.d_model,
                nhead=encoder_layer.self_attn.nhead,
                dim_feedforward=encoder_layer.linear1.out_features,
                dropout=encoder_layer.dropout.p,
                activation="gelu" if hasattr(encoder_layer, 'activation') else "relu",
                normalize_before=encoder_layer.normalize_before,
                num_body_parts=encoder_layer.self_attn.num_body_parts,
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward with skip connections every 3 layers (U-Net style)
        """
        output = src
        intermediates = []
        
        for i, layer in enumerate(self.layers):
            # Store for skip connection
            if i % 3 == 0:
                intermediates.append(output)
            
            # Apply layer
            output = layer(output, src_mask=mask, 
                         src_key_padding_mask=src_key_padding_mask, pos=pos)
            
            # Add skip connection from 3 layers ago
            if i >= 3 and i % 3 == 0:
                skip_idx = (i // 3) - 1
                if skip_idx < len(intermediates):
                    output = output + intermediates[skip_idx]
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output
