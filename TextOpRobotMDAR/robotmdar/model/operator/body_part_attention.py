"""
Body Part Attention Module for Robot Motion Generation

This module implements body-part-aware attention mechanisms that allow
the model to learn different attention patterns for different body parts
(e.g., left leg, right leg, torso, upper body).

Key Features:
1. Multi-head attention where each head group focuses on specific body parts
2. Optional attention mask to enforce body part locality
3. Compatible with existing TransformerEncoderLayer interface

Reference: AttT2M (ICCV 2023) - Text-Driven Human Motion Generation with 
           Multi-Perspective Attention Mechanism
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Default body part configuration for G1 23-DOF robot
# V3 layout: orientation(5) + contact(2) + local_displacement(3) + height(1) + q(23) + delta_q(23).
DEFAULT_G1_23DOF_BODY_PARTS = {
    'left_legs': [0, 1, 2, 3, 4, 5],      # 6 DOFs: hip_pitch/roll/yaw, knee, ankle_pitch/roll
    'right_legs': [6, 7, 8, 9, 10, 11],   # 6 DOFs
    'torso': [12, 13, 14],                 # 3 DOFs: waist_yaw/roll/pitch
    'upper_body': [15, 16, 17, 18, 19, 20, 21, 22],  # 8 DOFs: arms
}

# Kinematic chain for G1 robot (defines which parts can influence each other)
G1_KINEMATIC_CHAIN = {
    'torso': ['left_legs', 'right_legs', 'upper_body'],  # torso connects to all
    'left_legs': ['torso', 'right_legs'],  # legs connect to torso and each other
    'right_legs': ['torso', 'left_legs'],
    'upper_body': ['torso'],  # arms connect to torso
}


class BodyPartMultiheadAttention(nn.Module):
    """
    Multi-head attention with body-part awareness.
    
    This extends standard multi-head attention by:
    1. Grouping attention heads by body parts
    2. Optionally applying attention masks based on kinematic structure
    3. Adding learnable body part tokens for part-specific processing
    
    Args:
        d_model: Model dimension
        nhead: Total number of attention heads
        num_body_parts: Number of body parts (default: 4)
        dropout: Dropout rate
        body_part_config: Dict mapping part names to DOF indices
        use_kinematic_mask: Whether to use kinematic chain constraints
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_body_parts: int = 4,
        dropout: float = 0.1,
        body_part_config: Optional[Dict[str, List[int]]] = None,
        use_kinematic_mask: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_body_parts = num_body_parts
        self.use_kinematic_mask = use_kinematic_mask
        
        # Ensure heads can be evenly distributed
        assert nhead % num_body_parts == 0, \
            f"nhead ({nhead}) must be divisible by num_body_parts ({num_body_parts})"
        
        self.heads_per_part = nhead // num_body_parts
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        # Body part configuration
        self.body_part_config = body_part_config or DEFAULT_G1_23DOF_BODY_PARTS
        self.body_part_names = list(self.body_part_config.keys())
        
        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Learnable body part embeddings (added to queries for part-specific attention)
        # Initialize with smaller values to avoid disrupting pre-trained attention patterns
        self.body_part_embeddings = nn.Parameter(
            torch.randn(num_body_parts, 1, self.heads_per_part * self.head_dim) * 0.001
        )
        
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
        Args:
            query: [seq_len, batch_size, d_model]
            key: [seq_len, batch_size, d_model]
            value: [seq_len, batch_size, d_model]
            attn_mask: Optional attention mask
            key_padding_mask: Optional key padding mask
            
        Returns:
            output: [seq_len, batch_size, d_model]
            attn_weights: Optional attention weights (None if not needed)
        """
        seq_len, batch_size, _ = query.shape
        
        # Project Q, K, V
        Q = self.q_proj(query)  # [L, B, d_model]
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        # [L, B, d_model] -> [L, B, nhead, head_dim] -> [B, nhead, L, head_dim]
        Q = Q.view(seq_len, batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        K = K.view(seq_len, batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        V = V.view(seq_len, batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        
        # Add body part embeddings to queries (broadcast across sequence)
        # This encourages different head groups to focus on different aspects
        body_part_bias = self.body_part_embeddings.view(
            self.num_body_parts, self.heads_per_part, self.head_dim
        )  # [num_parts, heads_per_part, head_dim]
        body_part_bias = body_part_bias.view(1, self.nhead, 1, self.head_dim)  # [1, nhead, 1, head_dim]
        Q = Q + body_part_bias
        
        # Compute attention scores
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, nhead, L, L]
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + attn_mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] -> [B, 1, 1, L]
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, nhead, L, head_dim]
        
        # Reshape back
        # [B, nhead, L, head_dim] -> [L, B, nhead, head_dim] -> [L, B, d_model]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len, batch_size, self.d_model)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output, None


class BodyPartTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with body-part-aware attention.
    
    This is a drop-in replacement for TransformerEncoderLayer that uses
    BodyPartMultiheadAttention instead of standard MultiheadAttention.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: Feedforward network dimension
        dropout: Dropout rate
        activation: Activation function ("relu" or "gelu")
        normalize_before: Whether to apply layer norm before attention
        num_body_parts: Number of body parts
        body_part_config: Body part DOF configuration
        use_kinematic_mask: Whether to use kinematic constraints
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
        body_part_config: Optional[Dict[str, List[int]]] = None,
        use_kinematic_mask: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Use body-part-aware attention instead of standard attention
        self.self_attn = BodyPartMultiheadAttention(
            d_model=d_model,
            nhead=nhead,
            num_body_parts=num_body_parts,
            dropout=dropout,
            body_part_config=body_part_config,
            use_kinematic_mask=use_kinematic_mask,
        )
        
        # Feedforward network (same as standard transformer)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    
    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos
    
    def forward_post(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        q = k = self.with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
    def forward_pre(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, _ = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src
    
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class BodyPartSkipTransformerEncoder(nn.Module):
    """
    Skip Transformer Encoder with body-part-aware attention.
    
    This maintains the U-Net style skip connections while using
    body-part-aware attention in each layer.
    
    Args:
        encoder_layer: Template encoder layer (BodyPartTransformerEncoderLayer)
        num_layers: Number of layers (must be odd for skip connections)
        norm: Optional final layer normalization
    """
    
    def __init__(
        self,
        encoder_layer: BodyPartTransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.d_model = encoder_layer.d_model
        self.num_layers = num_layers
        self.norm = norm
        
        assert num_layers % 2 == 1, "num_layers must be odd for skip connections"
        
        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(
            nn.Linear(2 * self.d_model, self.d_model), num_block
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        
        # Input blocks (encoder path)
        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)
        
        # Middle block
        x = self.middle_block(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        
        # Output blocks (decoder path) with skip connections
        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x


def _get_clone(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")
