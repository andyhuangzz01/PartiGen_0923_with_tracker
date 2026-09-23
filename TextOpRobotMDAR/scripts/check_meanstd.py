"""
Check meanstd.pkl statistics to diagnose normalization issues
"""
import torch
import numpy as np
from pathlib import Path

dataset_path = Path('./dataset/HumanML3D-G1-23DOF-30fps')
meanstd_path = dataset_path / 'meanstd.pkl'

if meanstd_path.exists():
    print(f"Loading meanstd from: {meanstd_path}")
    mean, std = torch.load(meanstd_path)
    
    print("\n" + "="*60)
    print("MEAN STATISTICS")
    print("="*60)
    print(f"Shape: {mean.shape}")
    print(f"Min: {mean.min().item():.6f}")
    print(f"Max: {mean.max().item():.6f}")
    print(f"Mean of means: {mean.mean().item():.6f}")
    print(f"\nFirst 10 values: {mean[:10].tolist()}")
    
    print("\n" + "="*60)
    print("STD STATISTICS")
    print("="*60)
    print(f"Shape: {std.shape}")
    print(f"Min: {std.min().item():.6f}")
    print(f"Max: {std.max().item():.6f}")
    print(f"Mean of stds: {std.mean().item():.6f}")
    print(f"\nFirst 10 values: {std[:10].tolist()}")
    
    # Check for problematic values
    print("\n" + "="*60)
    print("DIAGNOSTICS")
    print("="*60)
    
    # Check for zeros or very small std
    small_std_mask = std < 1e-4
    if small_std_mask.any():
        print(f"⚠ WARNING: {small_std_mask.sum()} dimensions have std < 1e-4")
        print(f"  Indices: {torch.where(small_std_mask)[0].tolist()}")
    
    # Check for very large std
    large_std_mask = std > 100
    if large_std_mask.any():
        print(f"⚠ WARNING: {large_std_mask.sum()} dimensions have std > 100")
        print(f"  Indices: {torch.where(large_std_mask)[0].tolist()}")
    
    # Check for NaN or Inf
    if torch.isnan(mean).any() or torch.isinf(mean).any():
        print(f"⚠ ERROR: Mean contains NaN or Inf!")
    if torch.isnan(std).any() or torch.isinf(std).any():
        print(f"⚠ ERROR: Std contains NaN or Inf!")
    
    # Feature breakdown (assuming 57 features for 23DOF)
    if mean.shape[0] == 57:
        print("\n" + "="*60)
        print("FEATURE BREAKDOWN (23DOF)")
        print("="*60)
        print(f"Root translation [0:3]:")
        print(f"  Mean: {mean[0:3].tolist()}")
        print(f"  Std:  {std[0:3].tolist()}")
        
        print(f"\nRoot rotation 6D [3:9]:")
        print(f"  Mean: {mean[3:9].tolist()}")
        print(f"  Std:  {std[3:9].tolist()}")
        
        print(f"\nDOF positions [9:32]:")
        print(f"  Mean range: [{mean[9:32].min().item():.4f}, {mean[9:32].max().item():.4f}]")
        print(f"  Std range:  [{std[9:32].min().item():.4f}, {std[9:32].max().item():.4f}]")
        
        print(f"\nDOF velocities [32:55]:")
        print(f"  Mean range: [{mean[32:55].min().item():.4f}, {mean[32:55].max().item():.4f}]")
        print(f"  Std range:  [{std[32:55].min().item():.4f}, {std[32:55].max().item():.4f}]")
        
        print(f"\nFoot contact [55:57]:")
        print(f"  Mean: {mean[55:57].tolist()}")
        print(f"  Std:  {std[55:57].tolist()}")
    
else:
    print(f"❌ meanstd.pkl not found at: {meanstd_path}")
