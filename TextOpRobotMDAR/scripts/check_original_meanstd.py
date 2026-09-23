"""
Check the existing mean_std_23dof.pkl to see if it's correct
"""
import torch
import numpy as np
from pathlib import Path

# Check the existing mean_std_23dof.pkl
dataset_path = Path('./dataset/HumanML3D-G1-23DOF-30fps')
meanstd_original = dataset_path / 'mean_std_23dof.pkl'
meanstd_current = dataset_path / 'meanstd.pkl'

print("="*80)
print("CHECKING mean_std_23dof.pkl (ORIGINAL)")
print("="*80)

if meanstd_original.exists():
    mean_orig, std_orig = torch.load(meanstd_original, weights_only=False)
    
    print(f"Shape: {mean_orig.shape}")
    print(f"Data type: {mean_orig.dtype}")
    
    print("\n" + "-"*80)
    print("MEAN STATISTICS")
    print("-"*80)
    print(f"Min: {mean_orig.min().item():.6f}")
    print(f"Max: {mean_orig.max().item():.6f}")
    print(f"Mean of means: {mean_orig.mean().item():.6f}")
    
    print("\n" + "-"*80)
    print("STD STATISTICS")
    print("-"*80)
    print(f"Min: {std_orig.min().item():.6f}")
    print(f"Max: {std_orig.max().item():.6f}")
    print(f"Mean of stds: {std_orig.mean().item():.6f}")
    
    # Feature breakdown (assuming 57 features for 23DOF)
    if mean_orig.shape[0] == 57:
        print("\n" + "="*80)
        print("FEATURE BREAKDOWN (23DOF = 57 dimensions)")
        print("="*80)
        
        print(f"\n[0:3] Root translation:")
        print(f"  Mean: {mean_orig[0:3].tolist()}")
        print(f"  Std:  {std_orig[0:3].tolist()}")
        
        print(f"\n[3:9] Root rotation 6D:")
        print(f"  Mean: {mean_orig[3:9].tolist()}")
        print(f"  Std:  {std_orig[3:9].tolist()}")
        
        # Check for duplicate values in rotation
        if abs(mean_orig[5].item() - mean_orig[6].item()) < 1e-6:
            print(f"  ⚠ WARNING: Dims [5] and [6] have identical mean! (bug)")
        else:
            print(f"  ✓ Rotation means look different (good)")
        
        print(f"\n[9:32] DOF positions (23 DOFs):")
        print(f"  Mean range: [{mean_orig[9:32].min().item():.4f}, {mean_orig[9:32].max().item():.4f}]")
        print(f"  Std range:  [{std_orig[9:32].min().item():.4f}, {std_orig[9:32].max().item():.4f}]")
        print(f"  Mean values (first 5): {mean_orig[9:14].tolist()}")
        print(f"  Std values (first 5): {std_orig[9:14].tolist()}")
        
        print(f"\n[32:55] DOF velocities (23 DOFs):")
        print(f"  Mean range: [{mean_orig[32:55].min().item():.6f}, {mean_orig[32:55].max().item():.6f}]")
        print(f"  Std range:  [{std_orig[32:55].min().item():.4f}, {std_orig[32:55].max().item():.4f}]")
        
        # Check if velocity means are close to zero (they should be)
        vel_mean_abs = mean_orig[32:55].abs()
        if (vel_mean_abs < 0.01).all():
            print(f"  ✓ All velocity means < 0.01 (good, should be ~0)")
        else:
            large_vel = torch.where(vel_mean_abs >= 0.01)[0]
            print(f"  ⚠ WARNING: {len(large_vel)} velocity dims have |mean| >= 0.01:")
            print(f"    Indices: {large_vel.tolist()}")
            print(f"    Values: {mean_orig[32 + large_vel].tolist()}")
        
        print(f"\n[55:57] Foot contact (2 dims):")
        print(f"  Mean: {mean_orig[55:57].tolist()}")
        print(f"  Std:  {std_orig[55:57].tolist()}")
        
        # Check foot contact std (should be ~0.5 for binary values)
        if std_orig[55] > 0.3 and std_orig[56] > 0.3:
            print(f"  ✓ Contact std > 0.3 (reasonable for binary variable)")
        else:
            print(f"  ⚠ WARNING: Contact std too small (expected ~0.5 for binary)")
    
    # Diagnostics
    print("\n" + "="*80)
    print("DIAGNOSTICS")
    print("="*80)
    
    issues = []
    
    # Check for NaN or Inf
    if torch.isnan(mean_orig).any():
        issues.append("❌ Mean contains NaN")
    if torch.isinf(mean_orig).any():
        issues.append("❌ Mean contains Inf")
    if torch.isnan(std_orig).any():
        issues.append("❌ Std contains NaN")
    if torch.isinf(std_orig).any():
        issues.append("❌ Std contains Inf")
    
    # Check for very small std (near-constant features)
    small_std_mask = std_orig < 1e-4
    if small_std_mask.any():
        issues.append(f"⚠ {small_std_mask.sum()} dims have std < 1e-4 (near-constant)")
    
    # Check for suspiciously large std
    large_std_mask = std_orig > 10
    if large_std_mask.any():
        issues.append(f"⚠ {large_std_mask.sum()} dims have std > 10 (very large variance)")
    
    if not issues:
        print("✓ No obvious issues detected")
    else:
        for issue in issues:
            print(issue)
    
else:
    print("❌ mean_std_23dof.pkl NOT FOUND!")
    print(f"   Expected path: {meanstd_original}")

# Compare with current meanstd.pkl
if meanstd_current.exists() and meanstd_original.exists():
    print("\n" + "="*80)
    print("COMPARISON: mean_std_23dof.pkl vs meanstd.pkl")
    print("="*80)
    
    mean_cur, std_cur = torch.load(meanstd_current, weights_only=False)
    
    mean_diff = (mean_orig - mean_cur).abs().max().item()
    std_diff = (std_orig - std_cur).abs().max().item()
    
    print(f"Max absolute difference in mean: {mean_diff:.6f}")
    print(f"Max absolute difference in std:  {std_diff:.6f}")
    
    if mean_diff < 1e-6 and std_diff < 1e-6:
        print("✓ Files are identical (difference < 1e-6)")
    elif mean_diff < 1e-3 and std_diff < 1e-3:
        print("⚠ Files are similar but not identical (difference < 1e-3)")
    else:
        print("❌ Files are DIFFERENT!")
        print("\nShowing first 10 dimensions:")
        print("  mean_std_23dof mean:", mean_orig[:10].tolist())
        print("  meanstd.pkl mean:   ", mean_cur[:10].tolist())

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
if meanstd_original.exists():
    print("Copy the correct file:")
    print("  cp dataset/HumanML3D-G1-23DOF-30fps/mean_std_23dof.pkl \\")
    print("     dataset/HumanML3D-G1-23DOF-30fps/meanstd.pkl")
else:
    print("Recompute statistics from scratch:")
    print("  python TextOpRobotMDAR/scripts/compute_meanstd.py \\")
    print("         --dataset HumanML3D-G1-23DOF-30fps --update-yaml")
