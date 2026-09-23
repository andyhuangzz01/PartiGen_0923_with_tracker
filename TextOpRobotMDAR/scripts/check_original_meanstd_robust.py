"""
Robust checker for mean_std_23dof.pkl with multiple loading methods
"""
import torch
import numpy as np
import pickle
import joblib
from pathlib import Path

dataset_path = Path('./dataset/HumanML3D-G1-23DOF-30fps')
meanstd_original = dataset_path / 'mean_std_23dof.pkl'

print("="*80)
print("CHECKING mean_std_23dof.pkl")
print("="*80)
print(f"File path: {meanstd_original}")
print(f"File exists: {meanstd_original.exists()}")

if meanstd_original.exists():
    file_size = meanstd_original.stat().st_size
    print(f"File size: {file_size} bytes")
    
    # Try different loading methods
    methods = [
        ("torch.load", lambda f: torch.load(f, weights_only=False)),
        ("torch.load with map_location", lambda f: torch.load(f, map_location='cpu', weights_only=False)),
        ("pickle.load", lambda f: pickle.load(open(f, 'rb'))),
        ("joblib.load", lambda f: joblib.load(f)),
        ("numpy.load", lambda f: np.load(f, allow_pickle=True)),
    ]
    
    loaded = False
    for method_name, method_func in methods:
        try:
            print(f"\n{'='*80}")
            print(f"Trying: {method_name}")
            print(f"{'='*80}")
            
            data = method_func(meanstd_original)
            
            print(f"✓ Successfully loaded with {method_name}")
            print(f"Data type: {type(data)}")
            
            # Try to extract mean and std
            if isinstance(data, tuple) and len(data) == 2:
                mean_orig, std_orig = data
                print(f"Tuple of 2 elements (mean, std)")
            elif isinstance(data, dict):
                print(f"Dictionary with keys: {data.keys()}")
                if 'mean' in data and 'std' in data:
                    mean_orig = data['mean']
                    std_orig = data['std']
                else:
                    print(f"  ⚠ No 'mean'/'std' keys found")
                    continue
            elif isinstance(data, np.ndarray):
                print(f"NumPy array with shape: {data.shape}")
                if len(data.shape) == 2 and data.shape[0] == 2:
                    mean_orig = torch.from_numpy(data[0])
                    std_orig = torch.from_numpy(data[1])
                else:
                    print(f"  ⚠ Unexpected array shape")
                    continue
            else:
                print(f"  ⚠ Unknown data structure: {type(data)}")
                continue
            
            # Convert to torch tensors if needed
            if not isinstance(mean_orig, torch.Tensor):
                mean_orig = torch.from_numpy(np.array(mean_orig))
            if not isinstance(std_orig, torch.Tensor):
                std_orig = torch.from_numpy(np.array(std_orig))
            
            print(f"\nMean shape: {mean_orig.shape}, dtype: {mean_orig.dtype}")
            print(f"Std shape: {std_orig.shape}, dtype: {std_orig.dtype}")
            
            # Validate the data
            if mean_orig.shape[0] == 57:
                print("\n" + "="*80)
                print("VALIDATION RESULTS")
                print("="*80)
                
                print(f"\n[3:9] Root rotation 6D:")
                print(f"  Mean: {mean_orig[3:9].tolist()}")
                print(f"  Std:  {std_orig[3:9].tolist()}")
                
                # Check for duplicate rotation bug
                rot_5_6_diff = abs(mean_orig[5].item() - mean_orig[6].item())
                if rot_5_6_diff < 1e-6:
                    print(f"  ❌ PROBLEM: Rotation dims [5] and [6] are identical ({mean_orig[5].item():.6f})")
                    print(f"     This file has the SAME BUG as meanstd.pkl!")
                else:
                    print(f"  ✓ Rotation dims [5] and [6] are different (good)")
                
                print(f"\n[32:55] DOF velocities:")
                vel_mean_abs = mean_orig[32:55].abs()
                vel_mean_max = vel_mean_abs.max().item()
                print(f"  Max |mean|: {vel_mean_max:.6f}")
                
                if vel_mean_max < 0.01:
                    print(f"  ✓ All velocity means < 0.01 (good)")
                elif vel_mean_max < 0.1:
                    print(f"  ⚠ Some velocity means up to {vel_mean_max:.6f} (acceptable)")
                else:
                    print(f"  ❌ PROBLEM: Velocity mean too large ({vel_mean_max:.6f})")
                
                print(f"\n[55:57] Foot contact:")
                print(f"  Mean: {mean_orig[55:57].tolist()}")
                print(f"  Std:  {std_orig[55:57].tolist()}")
                
                if std_orig[55] > 0.3 and std_orig[56] > 0.3:
                    print(f"  ✓ Contact std > 0.3 (reasonable for binary)")
                else:
                    print(f"  ❌ PROBLEM: Contact std too small ({std_orig[55:57].tolist()})")
                
                # Final verdict
                print("\n" + "="*80)
                print("VERDICT")
                print("="*80)
                
                issues = []
                if rot_5_6_diff < 1e-6:
                    issues.append("Rotation bug (dims 5&6 identical)")
                if vel_mean_max > 0.1:
                    issues.append("Velocity mean too large")
                if std_orig[55] < 0.3 or std_orig[56] < 0.3:
                    issues.append("Contact std too small")
                
                if not issues:
                    print("✓ This file looks CORRECT!")
                    print("\nCopy it to meanstd.pkl:")
                    print(f"  cp {meanstd_original} {dataset_path / 'meanstd.pkl'}")
                else:
                    print("❌ This file has PROBLEMS:")
                    for issue in issues:
                        print(f"  - {issue}")
                    print("\nNeed to recompute from scratch:")
                    print("  python TextOpRobotMDAR/scripts/compute_meanstd.py \\")
                    print("         --dataset HumanML3D-G1-23DOF-30fps --update-yaml")
            
            loaded = True
            break
            
        except Exception as e:
            print(f"✗ Failed with {method_name}: {e}")
    
    if not loaded:
        print("\n" + "="*80)
        print("❌ FAILED TO LOAD WITH ANY METHOD")
        print("="*80)
        print("The file may be corrupted or in an unknown format.")
        print("\nRecommendation: Recompute statistics from scratch:")
        print("  python TextOpRobotMDAR/scripts/compute_meanstd.py \\")
        print("         --dataset HumanML3D-G1-23DOF-30fps --update-yaml")

else:
    print("\n❌ File does not exist!")
    print("\nRecompute statistics:")
    print("  python TextOpRobotMDAR/scripts/compute_meanstd.py \\")
    print("         --dataset HumanML3D-G1-23DOF-30fps --update-yaml")
