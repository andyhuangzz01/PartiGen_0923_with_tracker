"""
Compute mean and std statistics for a dataset systematically.
This script uses ALL training data to compute accurate statistics.

Usage:
    python scripts/compute_meanstd.py --dataset HumanML3D-G1-23DOF-30fps
    python scripts/compute_meanstd.py --dataset HumanML3D-G1-29DOF-30fps
    python scripts/compute_meanstd.py --dataset BABEL-AMASS-ROBOT-23dof-MINIMAL-50fps
"""

import argparse
import joblib
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_meanstd_from_raw_data(dataset_path: Path, nfeats: int):
    """
    Compute mean and std from raw motion data in train.pkl
    
    Args:
        dataset_path: Path to dataset directory
        nfeats: Number of features (DOF dimensions)
    
    Returns:
        Tuple of (mean, std) tensors
    """
    train_pkl = dataset_path / 'train.pkl'
    
    if not train_pkl.exists():
        raise FileNotFoundError(f"Training data not found: {train_pkl}")
    
    print(f"Loading training data from {train_pkl}...")
    train_data = joblib.load(train_pkl)
    print(f"Loaded {len(train_data)} sequences")
    
    # Collect all motion features
    all_features = []
    
    print("Extracting motion features from all sequences...")
    for seq in tqdm(train_data):
        motion_dict = seq['motion']
        
        # Extract components from motion dictionary
        # Format: root_trans(3) + root_rot_6d(6) + dof(23) + dof_vel(23) + contact(2) = 57
        if isinstance(motion_dict, dict):
            # Get basic fields
            root_trans = motion_dict['root_trans_offset']  # [T, 3]
            root_rot_quat = motion_dict['root_rot']  # [T, 4] quaternion
            dof = motion_dict['dof']  # [T, 23]
            contact = motion_dict['contact_mask']  # [T, 2]
            
            # Convert to numpy if needed
            if not isinstance(root_trans, np.ndarray):
                root_trans = np.array(root_trans)
            if not isinstance(root_rot_quat, np.ndarray):
                root_rot_quat = np.array(root_rot_quat)
            if not isinstance(dof, np.ndarray):
                dof = np.array(dof)
            if not isinstance(contact, np.ndarray):
                contact = np.array(contact)
            
            # Convert quaternion to 6D rotation
            root_rot_6d = []
            for quat in root_rot_quat:
                rot_mat = R.from_quat(quat).as_matrix()  # [3, 3]
                # 6D representation: first two columns
                rot_6d = np.concatenate([rot_mat[:, 0], rot_mat[:, 1]])  # [6]
                root_rot_6d.append(rot_6d)
            root_rot_6d = np.array(root_rot_6d)  # [T, 6]
            
            # Compute DOF velocities
            dof_vel = np.zeros_like(dof)
            dof_vel[1:] = dof[1:] - dof[:-1]
            dof_vel[0] = dof_vel[1]  # Copy first frame
            
            # Concatenate all features: [T, 57]
            motion_features = np.concatenate([
                root_trans,    # [T, 3]
                root_rot_6d,   # [T, 6]
                dof,           # [T, 23]
                dof_vel,       # [T, 23]
                contact        # [T, 2]
            ], axis=1)
            
            motion = torch.from_numpy(motion_features).float()
        else:
            # Old format: already concatenated
            motion = motion_dict
            if isinstance(motion, np.ndarray):
                motion = torch.from_numpy(motion).float()
        
        all_features.append(motion)
    
    # Concatenate all features
    print("Concatenating features...")
    all_features = torch.cat(all_features, dim=0)  # Shape: [total_frames, nfeats]
    print(f"Total frames: {all_features.shape[0]}, Features: {all_features.shape[1]}")
    
    # Compute statistics
    print("Computing mean and std...")
    mean = all_features.mean(dim=0)  # Shape: [nfeats]
    std = all_features.std(dim=0)    # Shape: [nfeats]
    
    # Avoid division by zero
    std = torch.clamp(std, min=1e-6)
    
    print("\nStatistics computed:")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")
    print(f"  Mean range: [{mean.min().item():.4f}, {mean.max().item():.4f}]")
    print(f"  Std range: [{std.min().item():.4f}, {std.max().item():.4f}]")
    
    return mean, std


def save_meanstd(mean: torch.Tensor, std: torch.Tensor, output_path: Path):
    """Save mean and std to pickle file"""
    meanstd = (mean, std)
    torch.save(meanstd, output_path)
    print(f"\n✓ Saved meanstd to: {output_path}")


def update_statistics_yaml(dataset_path: Path, mean: torch.Tensor, std: torch.Tensor, fps: int, nfeats: int):
    """Update or create statistics.yaml with computed values"""
    stats_yaml = dataset_path / 'statistics.yaml'
    
    statistics = {
        'fps': fps,
        'nfeats': nfeats,
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    
    with open(stats_yaml, 'w') as f:
        yaml.dump(statistics, f, default_flow_style=False)
    
    print(f"✓ Updated statistics.yaml: {stats_yaml}")


def main():
    parser = argparse.ArgumentParser(description='Compute dataset mean/std statistics')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., HumanML3D-G1-23DOF-30fps)')
    parser.add_argument('--dataset-root', type=str, default='./dataset',
                        help='Root directory containing datasets')
    parser.add_argument('--nfeats', type=int, default=None,
                        help='Number of features (auto-detect if not specified)')
    parser.add_argument('--fps', type=int, default=None,
                        help='Frame rate (auto-detect from name if not specified)')
    parser.add_argument('--output-name', type=str, default='meanstd.pkl',
                        help='Output filename (default: meanstd.pkl)')
    parser.add_argument('--update-yaml', action='store_true',
                        help='Update statistics.yaml with computed values')
    
    args = parser.parse_args()
    
    # Setup paths
    dataset_path = Path(args.dataset_root) / args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Auto-detect parameters from dataset name
    if args.fps is None:
        if '50fps' in args.dataset:
            args.fps = 50
        elif '30fps' in args.dataset:
            args.fps = 30
        else:
            args.fps = 30  # default
        print(f"Auto-detected FPS: {args.fps}")
    
    if args.nfeats is None:
        if '23DOF' in args.dataset or '23dof' in args.dataset:
            args.nfeats = 57  # 3+6+23+23+2
        elif '29DOF' in args.dataset or '29dof' in args.dataset:
            args.nfeats = 69  # Adjust based on actual format
        else:
            # Try to detect from data
            train_pkl = dataset_path / 'train.pkl'
            train_data = joblib.load(train_pkl)
            args.nfeats = train_data[0]['motion'].shape[1]
        print(f"Auto-detected nfeats: {args.nfeats}")
    
    print(f"\nDataset: {args.dataset}")
    print(f"Path: {dataset_path}")
    print(f"FPS: {args.fps}")
    print(f"Features: {args.nfeats}")
    print(f"Output: {args.output_name}\n")
    
    # Compute statistics
    mean, std = compute_meanstd_from_raw_data(dataset_path, args.nfeats)
    
    # Save results
    output_path = dataset_path / args.output_name
    save_meanstd(mean, std, output_path)
    
    # Optionally update statistics.yaml
    if args.update_yaml:
        update_statistics_yaml(dataset_path, mean, std, args.fps, args.nfeats)
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Total features: {args.nfeats}")
    print(f"  Output file: {output_path}")
    print(f"  Mean (first 5): {mean[:5].tolist()}")
    print(f"  Std (first 5): {std[:5].tolist()}")
    print("="*60)
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
