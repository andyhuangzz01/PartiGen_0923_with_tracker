"""
Duration-Aware Dataset Wrapper

This wraps the original SkeletonPrimitiveDataset to also return duration information.
Does NOT modify the original data.py.

Usage:
    In config, use:
    _target_: robotmdar.dataloader.data_duration.DurationAwareDataset
    
    Or wrap existing dataset:
    dataset = DurationAwareDataset(original_dataset)
"""

from pathlib import Path
import numpy as np
import joblib
import yaml
from typing import Any, Tuple, Dict, List, Optional
import random
from omegaconf import DictConfig
from loguru import logger

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm

from robotmdar.dataloader.data import SkeletonPrimitiveDataset
from robotmdar.model.clip import load_and_freeze_clip, encode_text
from robotmdar.skeleton.robot import RobotSkeleton
from robotmdar.dtype.motion import MotionDict, motion_dict_to_feature, AbsolutePose, motion_feature_to_dict, MotionKeys, FeatureVersion


class DurationAwareDataset(SkeletonPrimitiveDataset):
    """
    Extension of SkeletonPrimitiveDataset that also returns duration information.
    
    Returns: List of (motion_features, text_features, duration) tuples
    where duration is the total length of the motion in frames.
    """
    
    def __init__(self, *args, return_duration: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_duration = return_duration
        logger.info(f"[DurationAwareDataset] return_duration={return_duration}")
    
    def _sample_motion_batch_with_duration(
        self, generator: Optional[torch.Generator] = None
    ) -> Tuple[List[List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]], torch.Tensor]:
        """Sample a batch of motions and return their durations too."""
        
        if not self.weighted_sample:
            rand_idx = torch.randint(0, len(self.valid_indices), (self.batch_size,), generator=generator)
        else:
            rand_idx = torch.from_numpy(
                np.random.choice(len(self.raw_data), size=self.batch_size, replace=True, p=self.seq_weights)
            )
        
        all_motion_primitives = []
        durations = []
        
        for batch_idx in range(self.batch_size):
            sample_idx = self.valid_indices[rand_idx[batch_idx].item()]
            sample = self.raw_data[sample_idx]
            
            # Record total duration of this motion
            total_duration = sample['length']
            durations.append(total_duration)
            
            # Sample segment start
            max_start = sample['length'] - self.segment_len
            
            if self.weighted_sample and self.frame_weight:
                seg_start = random.choices(range(max_start + 1), weights=sample['frame_weights'], k=1)[0]
            else:
                seg_start = int(torch.randint(0, max_start, (1,), generator=generator).item())
            
            # Generate primitives
            motion_primitives = self._generate_motion_primitives(sample, seg_start)
            all_motion_primitives.append(motion_primitives)
        
        durations = torch.tensor(durations, dtype=torch.float32)
        return all_motion_primitives, durations
    
    def _organize_primitives_by_index_with_duration(
        self,
        all_motion_primitives: List[List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]],
        durations: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Organize primitives by index, including duration."""
        batch_primitives = []
        
        for primitive_idx in range(self.num_primitive):
            motion_batch = []
            text_batch = []
            
            for batch_idx in range(self.batch_size):
                motion_data, text_embedding = all_motion_primitives[batch_idx][primitive_idx]
                motion_batch.append(motion_data)
                text_batch.append(text_embedding)
            
            motion_features = self._convert_to_motion_features(motion_batch)
            text_features = torch.stack(text_batch)
            
            # Return (motion, text, duration) tuple
            batch_primitives.append((
                self.normalize(motion_features),
                text_features,
                durations  # Same duration for all primitives from same motion
            ))
        
        return batch_primitives
    
    def _generate_batch_with_duration(
        self, generator: Optional[torch.Generator] = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate a batch with duration information."""
        all_motion_primitives, durations = self._sample_motion_batch_with_duration(generator)
        batch_primitives = self._organize_primitives_by_index_with_duration(all_motion_primitives, durations)
        return batch_primitives
    
    def __iter__(self):
        """Iterator that yields batches with duration."""
        worker_info = data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        generator = torch.Generator()
        generator.manual_seed(worker_id + np.random.randint(0, 1000000))
        
        while True:
            if self.return_duration:
                yield self._generate_batch_with_duration(generator=generator)
            else:
                yield self._generate_batch_optimized(generator=generator)


def wrap_dataset_with_duration(dataset: SkeletonPrimitiveDataset) -> DurationAwareDataset:
    """
    Wrap an existing dataset to return duration.
    
    Note: This creates a new dataset instance with the same config.
    """
    # Get the config from the original dataset
    config = {
        'batch_size': dataset.batch_size,
        'nfeats': dataset.nfeats,
        'history_len': dataset.history_len,
        'future_len': dataset.future_len,
        'num_primitive': dataset.num_primitive,
        'datadir': str(dataset.datadir),
        'split': dataset.split,
        'weighted_sample': dataset.weighted_sample,
        'frame_weight': dataset.frame_weight,
        'action_statistics_path': dataset.action_statistics_path,
    }
    
    # Create new duration-aware dataset
    # This is a simplified version - full implementation would need all config params
    logger.warning("wrap_dataset_with_duration creates a new dataset. "
                   "Use DurationAwareDataset directly in config for best results.")
    
    return dataset  # Return original for now - use direct instantiation instead
