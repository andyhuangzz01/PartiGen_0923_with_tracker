"""Endpoint-aware sampling over the existing server PKL and meanstd.pkl assets."""
import math
import random

import torch

from robotmdar.dataloader.data import SkeletonPrimitiveDataset
from robotmdar.dtype.motion import MotionKeys, FeatureVersion
from .eos import primitive_targets


class PartiGenDataset(SkeletonPrimitiveDataset):
    def __init__(self, *args, max_frames=320, **kwargs):
        self.max_frames = max_frames
        super().__init__(*args, **kwargs)
        if FeatureVersion != 3 or self.nfeats != 57 or self.fps != 30:
            raise ValueError("PartiGen requires FeatureVersion=3, 57 features and 30 Hz data")

    def _load_data(self):
        super()._load_data()
        # Short clips and final partial primitives are essential for EOS labels.
        self.valid_indices = [i for i, item in enumerate(self.raw_data) if item['length'] >= 1]
        if not self.valid_indices:
            raise ValueError("No motion samples in the selected split")

    def _load_text_embeddings(self):
        # Never silently use the old loader's Qwen cache for a CLIP experiment.
        path = self.datadir / f'{self.split}_text_embed.pkl'
        if path.exists():
            self.text_embeddings_dict = torch.load(path, map_location='cpu', weights_only=False)
        else:
            from robotmdar.model.clip import load_and_freeze_clip
            model = load_and_freeze_clip('ViT-B/32', 'cuda' if torch.cuda.is_available() else 'cpu')
            self.text_embeddings_dict = self._compute_text_embeddings(self.raw_data, model)
            torch.save(self.text_embeddings_dict, path)
        for text, embedding in self.text_embeddings_dict.items():
            if tuple(embedding.shape) != (512,):
                raise ValueError(f"CLIP embedding for {text!r} must have width 512")
        self.text_embeddings_dict.setdefault('', torch.zeros(512))

    def _sample_one(self, generator):
        index = torch.randint(len(self.valid_indices), (1,), generator=generator).item()
        sample = self.raw_data[self.valid_indices[index]]
        annotations = []
        for ann in sample['frame_ann']:
            start = max(0, round(float(ann[0]) * self.fps))
            end = min(sample['length'], round(float(ann[1]) * self.fps))
            if end > start:
                annotations.append((start, end, ann[2]))
        start, end, text = random.choice(annotations) if annotations else (0, sample['length'], '')
        blocks = math.ceil(min(end - start, self.max_frames) / self.future_len)
        block = torch.randint(blocks, (1,), generator=generator).item()
        indices, valid, eos, positions = primitive_targets(
            start, end, block, self.future_len, self.history_len, self.max_frames,
            sequence_len=sample['length'])
        motion = {key: torch.as_tensor(sample['motion'][key], dtype=torch.float32)[indices]
                  for key in MotionKeys}
        if text not in self.text_embeddings_dict:
            raise KeyError(f"CLIP cache is missing annotation {text!r}")
        return motion, self.text_embeddings_dict[text].float(), valid, eos, positions

    def _generate_batch_optimized(self, generator=None):
        examples = [self._sample_one(generator) for _ in range(self.batch_size)]
        frames = self._convert_to_motion_features([item[0] for item in examples])
        return {'motion': self.normalize(frames),
                'text': torch.stack([item[1] for item in examples]),
                'valid': torch.stack([item[2] for item in examples]),
                'eos': torch.stack([item[3] for item in examples]),
                'positions': torch.stack([item[4] for item in examples])}

    def _compute_meanstd(self):
        # Cache format and sampled-training-statistics convention stay unchanged.
        self.mean, self.std = torch.zeros(self.nfeats), torch.ones(self.nfeats)
        total, square, count = torch.zeros(self.nfeats), torch.zeros(self.nfeats), 0
        for i in range(10000 // self.batch_size + 1):
            batch = self._generate_batch_optimized(torch.Generator().manual_seed(i))
            valid = torch.cat((torch.ones(self.batch_size, self.history_len, dtype=torch.bool), batch['valid']), dim=1)
            frames = batch['motion'][valid]
            total += frames.sum(0)
            square += frames.square().sum(0)
            count += len(frames)
        mean = total / count
        return mean, (square / count - mean.square()).clamp_min(1e-12).sqrt()

    def normalize(self, feat):
        return (feat - self.mean.to(feat)) / self.std.to(feat).clamp_min(1e-6)
