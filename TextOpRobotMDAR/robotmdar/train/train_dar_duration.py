"""
Duration-Conditioned DAR Training Script (Scheme 4)

This script trains a DAR denoiser that accepts duration as an additional condition.
The key idea: same text prompt + different duration = different rhythm/pace of motion.

Key features:
1. Uses ground truth duration during training
2. At inference, duration can come from:
   - User specification (explicit control)
   - Length Predictor (automatic prediction)
3. Does NOT modify the original train_dar.py

Usage:
    python train_dar_duration.py --config-name=train_dar_duration_humanml3d

Author: TextOp Team  
Date: 2026-01-03
"""

# Limit CPU threads
import os
num_threads = os.environ.get('OMP_NUM_THREADS', '30')
os.environ['OMP_NUM_THREADS'] = num_threads
os.environ['MKL_NUM_THREADS'] = num_threads
os.environ['OPENBLAS_NUM_THREADS'] = num_threads
os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_threads
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate

torch.set_num_threads(int(num_threads))

from robotmdar.dtype import seed, logger
from robotmdar.dtype.abc import VAE, Dataset, Denoiser, Diffusion, Optimizer, SSampler
from robotmdar.train.manager import DARManager

USE_VAE = True


def main(cfg: DictConfig):
    seed.set(cfg.seed)
    logger.set(cfg)

    train_data: Dataset = instantiate(cfg.data.train)
    val_data: Dataset = instantiate(cfg.data.val)

    vae: VAE = instantiate(cfg.vae)
    
    # Instantiate duration-conditioned denoiser
    denoiser: Denoiser = instantiate(cfg.denoiser)
    
    # Check if denoiser supports duration conditioning
    has_duration_cond = hasattr(denoiser, 'embed_duration') or \
                        cfg.denoiser.get('use_duration_cond', False)
    if has_duration_cond:
        print("[Duration-Conditioned DAR] Duration conditioning enabled")
    else:
        print("[Warning] Denoiser does not have duration conditioning. "
              "Make sure you're using the right denoiser config.")

    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    optimizer: Optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=cfg.train.manager.learning_rate)

    # Load pretrained DAR weights BEFORE manager.hold_model (which uses strict=True)
    # We need to load with strict=False to allow new duration parameters
    if cfg.ckpt.dar is not None:
        print(f"[Duration-Conditioned DAR] Loading pretrained DAR from: {cfg.ckpt.dar}")
        ckpt = torch.load(cfg.ckpt.dar, map_location=cfg.device)
        if 'denoiser' in ckpt:
            state_dict = ckpt['denoiser']
        else:
            state_dict = ckpt
        
        # Load with strict=False to allow missing duration keys
        missing, unexpected = denoiser.load_state_dict(state_dict, strict=False)
        
        # Report what was loaded
        duration_keys = ['embed_duration', 'null_duration_emb']
        new_params = [k for k in missing if any(dk in k for dk in duration_keys)]
        other_missing = [k for k in missing if not any(dk in k for dk in duration_keys)]
        
        print(f"  ✓ Loaded {len(state_dict) - len(unexpected)} params from pretrained DAR")
        print(f"  ✓ New duration params (randomly initialized): {len(new_params)}")
        if other_missing:
            print(f"  ⚠ Other missing keys: {other_missing}")
        if unexpected:
            print(f"  ⚠ Unexpected keys (ignored): {len(unexpected)}")
        
        # Clear dar checkpoint path so manager doesn't try to load it again
        cfg.ckpt.dar = None

    manager: DARManager = instantiate(cfg.train.manager)
    manager.hold_model(vae, denoiser, optimizer, train_data)

    num_primitive: int = cfg.data.num_primitive
    future_len: int = cfg.data.future_len
    history_len: int = cfg.data.history_len

    # Duration conditioning params
    max_frames = cfg.get('max_frames', 300)  # Maximum motion length
    duration_mask_prob = cfg.get('duration_mask_prob', 0.1)  # For CFG

    # DataLoader setup
    num_workers = cfg.get('num_workers', 0)
    
    import torch.multiprocessing as mp
    if num_workers > 0:
        mp.set_sharing_strategy('file_system')
    
    train_loader = DataLoader(
        train_data, 
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_data,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    train_dataiter = iter(train_loader)
    val_dataiter = iter(val_loader)

    # Training loop
    while manager:
        denoiser.train()
        batch = next(train_dataiter)

        prev_motion = None
        for pidx in range(num_primitive):
            manager.pre_step()
            
            # Unpack batch - now includes duration
            if len(batch[pidx]) == 3:
                motion, cond, duration = batch[pidx]
                duration = duration.to(cfg.device)
            else:
                motion, cond = batch[pidx]
                # Fallback: compute duration from motion length
                duration = torch.full((motion.shape[0],), motion.shape[1], 
                                      dtype=torch.float32, device=cfg.device)
            
            motion, cond = motion.to(cfg.device), cond.to(cfg.device)

            future_motion_gt = motion[:, -future_len:, :]
            gt_history = motion[:, :history_len, :]

            history_motion = manager.choose_history(gt_history, prev_motion, history_len)

            batch_size = motion.shape[0]
            t, weights = schedule_sampler.sample(batch_size, device=cfg.device)

            if USE_VAE:
                latent_gt, _ = vae.encode(
                    future_motion=future_motion_gt,
                    history_motion=history_motion
                )
                x_start = latent_gt.permute(1, 0, 2)
            else:
                latent_gt = None
                x_start = torch.cat((history_motion, future_motion_gt), dim=1)

            x_t = diffusion.q_sample(x_start=x_start, t=t, noise=torch.randn_like(x_start))

            # Build condition dict with duration
            y = {
                'text_embedding': cond,
                'history_motion_normalized': history_motion,
                'duration': duration,  # Add duration condition
                'duration_type': 'frames',
            }
            
            # Optional: duration masking for classifier-free guidance
            if duration_mask_prob > 0 and torch.rand(1).item() < duration_mask_prob:
                y['uncond_duration'] = True

            x_start_pred = denoiser(x_t=x_t, timesteps=diffusion._scale_timesteps(t), y=y)

            if USE_VAE:
                latent_pred = x_start_pred.permute(1, 0, 2)
                future_motion_pred = vae.decode(latent_pred, history_motion, nfuture=future_len)
            else:
                latent_pred = None
                future_motion_pred = x_start_pred[:, -future_len:]

            loss_dict, extras = manager.calc_loss(
                future_motion_gt, future_motion_pred,
                latent_gt, None, latent_pred, weights,
                history_motion=history_motion
            )
            loss = loss_dict['total']

            optimizer.zero_grad()
            loss.backward()
            
            has_nan_grad = False
            for param in denoiser.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True

            if not has_nan_grad:
                manager.grad_clip(denoiser)
                optimizer.step()

            # Update prev_motion for next primitive
            if manager.should_use_full_sample():
                with torch.no_grad():
                    sample_fn = diffusion.p_sample_loop
                    x_start_full = sample_fn(
                        denoiser, x_start.shape, clip_denoised=False,
                        model_kwargs={'y': y}, skip_timesteps=0,
                        init_image=None, progress=False, dump_steps=None,
                        noise=None, const_noise=False,
                    )
                    if isinstance(x_start_full, torch.Tensor):
                        latent_full = x_start_full.permute(1, 0, 2)
                    else:
                        latent_full = latent_pred
                    future_motion_full = vae.decode(latent_full, history_motion, nfuture=future_len)
                    prev_motion = torch.cat([history_motion, future_motion_full], dim=1).detach()
            else:
                prev_motion = torch.cat([history_motion, future_motion_pred], dim=1).detach()

            manager.post_step(
                is_eval=False,
                loss_dict={k: v.detach().cpu() for k, v in loss_dict.items()},
                extras={k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in extras.items()}
            )

        # Validation loop
        denoiser.eval()
        while manager.should_eval():
            batch = next(val_dataiter)
            for pidx in range(num_primitive):
                manager.pre_step(is_eval=True)
                
                if len(batch[pidx]) == 3:
                    motion, cond, duration = batch[pidx]
                    duration = duration.to(cfg.device)
                else:
                    motion, cond = batch[pidx]
                    duration = torch.full((motion.shape[0],), motion.shape[1],
                                          dtype=torch.float32, device=cfg.device)
                
                motion, cond = motion.to(cfg.device), cond.to(cfg.device)

                future_motion_gt = motion[:, -future_len:, :]
                history_motion = motion[:, :history_len, :]

                with torch.no_grad():
                    t, weights = schedule_sampler.sample(motion.shape[0], device=cfg.device)

                    if USE_VAE:
                        latent_gt, _ = vae.encode(
                            future_motion=future_motion_gt,
                            history_motion=history_motion
                        )
                        x_start = latent_gt.permute(1, 0, 2)
                    else:
                        latent_gt = None
                        x_start = torch.cat((history_motion, future_motion_gt), dim=1)

                    x_t = diffusion.q_sample(x_start=x_start, t=t, noise=torch.randn_like(x_start))

                    y = {
                        'text_embedding': cond,
                        'history_motion_normalized': history_motion,
                        'duration': duration,
                        'duration_type': 'frames',
                    }
                    
                    x_start_pred = denoiser(x_t=x_t, timesteps=diffusion._scale_timesteps(t), y=y)

                    if USE_VAE:
                        latent_pred = x_start_pred.permute(1, 0, 2)
                        future_motion_pred = vae.decode(latent_pred, history_motion, nfuture=future_len)
                    else:
                        latent_pred = None
                        future_motion_pred = x_start_pred[:, -future_len:]

                    loss_dict, extras = manager.calc_loss(
                        future_motion_gt, future_motion_pred,
                        latent_gt, None, latent_pred, weights,
                        history_motion=history_motion
                    )

                manager.post_step(
                    is_eval=True,
                    loss_dict={k: v.detach().cpu() for k, v in loss_dict.items()},
                    extras={k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in extras.items()}
                )
