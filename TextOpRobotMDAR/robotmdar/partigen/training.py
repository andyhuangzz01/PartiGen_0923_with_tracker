"""Two-stage training with explicit (mandatory) experiment hyperparameters."""
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .eos import eos_loss
from .losses import geometry_terms, learning_rate_at, masked_huber, stage_at, weighted_geometry


def missing_fields(cfg):
    """Collect mandatory fields without resolving references to missing values."""
    from omegaconf import OmegaConf
    result = []
    def walk(value, prefix=''):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f'{prefix}.{key}' if prefix else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f'{prefix}[{index}]')
        elif value == '???':
            result.append(prefix)
    walk(OmegaConf.to_container(cfg, resolve=False))
    return sorted(result)


def require_complete_config(cfg):
    missing = missing_fields(cfg)
    if missing:
        raise ValueError("Provide explicit experiment parameters; no training defaults are supplied:\n  "
                         + "\n  ".join(missing))
    if cfg.data.future_len not in (8, 16):
        raise ValueError("The paper supports 8-frame and 16-frame experiments")
    if cfg.train.lr <= 0 or cfg.train.batch_size <= 0 or cfg.train.eval_batch_size <= 0:
        raise ValueError("Learning rate and batch sizes must be positive")
    if not cfg.train.stages or any(int(n) <= 0 for n in cfg.train.stages):
        raise ValueError("train.stages must contain positive step counts")
    if cfg.train.save_every <= 0 or cfg.train.eval_every <= 0 or cfg.train.eval_steps <= 0:
        raise ValueError("Checkpoint and evaluation intervals must be positive")
    if cfg.train.max_grad_norm <= 0:
        raise ValueError("Gradient clipping norm must be positive")
    if cfg.stage not in ('mvae', 'dar'):
        raise ValueError("stage must be mvae or dar")
    if cfg.data.history_len != 2:
        raise ValueError("The reconstructed model requires two history frames")
    if cfg.stage == 'dar' and not (cfg.ckpt.vae or cfg.ckpt.dar):
        raise ValueError("Stage 2 requires ckpt.vae or a self-contained ckpt.dar")
    if cfg.stage == 'mvae':
        for name, index in cfg.train.loss_start_stage.items():
            if not isinstance(index, int) or not 0 <= index < len(cfg.train.stages):
                raise ValueError(f"Invalid activation stage for {name}: {index}")
    learning_rate_at(0, sum(cfg.train.stages), cfg.train.lr, cfg.train.lr_schedule)


def active_geometry_weights(cfg, step):
    weights = dict(cfg.train.geometry_weights)
    stage = stage_at(step, cfg.train.stages)
    if cfg.stage == 'mvae':
        for key in weights:
            if stage < cfg.train.loss_start_stage[key]:
                weights[key] = 0.0
    return weights


def compute_objective(cfg, vae, denoiser, diffusion, dataset, batch, step):
    history = batch['motion'][:, :cfg.data.history_len]
    future = batch['motion'][:, cfg.data.history_len:]
    valid = batch['valid']
    weights = active_geometry_weights(cfg, step)
    temporal = any(weights.get(key, 0) for key in ('smooth', 'joints_delta', 'trans_delta', 'orient_delta', 'dof_delta'))
    if cfg.stage == 'mvae':
        latent, dist = vae.encode(future, history, valid_mask=valid)
        pred = vae.decode(latent, history, future.shape[1])
        terms = {'rec': masked_huber(pred, future, valid)}
        prior = torch.distributions.Normal(torch.zeros_like(dist.loc), torch.ones_like(dist.scale))
        terms['kl'] = torch.distributions.kl_divergence(dist, prior).mean()
        geometry = geometry_terms(dataset, pred, future, history, valid, temporal=temporal)
        kl_active = stage_at(step, cfg.train.stages) >= cfg.train.loss_start_stage.kl
        total = (cfg.train.rec_weight * terms['rec'] + cfg.train.kl_weight * kl_active * terms['kl']
                 + cfg.train.kinematic_weight * weighted_geometry(geometry, weights))
    else:
        # Frozen VAE encoding is detached; decoding remains differentiable in z.
        with torch.no_grad():
            latent, _ = vae.encode(future, history, valid_mask=valid)
        clean = latent.transpose(0, 1)
        timesteps = torch.randint(diffusion.num_timesteps, (len(clean),), device=clean.device)
        noisy = diffusion.q_sample(clean, timesteps, noise=torch.randn_like(clean))
        y = {'text_embedding': batch['text'], 'history_motion_normalized': history}
        estimated, conditioned_text = denoiser(noisy, timesteps, y=y, return_condition=True)
        pred = vae.decode(estimated.transpose(0, 1), history, future.shape[1])
        terms = {'rec': masked_huber(pred, future, valid),
                 'latent_rec': F.huber_loss(estimated, clean, delta=1.0)}
        logits = denoiser.predict_eos(pred, estimated, conditioned_text, batch['positions'], cfg.sampling.max_frames)
        terms['eos'] = eos_loss(logits, batch['eos'], valid, cfg.train.eos_positive_weight)
        geometry = geometry_terms(dataset, pred, future, history, valid, temporal=temporal)
        total = (cfg.train.rec_weight * terms['rec'] + cfg.train.latent_weight * terms['latent_rec']
                 + cfg.train.kinematic_weight * weighted_geometry(geometry, weights)
                 + cfg.train.eos_weight * terms['eos'])
    terms.update(geometry)
    terms['total'] = total
    return terms


def load_weights(model, path, key, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(checkpoint[key], strict=True)
    except (RuntimeError, KeyError) as exc:
        raise ValueError(f"{path} does not match this PartiGen {key} architecture. "
                         "Use the matching architecture/config; legacy BodyPart weights are not interchangeable.") from exc
    return checkpoint


def checkpoint_state(cfg, vae, denoiser, optimizer, step):
    from omegaconf import OmegaConf
    state = {'format': 'partigen-v1', 'step': step, 'cfg': OmegaConf.to_container(cfg, resolve=True),
             'optimizer': optimizer.state_dict(), 'vae': vae.state_dict(),
             'rng_torch': torch.get_rng_state(), 'rng_numpy': np.random.get_state(),
             'rng_python': random.getstate()}
    if denoiser is not None:
        state['denoiser'] = denoiser.state_dict()
    if torch.cuda.is_available():
        state['rng_cuda'] = torch.cuda.get_rng_state_all()
    return state


def main(cfg):
    # Validate before importing server-only robot/CLIP dependencies or touching assets.
    require_complete_config(cfg)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from .data import PartiGenDataset

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    common = dict(robot_cfg=cfg.skeleton, nfeats=57, history_len=cfg.data.history_len,
                  future_len=cfg.data.future_len, num_primitive=1, datadir=cfg.data.datadir,
                  action_statistics_path=cfg.data.action_statistics_path, max_frames=cfg.sampling.max_frames,
                  weighted_sample=False, frame_weight=False, use_weighted_meanstd=False)
    train_data = PartiGenDataset(batch_size=cfg.train.batch_size, split='train', **common)
    val_data = PartiGenDataset(batch_size=cfg.train.eval_batch_size, split='val', **common)
    vae = instantiate(cfg.vae).to(device)
    denoiser = instantiate(cfg.denoiser).to(device) if cfg.stage == 'dar' else None
    diffusion = instantiate(cfg.diffusion.model) if denoiser is not None else None
    model = vae if denoiser is None else denoiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, betas=tuple(cfg.train.betas),
                                  eps=cfg.train.eps, weight_decay=cfg.train.weight_decay)
    step = 0
    resume = cfg.ckpt.vae if cfg.stage == 'mvae' else cfg.ckpt.dar
    if resume:
        state = load_weights(model, resume, 'vae' if cfg.stage == 'mvae' else 'denoiser', device)
        if state.get('cfg', {}).get('data', {}).get('future_len') != cfg.data.future_len:
            raise ValueError("Resume checkpoint and configured primitive lengths differ")
        if cfg.stage == 'dar':
            vae.load_state_dict(state['vae'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        if 'rng_torch' in state:
            torch.set_rng_state(state['rng_torch'].cpu())
            np.random.set_state(state['rng_numpy'])
            random.setstate(state['rng_python'])
            if torch.cuda.is_available() and 'rng_cuda' in state:
                torch.cuda.set_rng_state_all([value.cpu() for value in state['rng_cuda']])
    elif cfg.stage == 'dar':
        if not cfg.ckpt.vae:
            raise ValueError("Stage 2 requires ckpt.vae pointing to a matching BP-MVAE checkpoint")
        state = load_weights(vae, cfg.ckpt.vae, 'vae', device)
        if state.get('cfg', {}).get('data', {}).get('future_len') != cfg.data.future_len:
            raise ValueError("BP-MVAE and DA-LDM must use the same primitive length")
    if cfg.stage == 'dar':
        vae.requires_grad_(False).eval()
    output = Path(cfg.experiment_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / 'cfg.yaml', resolve=True)
    writer = SummaryWriter(str(output))
    max_steps = sum(cfg.train.stages)
    progress = tqdm(total=max_steps, initial=step)
    # Exactly one batch and one optimizer update per step; no inner-loop overshoot.
    def batch_on_device(dataset):
        return {k: value.to(device) for k, value in dataset._generate_batch_optimized().items()}
    try:
        while step < max_steps:
            model.train()
            lr = learning_rate_at(step, max_steps, cfg.train.lr, cfg.train.lr_schedule)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            terms = compute_objective(cfg, vae, denoiser, diffusion, train_data, batch_on_device(train_data), step)
            if not torch.isfinite(terms['total']):
                raise FloatingPointError(f"Non-finite objective at step {step}")
            terms['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            progress.update(1)
            for name, value in terms.items():
                writer.add_scalar('train/' + name, value.detach().item(), step)
            writer.add_scalar('train/lr', lr, step)
            if step % cfg.train.eval_every == 0 or step == max_steps:
                model.eval()
                values = {}
                with torch.no_grad():
                    for _ in range(cfg.train.eval_steps):
                        losses = compute_objective(cfg, vae, denoiser, diffusion, val_data, batch_on_device(val_data), step)
                        for name, value in losses.items():
                            values[name] = values.get(name, 0.0) + value.item() / cfg.train.eval_steps
                for name, value in values.items():
                    writer.add_scalar('val/' + name, value, step)
            if step % cfg.train.save_every == 0 or step == max_steps:
                # Atomic replacement avoids leaving an apparently valid partial checkpoint.
                target = output / f'ckpt_{step}.pth'
                temporary = target.with_suffix('.pth.tmp')
                torch.save(checkpoint_state(cfg, vae, denoiser, optimizer, step), temporary)
                temporary.replace(target)
    finally:
        progress.close()
        writer.close()
