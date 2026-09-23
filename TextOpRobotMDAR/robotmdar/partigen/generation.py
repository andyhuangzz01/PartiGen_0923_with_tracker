"""Complete candidate generation followed by frame-level EOS selection."""
from pathlib import Path

import torch

from .eos import endpoint_lengths
from .networks import GuidedDenoiser


@torch.no_grad()
def generate_candidate(vae, denoiser, diffusion, history, text, *, future_len,
                       max_frames=320, guidance_scale=5.0, eos_threshold=0.9):
    if future_len not in (8, 16) or max_frames <= 0 or max_frames % future_len:
        raise ValueError("Use 8 or 16 new frames per primitive and a divisible positive horizon")
    vae.eval()
    denoiser.eval()
    guided = GuidedDenoiser(denoiser, guidance_scale).eval()
    frames, latents = [], []
    history_len = history.shape[1]
    for _ in range(max_frames // future_len):
        latent = diffusion.p_sample_loop(
            guided, (len(history), *denoiser.noise_shape), device=history.device,
            clip_denoised=False,
            model_kwargs={'y': {'text_embedding': text, 'history_motion_normalized': history}},
            progress=False)
        primitive = vae.decode(latent.transpose(0, 1), history, future_len)
        frames.append(primitive)
        latents.append(latent)
        history = primitive[:, -history_len:]
    candidate = torch.cat(frames, dim=1)
    # Deliberately defer all EOS evaluations until the complete candidate exists.
    probabilities = []
    for index, (primitive, latent) in enumerate(zip(frames, latents)):
        positions = torch.arange(index * future_len + 1, (index + 1) * future_len + 1,
                                 device=candidate.device)[None].expand(len(candidate), -1)
        logits = denoiser.predict_eos(primitive, latent, text, positions, max_frames)
        probabilities.append(logits.sigmoid())
    probabilities = torch.cat(probabilities, dim=1)
    lengths = endpoint_lengths(probabilities, eos_threshold)
    return {'candidate': candidate, 'eos_probabilities': probabilities, 'lengths': lengths,
            'motions': [motion[:length.item()] for motion, length in zip(candidate, lengths)]}


def main(cfg):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from .training import load_weights, missing_fields
    missing = missing_fields(cfg)
    if missing:
        raise ValueError('Missing generation inputs: ' + ', '.join(missing))
    state = torch.load(cfg.checkpoint, map_location=cfg.device, weights_only=False)
    saved = OmegaConf.create(state['cfg'])
    vae = instantiate(saved.vae).to(cfg.device)
    denoiser = instantiate(saved.denoiser).to(cfg.device)
    load_weights(vae, cfg.checkpoint, 'vae', cfg.device)
    load_weights(denoiser, cfg.checkpoint, 'denoiser', cfg.device)
    diffusion = instantiate(saved.diffusion.model)
    # Inputs use the server's training mean/std and CLIP ViT-B/32 cache.
    inputs = torch.load(cfg.inputs, map_location=cfg.device, weights_only=False)
    result = generate_candidate(vae, denoiser, diffusion, inputs['history'], inputs['text'],
                                future_len=saved.data.future_len, max_frames=cfg.max_frames,
                                guidance_scale=cfg.guidance_scale, eos_threshold=cfg.eos_threshold)
    result = {key: [v.cpu() for v in value] if isinstance(value, list) else value.cpu()
              for key, value in result.items()}
    target = Path(cfg.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, target)
    print(f'Saved normalized motion features, endpoints and full candidates to {target}')
