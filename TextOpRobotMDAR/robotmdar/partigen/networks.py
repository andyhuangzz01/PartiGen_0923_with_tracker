"""BP-MVAE and clean-latent DA-LDM from main.pdf, Appendix S2.

The paper leaves non-frame tokens, decoder raw-feature inputs and EOS inputs
unspecified. The concrete conventions here are recorded in README.md.
"""
import math

import torch
from torch import nn


BODY_PARTS = ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11),
              (12, 13, 14), (15, 16, 17, 18, 19, 20, 21, 22))


class RawFeatureAttention(nn.Module):
    """Gate joint angles BEFORE four independent 57 -> head-width Q/K/Vs."""

    def __init__(self, h_dim, dropout, body_parts=BODY_PARTS):
        super().__init__()
        if h_dim % 4 or len(body_parts) != 4:
            raise ValueError("Four body parts and a hidden width divisible by four are required")
        if sorted(j for part in body_parts for j in part) != list(range(23)):
            raise ValueError("Body parts must partition the 23 joint-angle indices exactly once")
        masks = torch.ones(4, 57)
        masks[:, 11:34] = 0
        for k, part in enumerate(body_parts):
            masks[k, [11 + j for j in part]] = 1
        self.register_buffer("feature_masks", masks)
        self.head_dim = h_dim // 4
        self.q = nn.ModuleList(nn.Linear(57, self.head_dim) for _ in range(4))
        self.k = nn.ModuleList(nn.Linear(57, self.head_dim) for _ in range(4))
        self.v = nn.ModuleList(nn.Linear(57, self.head_dim) for _ in range(4))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(h_dim, h_dim)

    def project(self, raw):
        if raw.shape[-1] != 57:
            raise ValueError("Part masks apply to raw 57-D frames, not hidden channels")
        masked = raw.unsqueeze(1) * self.feature_masks[None, :, None, :]
        return tuple(torch.stack([layers[i](masked[:, i]) for i in range(4)], dim=1)
                     for layers in (self.q, self.k, self.v))

    def forward(self, raw, padding_mask=None):
        q, k, v = self.project(raw)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        weights = self.dropout(scores.softmax(dim=-1))
        heads = (weights @ v).transpose(1, 2).flatten(2)
        return self.out(heads)


class PartBlock(nn.Module):
    def __init__(self, h_dim, ff_size, dropout, masked, body_parts):
        super().__init__()
        self.masked = masked
        self.attention = (RawFeatureAttention(h_dim, dropout, body_parts) if masked else
                          nn.MultiheadAttention(h_dim, 4, dropout=dropout, batch_first=True))
        self.norm1 = nn.LayerNorm(h_dim)
        self.norm2 = nn.LayerNorm(h_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(h_dim, ff_size), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ff_size, h_dim))

    def forward(self, hidden, raw, padding_mask=None):
        if self.masked:
            update = self.attention(raw, padding_mask)
        else:
            update = self.attention(hidden, hidden, hidden, key_padding_mask=padding_mask,
                                    need_weights=False)[0]
        hidden = self.norm1(hidden + self.dropout(update))
        return self.norm2(hidden + self.dropout(self.ffn(hidden)))


class PartStack(nn.Module):
    def __init__(self, h_dim, ff_size, num_layers, dropout, body_parts):
        super().__init__()
        self.layers = nn.ModuleList(PartBlock(h_dim, ff_size, dropout, i % 3 != 2, body_parts)
                                    for i in range(num_layers))
        self.norm = nn.LayerNorm(h_dim)

    def forward(self, hidden, raw, padding_mask=None):
        for layer in self.layers:
            hidden = layer(hidden, raw, padding_mask)
        return self.norm(hidden)


class BPMotionVAE(nn.Module):
    def __init__(self, *, dropout, nfeats=57, latent_dim=(1, 128), h_dim=512, ff_size=1024,
                 num_layers=9, num_heads=4, max_frames=322,
                 body_parts=BODY_PARTS):
        super().__init__()
        if nfeats != 57 or num_heads != 4 or latent_dim[0] != 1 or num_layers % 3:
            raise ValueError("BP-MVAE requires 57 features, four heads, one latent and 3n layers")
        self.latent_size, self.latent_dim = latent_dim
        self.h_dim = h_dim
        self.skel_embedding = nn.Linear(57, h_dim)
        self.global_motion_token = nn.Parameter(torch.randn(1, 2, h_dim) * 0.02)
        self.encoder_pos = nn.Parameter(torch.randn(1, max_frames + 2, h_dim) * 0.02)
        self.decoder_pos = nn.Parameter(torch.randn(1, max_frames + 1, h_dim) * 0.02)
        self.encoder = PartStack(h_dim, ff_size, num_layers, dropout, body_parts)
        self.decoder = PartStack(h_dim, ff_size, num_layers, dropout, body_parts)
        self.encoder_latent_proj = nn.Linear(h_dim, self.latent_dim)
        self.decoder_latent_proj = nn.Linear(self.latent_dim, h_dim)
        self.final_layer = nn.Linear(h_dim, 57)
        self.register_buffer("latent_mean", torch.tensor(0.0))
        self.register_buffer("latent_std", torch.tensor(1.0))

    def encode(self, future_motion, history_motion, scale_latent=False, valid_mask=None):
        frames = torch.cat((history_motion, future_motion), dim=1)
        batch = len(frames)
        hidden = torch.cat((self.global_motion_token.expand(batch, -1, -1),
                            self.skel_embedding(frames)), dim=1)
        # Distribution tokens have zero raw features, but learned hidden residuals.
        raw = torch.cat((frames.new_zeros(batch, 2, 57), frames), dim=1)
        padding = None
        if valid_mask is not None:
            prefix = torch.zeros(batch, 2 + history_motion.shape[1], dtype=torch.bool, device=frames.device)
            padding = torch.cat((prefix, ~valid_mask.bool()), dim=1)
        hidden = hidden + self.encoder_pos[:, :hidden.shape[1]]
        stats = self.encoder_latent_proj(self.encoder(hidden, raw, padding)[:, :2])
        mu, logvar = stats[:, 0].unsqueeze(0), stats[:, 1].unsqueeze(0).clamp(-10, 10)
        dist = torch.distributions.Normal(mu, (0.5 * logvar).exp())
        latent = dist.rsample()
        return (latent / self.latent_std if scale_latent else latent), dist

    def decode(self, z, history_motion, nfuture=8, scale_latent=False):
        if nfuture <= 0:
            raise ValueError("nfuture must be positive")
        if scale_latent:
            z = z * self.latent_std
        batch = len(history_motion)
        latent_token = self.decoder_latent_proj(z.transpose(0, 1))
        hidden = torch.cat((latent_token, self.skel_embedding(history_motion),
                            history_motion.new_zeros(batch, nfuture, self.h_dim)), dim=1)
        # Future frames are unknown: zero raw placeholders, never target leakage.
        raw = torch.cat((history_motion.new_zeros(batch, 1, 57), history_motion,
                         history_motion.new_zeros(batch, nfuture, 57)), dim=1)
        hidden = hidden + self.decoder_pos[:, :hidden.shape[1]]
        return self.final_layer(self.decoder(hidden, raw)[:, -nfuture:])

    def forward(self, z, history_motion, nfuture=8):
        return self.decode(z, history_motion, nfuture)


class FrameEOSHead(nn.Module):
    """Four linear layers; three GELUs; last hidden width 128 (Appendix S2.5)."""
    def __init__(self, latent_dim, clip_dim, hidden_dims):
        super().__init__()
        if len(hidden_dims) != 3 or hidden_dims[-1] != 128:
            raise ValueError("EOS requires three hidden layers ending in width 128")
        widths = [57 + latent_dim + clip_dim + 1, *hidden_dims, 1]
        layers = []
        for i in range(4):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < 3:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, frames, latent, text, frame_indices, max_frames=320):
        count = frames.shape[1]
        features = torch.cat((frames, latent[:, :1].expand(-1, count, -1),
                              text[:, None].expand(-1, count, -1),
                              frame_indices.to(frames).unsqueeze(-1) / max_frames), dim=-1)
        return self.mlp(features).squeeze(-1)


class DurationAdaptiveDenoiser(nn.Module):
    """Shared x0 prediction, retained history cross-attention and a frame EOS head."""
    def __init__(self, *, dropout, cond_mask_prob, num_timesteps, eos_hidden_dims,
                 h_dim=512, ff_size=1024, num_layers=8, num_heads=4,
                 clip_dim=512, history_shape=(2, 57), noise_shape=(1, 128)):
        super().__init__()
        self.noise_shape = tuple(noise_shape)
        self.history_shape = tuple(history_shape)
        self.cond_mask_prob = cond_mask_prob
        self.embed_noise = nn.Linear(noise_shape[-1], h_dim)
        self.embed_text = nn.Linear(clip_dim, h_dim)
        self.embed_history = nn.Linear(history_shape[-1], h_dim)
        self.history_pos = nn.Parameter(torch.randn(1, history_shape[0], h_dim) * 0.02)
        self.embed_timestep = nn.Embedding(num_timesteps, h_dim)
        layer = nn.TransformerDecoderLayer(h_dim, num_heads, ff_size, dropout,
                                           activation="gelu", batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(h_dim))
        self.output_process = nn.Linear(h_dim, noise_shape[-1])
        self.eos_head = FrameEOSHead(noise_shape[-1], clip_dim, eos_hidden_dims)

    def mask_text(self, text, force_mask=False):
        if force_mask:
            return torch.zeros_like(text)
        if self.training and self.cond_mask_prob:
            keep = torch.rand(len(text), 1, device=text.device) >= self.cond_mask_prob
            return text * keep
        return text

    def forward(self, x_t, timesteps, y=None, return_condition=False):
        text = self.mask_text(y["text_embedding"], y.get("uncond", False))
        hidden = self.embed_noise(x_t) + self.embed_timestep(timesteps.long())[:, None]
        hidden = hidden + self.embed_text(text)[:, None]
        history = y["history_motion_normalized"]
        memory = self.embed_history(history) + self.history_pos[:, :history.shape[1]]
        prediction = self.output_process(self.transformer(hidden, memory))
        # Both training heads consume the SAME sampled condition. Returning it
        # avoids sampling a second CFG mask or leaking full text into EOS.
        return (prediction, text) if return_condition else prediction

    def predict_eos(self, frames, latent, text, frame_indices, max_frames=320):
        return self.eos_head(frames, latent, text, frame_indices, max_frames)


class GuidedDenoiser(nn.Module):
    """CFG acts on clean latents; both branches retain identical motion history."""
    def __init__(self, model, scale=5.0):
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, x_t, timesteps, y=None):
        text = self.model(x_t, timesteps, y={**y, "uncond": False})
        zero = self.model(x_t, timesteps, y={**y, "uncond": True})
        return zero + self.scale * (text - zero)
