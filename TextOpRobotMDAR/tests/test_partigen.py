"""CPU behavior tests; run: python -m unittest discover -s tests -v."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from robotmdar.partigen.eos import endpoint_lengths, eos_loss, primitive_targets
from robotmdar.partigen.generation import generate_candidate
from robotmdar.partigen.losses import geometry_terms, learning_rate_at, masked_huber, stage_at
from robotmdar.partigen.networks import BPMotionVAE, DurationAdaptiveDenoiser, GuidedDenoiser, RawFeatureAttention
from robotmdar.partigen.training import compute_objective
from robotmdar.diffusion.gaussian_diffusion import (GaussianDiffusion, get_named_beta_schedule,
                                                   ModelMeanType, ModelVarType, LossType)


torch.set_num_threads(1)


def diffusion():
    return GaussianDiffusion(betas=get_named_beta_schedule('cosine', 5),
                             model_mean_type=ModelMeanType.START_X,
                             model_var_type=ModelVarType.FIXED_SMALL,
                             loss_type=LossType.MSE, rescale_timesteps=False)


class NetworkTests(unittest.TestCase):
    def test_mask_before_projection_and_shared_increments(self):
        model = RawFeatureAttention(32, 0)
        raw = torch.randn(2, 4, 57, requires_grad=True)
        q, _, _ = model.project(raw)
        q[:, 0].sum().backward()
        # Left-leg query cannot access right-leg/torso/arm ANGLES.
        self.assertEqual(raw.grad[..., 17:34].abs().sum().item(), 0)
        self.assertGreater(raw.grad[..., :11].abs().sum().item(), 0)
        self.assertGreater(raw.grad[..., 34:57].abs().sum().item(), 0)
        self.assertTrue(torch.equal(model.feature_masks[:, 11:34].sum(0), torch.ones(23)))

    def test_vae_mixed_stack_shape_gradients_and_padding(self):
        model = BPMotionVAE(dropout=0, h_dim=32, ff_size=64)
        self.assertEqual([layer.masked for layer in model.encoder.layers], [True, True, False] * 3)
        self.assertEqual([layer.masked for layer in model.decoder.layers], [True, True, False] * 3)
        history, future = torch.randn(2, 2, 57), torch.randn(2, 8, 57)
        valid = torch.tensor([[True] * 8, [True] * 3 + [False] * 5])
        z, dist = model.encode(future, history, valid_mask=valid)
        changed = future.clone()
        changed[1, 3:] += 100
        _, dist_changed = model.encode(changed, history, valid_mask=valid)
        torch.testing.assert_close(dist.loc, dist_changed.loc)
        self.assertEqual(z.shape, (1, 2, 128))
        pred = model.decode(z, history, 8)
        masked_huber(pred, future, valid).backward()
        self.assertGreater(model.encoder.layers[0].attention.q[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.decoder_latent_proj.weight.grad.abs().sum().item(), 0)

    def test_denoiser_retains_history_when_text_is_zeroed(self):
        model = DurationAdaptiveDenoiser(dropout=0, cond_mask_prob=1, num_timesteps=5,
                                        eos_hidden_dims=(64, 32, 128), h_dim=32, ff_size=64, num_layers=2)
        model.eval()
        noise, t = torch.randn(2, 1, 128), torch.tensor([0, 4])
        y = {'text_embedding': torch.randn(2, 512), 'history_motion_normalized': torch.randn(2, 2, 57), 'uncond': True}
        first = model(noise, t, y)
        changed = model(noise, t, {**y, 'history_motion_normalized': y['history_motion_normalized'] + 1})
        self.assertFalse(torch.allclose(first, changed))
        self.assertEqual(first.shape, noise.shape)
        guided = GuidedDenoiser(model, 5)
        cond = model(noise, t, {**y, 'uncond': False})
        torch.testing.assert_close(guided(noise, t, y), first + 5 * (cond - first))
        linear = [layer for layer in model.eos_head.mlp if isinstance(layer, nn.Linear)]
        self.assertEqual(len(linear), 4)
        self.assertEqual(linear[-1].in_features, 128)

    def test_training_returns_the_single_masked_text_condition(self):
        model = DurationAdaptiveDenoiser(dropout=0, cond_mask_prob=1, num_timesteps=5,
                                        eos_hidden_dims=(64, 32, 128), h_dim=32, ff_size=64, num_layers=2)
        model.train()
        text = torch.ones(2, 512)
        history = torch.randn(2, 2, 57)
        _, used_text = model(torch.randn(2, 1, 128), torch.tensor([0, 4]),
                             y={'text_embedding': text, 'history_motion_normalized': history},
                             return_condition=True)
        self.assertEqual(used_text.count_nonzero().item(), 0)
        model.eval()
        _, used_text = model(torch.randn(2, 1, 128), torch.tensor([0, 4]),
                             y={'text_embedding': text, 'history_motion_normalized': history},
                             return_condition=True)
        torch.testing.assert_close(used_text, text)


class EOSTests(unittest.TestCase):
    def test_terminal_frame_and_padding(self):
        raw, valid, labels, positions = primitive_targets(0, 11, 1, 8, sequence_len=11)
        self.assertEqual(valid.tolist(), [True] * 3 + [False] * 5)
        self.assertEqual(labels.tolist(), [0, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(raw[-1].item(), 10)
        self.assertEqual(positions[:3].tolist(), [9, 10, 11])
        # Single-frame actions remain usable; conditioning is separate from future.
        _, valid, labels, _ = primitive_targets(0, 1, 0, 8, sequence_len=1)
        self.assertEqual(valid.sum().item(), 1)
        self.assertEqual(labels[0].item(), 1)

    def test_horizon_is_not_a_false_endpoint(self):
        _, valid, labels, _ = primitive_targets(0, 400, 39, 8, sequence_len=400)
        self.assertTrue(valid.all())
        self.assertEqual(labels.sum().item(), 0)

    def test_weighted_bce_masks_padding_and_gradients(self):
        logits = torch.tensor([[0., 0., 100.]], requires_grad=True)
        labels = torch.tensor([[0., 1., 0.]])
        mask = torch.tensor([[True, True, False]])
        actual = eos_loss(logits, labels, mask, 0.3)
        expected = F.binary_cross_entropy_with_logits(logits[:, :2], labels[:, :2], pos_weight=torch.tensor(0.3))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(logits.grad[0, 2].item(), 0)

    def test_cutoff_strict_inclusive_fallback(self):
        probabilities = torch.tensor([[.9, .91, .99], [.1, .9, .2], [.99, .2, .2]])
        self.assertEqual(endpoint_lengths(probabilities).tolist(), [2, 3, 1])

    def test_complete_generation_precedes_eos(self):
        events = []
        class FakeDiffusion:
            def p_sample_loop(self, model, shape, **kwargs):
                events.append('sample')
                self.assert_clip = kwargs['clip_denoised']
                return torch.zeros(shape)
        class FakeVAE(nn.Module):
            def decode(self, latent, history, future_len):
                return torch.zeros(len(history), future_len, 57)
        class FakeDenoiser(nn.Module):
            noise_shape = (1, 128)
            def predict_eos(self, frames, latent, text, positions, max_frames):
                events.append('eos')
                return torch.full(frames.shape[:2], 20.)  # Immediate cutoff, but still sample all blocks.
        for block, count in [(8, 40), (16, 20)]:
            events.clear()
            result = generate_candidate(FakeVAE(), FakeDenoiser(), FakeDiffusion(),
                                        torch.zeros(1, 2, 57), torch.zeros(1, 512), future_len=block)
            self.assertEqual(events, ['sample'] * count + ['eos'] * count)
            self.assertEqual(result['candidate'].shape, (1, 320, 57))
            self.assertEqual(result['lengths'].tolist(), [1])


class TrainingTests(unittest.TestCase):
    def test_schedule_boundaries(self):
        self.assertEqual(stage_at(3, [3, 5, 6, 6]), 1)
        self.assertEqual(learning_rate_at(0, 20, 0.02, 'cosine'), .02)
        self.assertAlmostEqual(learning_rate_at(10, 20, .02, 'cosine'), .01)
        self.assertEqual(learning_rate_at(20, 20, .02, 'linear'), 0)

    def test_padded_motion_does_not_contribute(self):
        pred = torch.tensor([[[1.], [999.]]], requires_grad=True)
        loss = masked_huber(pred, torch.zeros_like(pred), torch.tensor([[True, False]]))
        self.assertEqual(loss.item(), .5)
        loss.backward()
        self.assertEqual(pred.grad[0, 1].item(), 0)

    def test_real_diffusion_updates_and_both_training_objectives(self):
        # Differentiable synthetic FK exercises training loss plumbing. Server MJCF
        # execution must still be validated with the real G1 assets.
        class ToyDataset:
            skeleton = SimpleNamespace(foot_id=[0, 1])
            def denormalize(self, value):
                return value
            def reconstruct_motion(self, motion, **kwargs):
                q = motion[..., 11:34]
                velocity = torch.cat(((q[:, 1:] - q[:, :-1]) * 30, q[:, -1:] * 0), dim=1)
                xyz = motion[..., 7:10].cumsum(1)[:, :, None].expand(-1, -1, 2, -1)
                return {'global_translation_extend': xyz, 'global_rotation': motion[..., :4, None].transpose(-1, -2).expand(-1, -1, 2, -1),
                        'dof_pos': q, 'dof_vel': velocity, 'contact_mask': torch.ones(*q.shape[:2], 2)}
        names = ['body_trans', 'body_rot', 'dof_pos', 'dof_vel', 'foot_contact', 'joints_delta',
                 'trans_delta', 'orient_delta', 'dof_delta', 'smooth', 'drift_yaw', 'drift_xy']
        # Test-only explicit values, not shipped training defaults.
        starts = type('Starts', (dict,), {'__getattr__': dict.__getitem__})({key: 0 for key in names + ['kl']})
        train = SimpleNamespace(stages=[1, 1, 1, 1], geometry_weights={key: .1 for key in names},
                                loss_start_stage=starts, rec_weight=1., kl_weight=.01, kinematic_weight=.8,
                                latent_weight=1., eos_weight=.5, eos_positive_weight=.3)
        cfg = SimpleNamespace(stage='mvae', data=SimpleNamespace(history_len=2), train=train,
                              sampling=SimpleNamespace(max_frames=320))
        batch = {'motion': torch.randn(2, 10, 57), 'text': torch.randn(2, 512),
                 'valid': torch.tensor([[True] * 8, [True] * 3 + [False] * 5]),
                 'eos': torch.zeros(2, 8), 'positions': torch.arange(1, 9)[None].expand(2, -1)}
        batch['eos'][1, 2] = 1
        vae = BPMotionVAE(dropout=0, h_dim=32, ff_size=64, num_layers=3)
        terms = compute_objective(cfg, vae, None, None, ToyDataset(), batch, 0)
        terms['total'].backward()
        self.assertTrue(torch.isfinite(terms['total']))
        cfg.stage = 'dar'
        vae.zero_grad(set_to_none=True)
        vae.requires_grad_(False).eval()
        denoiser = DurationAdaptiveDenoiser(dropout=0, cond_mask_prob=1, num_timesteps=5,
                                           eos_hidden_dims=(64, 32, 128), h_dim=32, ff_size=64, num_layers=2)
        process = diffusion()
        eos_inputs = []
        handle = denoiser.eos_head.mlp[0].register_forward_pre_hook(
            lambda module, args: eos_inputs.append(args[0].detach().clone()))
        terms = compute_objective(cfg, vae, denoiser, process, ToyDataset(), batch, 0)
        handle.remove()
        self.assertEqual(eos_inputs[0][..., 57 + 128:-1].count_nonzero().item(), 0)
        terms['total'].backward()
        self.assertTrue(all(p.grad is None for p in vae.parameters()))
        self.assertGreater(denoiser.output_process.weight.grad.abs().sum().item(), 0)
        self.assertGreater(denoiser.eos_head.mlp[-1].weight.grad.abs().sum().item(), 0)
        result = generate_candidate(vae, denoiser, process, batch['motion'][:, :2], batch['text'],
                                    future_len=8, max_frames=16)
        self.assertTrue(torch.isfinite(result['candidate']).all())
        # Neither masked reconstruction nor temporal FK losses may observe padding.
        predicted = torch.randn(2, 8, 57, requires_grad=True)
        geo = geometry_terms(ToyDataset(), predicted, batch['motion'][:, 2:], batch['motion'][:, :2],
                             batch['valid'], temporal=True)
        sum(geo.values()).backward()
        self.assertEqual(predicted.grad[1, 3:].abs().sum().item(), 0)


class ConfigTests(unittest.TestCase):
    def test_templates_require_explicit_training_values(self):
        try:
            from hydra import compose, initialize_config_dir
        except ImportError:
            self.skipTest('Hydra is required for configuration checks')
        from pathlib import Path
        from omegaconf import OmegaConf
        from robotmdar.partigen.training import missing_fields, require_complete_config
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'robotmdar/config'), version_base='1.1'):
            for name in ('train_partigen_mvae', 'train_partigen_dar'):
                cfg = compose(config_name=name)
                missing = missing_fields(cfg)
                for field in ('seed', 'train.lr', 'train.stages', 'train.batch_size', 'vae.dropout', 'train.weight_decay'):
                    self.assertIn(field, missing)
                if name == 'train_partigen_dar':
                    self.assertIn('diffusion.num_timesteps', missing)
                    self.assertTrue(OmegaConf.is_missing(cfg.diffusion, 'num_timesteps'))
                with self.assertRaisesRegex(ValueError, 'no training defaults'):
                    require_complete_config(cfg)
            cfg = compose(config_name='train_partigen_dar', overrides=['diffusion.num_timesteps=7'])
            from hydra.utils import instantiate
            process = instantiate(cfg.diffusion.model)
            self.assertNotIn('diffusion.num_timesteps', missing_fields(cfg))
            self.assertEqual(cfg.denoiser.num_timesteps, 7)
            self.assertEqual(process.num_timesteps, 7)
            self.assertEqual(process.timestep_map, list(range(7)))
            self.assertEqual(process.model_mean_type, ModelMeanType.START_X)

    def test_checkpoint_roundtrip_includes_eos_and_frozen_vae(self):
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest('OmegaConf is required for checkpoint checks')
        import tempfile
        from pathlib import Path
        from robotmdar.partigen.training import checkpoint_state, load_weights
        vae = BPMotionVAE(dropout=0, h_dim=32, ff_size=64, num_layers=3)
        model = DurationAdaptiveDenoiser(dropout=0, cond_mask_prob=0, num_timesteps=5,
                                         eos_hidden_dims=(64, 32, 128), h_dim=32, ff_size=64, num_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        cfg = OmegaConf.create({'data': {'future_len': 8}})
        state = checkpoint_state(cfg, vae, model, optimizer, 17)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pth'
            torch.save(state, path)
            saved = model.eos_head.mlp[-1].weight.detach().clone()
            with torch.no_grad():
                model.eos_head.mlp[-1].weight.zero_()
            reloaded = load_weights(model, path, 'denoiser', 'cpu')
            self.assertEqual(reloaded['step'], 17)
            self.assertIn('vae', reloaded)
            torch.testing.assert_close(saved, model.eos_head.mlp[-1].weight)


if __name__ == '__main__':
    unittest.main()
