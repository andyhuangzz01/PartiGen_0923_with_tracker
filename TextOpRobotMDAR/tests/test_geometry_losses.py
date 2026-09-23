"""Regression tests for the paper's root-only orientation increment loss."""
import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from robotmdar.partigen.losses import geometry_terms


def yaw_quaternion(angle):
    zero = torch.zeros_like(angle)
    return torch.stack((zero, zero, (angle / 2).sin(), (angle / 2).cos()), dim=-1)


class YawTreeDataset:
    """Minimal differentiable FK: each child has a yaw joint on the root."""

    def __init__(self, bodies):
        self.bodies = bodies
        self.skeleton = SimpleNamespace(foot_id=[0, 1])

    def denormalize(self, motion):
        return motion

    def reconstruct_motion(self, motion, **kwargs):
        root = yaw_quaternion(motion[..., 0])
        child = yaw_quaternion(motion[..., 0] + motion[..., 1])
        rotation = torch.cat((root[:, :, None],
                              child[:, :, None].expand(-1, -1, self.bodies - 1, -1)), dim=2)
        position = motion[..., 7:10, None].transpose(-1, -2).expand(-1, -1, self.bodies, -1)
        joint = motion[..., 1:2]
        velocity = torch.cat((joint[:, 1:] - joint[:, :-1], joint[:, -1:] * 0), dim=1) * 30
        return {'global_translation_extend': position, 'global_rotation': rotation,
                'dof_pos': joint, 'dof_vel': velocity,
                'contact_mask': motion.new_ones(*motion.shape[:2], 2)}


class RootOrientationDeltaTests(unittest.TestCase):
    def terms(self, pred, bodies=2, valid=None):
        if valid is None:
            valid = torch.ones(pred.shape[:2], dtype=torch.bool)
        return geometry_terms(YawTreeDataset(bodies), pred, torch.zeros_like(pred),
                              pred.new_zeros(pred.shape[0], 2, pred.shape[2]), valid, temporal=True)

    def test_child_rotation_does_not_penalize_stationary_root(self):
        pred = torch.zeros(1, 3, 57)
        pred[0, :, 1] = torch.tensor([.2, .6, 1.])
        terms = self.terms(pred)
        self.assertGreater(terms['body_rot'].item(), 0)
        self.assertEqual(terms['orient_delta'].item(), 0)

    def test_root_rotation_is_not_diluted_by_body_count(self):
        pred = torch.zeros(1, 3, 57)
        pred[0, :, 0] = torch.tensor([.2, .6, 1.])
        # Child joints counter-rotate, keeping their global orientations fixed.
        pred[0, :, 1] = -pred[0, :, 0]
        root = yaw_quaternion(torch.tensor([[0., .2, .6, 1.]]))
        delta = root[:, 1:] - root[:, :-1]
        expected = F.huber_loss(delta, torch.zeros_like(delta), delta=1.)
        self.assertGreater(expected.item(), 0)
        for bodies in (2, 17):
            with self.subTest(bodies=bodies):
                torch.testing.assert_close(self.terms(pred, bodies)['orient_delta'], expected)

    def test_padding_receives_no_orientation_gradient(self):
        pred = torch.zeros(1, 4, 57)
        pred[0, :, 0] = torch.tensor([.2, .6, 123., -456.])
        pred.requires_grad_()
        valid = torch.tensor([[True, True, False, False]])
        loss = self.terms(pred, valid=valid)['orient_delta']
        unpadded_loss = self.terms(pred[:, :2])['orient_delta']
        torch.testing.assert_close(loss, unpadded_loss)
        loss.backward()
        self.assertGreater(pred.grad[0, :2, 0].abs().sum().item(), 0)
        self.assertEqual(pred.grad[0, 2:].abs().sum().item(), 0)
        self.assertEqual(pred.grad[0, :, 1:].abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
