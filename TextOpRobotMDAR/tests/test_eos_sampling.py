"""Regression tests for semantic EOS and physical look-ahead boundaries."""
import unittest

import torch

from robotmdar.partigen.eos import eos_loss, primitive_targets


class EOSSamplingTests(unittest.TestCase):
    @staticmethod
    def sampled_increments(indices, sequence_len, history_len=2):
        # A physical motion with q[t] = t exposes synthetic endpoint holds.
        # Table S3's joint increment is q[t + 1] - q[t].
        q = torch.arange(sequence_len, dtype=torch.float32)[indices]
        return (q[1:] - q[:-1])[history_len:]

    def test_annotation_endpoint_preserves_real_next_state(self):
        indices, valid, labels, positions = primitive_targets(
            10, 18, 0, 8, sequence_len=24)
        self.assertEqual(indices.tolist(), list(range(8, 19)))
        self.assertTrue(valid.all())
        self.assertEqual(labels.tolist(), [0] * 7 + [1])
        self.assertEqual(positions.tolist(), list(range(1, 9)))
        torch.testing.assert_close(self.sampled_increments(indices, 24), torch.ones(8))

    def test_physical_endpoint_holds_last_state(self):
        indices, valid, labels, _ = primitive_targets(
            10, 18, 0, 8, sequence_len=18)
        self.assertEqual(indices[-2:].tolist(), [17, 17])
        self.assertTrue(valid.all())
        self.assertEqual(labels.tolist(), [0] * 7 + [1])
        torch.testing.assert_close(self.sampled_increments(indices, 18),
                                   torch.tensor([1.] * 7 + [0.]))

    def test_short_annotation_masks_padding_but_keeps_endpoint_increment(self):
        indices, valid, labels, _ = primitive_targets(
            10, 11, 0, 8, sequence_len=24)
        self.assertEqual(valid.tolist(), [True] + [False] * 7)
        self.assertEqual(labels.tolist(), [1] + [0] * 7)
        self.assertEqual(self.sampled_increments(indices, 24)[0].item(), 1.)
        # The following real frames are available to construct increments, but
        # are not extra EOS training targets for this annotation.
        logits = torch.zeros(1, 8, requires_grad=True)
        eos_loss(logits, labels[None], valid[None], positive_weight=1).backward()
        self.assertNotEqual(logits.grad[0, 0].item(), 0.)
        self.assertEqual(logits.grad[0, 1:].abs().sum().item(), 0.)

    def test_single_frame_physical_sequence_is_usable(self):
        indices, valid, labels, _ = primitive_targets(
            0, 1, 0, 8, sequence_len=1)
        self.assertTrue((indices == 0).all())
        self.assertEqual(valid.tolist(), [True] + [False] * 7)
        self.assertEqual(labels.tolist(), [1] + [0] * 7)
        self.assertEqual(self.sampled_increments(indices, 1).abs().sum().item(), 0.)

    def test_horizon_preserves_lookahead_without_synthetic_eos(self):
        indices, valid, labels, _ = primitive_targets(
            0, 400, 39, 8, sequence_len=420)
        self.assertTrue(valid.all())
        self.assertEqual(labels.sum().item(), 0.)
        self.assertEqual(indices[-1].item(), 320)
        self.assertEqual(self.sampled_increments(indices, 420)[-1].item(), 1.)
        # An actual annotation endpoint at the horizon remains a positive EOS.
        _, _, labels, _ = primitive_targets(0, 320, 39, 8, sequence_len=420)
        self.assertEqual(labels.tolist(), [0] * 7 + [1])

    def test_lookahead_length_does_not_change_annotation_labels(self):
        terminal = primitive_targets(0, 11, 1, 8, sequence_len=11)
        continuing = primitive_targets(0, 11, 1, 8, sequence_len=32)
        for terminal_value, continuing_value in zip(terminal[1:], continuing[1:]):
            torch.testing.assert_close(terminal_value, continuing_value)
        self.assertEqual(continuing[1].tolist(), [True] * 3 + [False] * 5)
        self.assertEqual(continuing[2].tolist(), [0, 0, 1, 0, 0, 0, 0, 0])

    def test_annotation_must_fit_physical_sequence(self):
        for start, end, sequence_len in [(10, 21, 20), (-1, 10, 20), (0, 1, 0)]:
            with self.subTest(start=start, end=end, sequence_len=sequence_len):
                with self.assertRaisesRegex(ValueError, 'physical sequence'):
                    primitive_targets(start, end, 0, 8, sequence_len=sequence_len)


if __name__ == '__main__':
    unittest.main()
