import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gt_slow
import run_generated


class GTSlowTest(unittest.TestCase):
    def test_factor_meets_speed_target(self):
        for speed in (0, 3.5, 11.596, 55.426, 28.016):
            factor = gt_slow.slowdown_factor(speed)
            self.assertGreaterEqual(factor, 2)
            self.assertLessEqual(speed / factor, gt_slow.TARGET_SPEED)
        for speed in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                gt_slow.slowdown_factor(speed)

    def test_stretch_preserves_knots_and_scales_velocity(self):
        pos = np.array([[0, 0, .75], [.02, 0, .76], [.04, 0, .77]])
        quat = np.tile([1., 0, 0, 0], (3, 1))
        joints = np.tile(np.arange(3)[:, None] * .02, (1, 29))
        times, p, q, j = gt_slow.stretch(pos, quat, joints, 50, 4, 50)
        self.assertEqual(len(times), 9)
        np.testing.assert_allclose(p[::4], pos)
        np.testing.assert_allclose(q[::4], quat)
        np.testing.assert_allclose(j[::4], joints)
        self.assertAlmostEqual(float(np.abs(np.diff(j, axis=0)).max()*50), .25)
        self.assertAlmostEqual(times[-1], 4 * 2/50)

    def test_slow_not_allowed_for_other_sources(self):
        with self.assertRaises(ValueError):
            run_generated.build_commands(1183, False, slow=True)

    def test_slow_hardware_uses_slow_artifact_and_manual_start(self):
        with TemporaryDirectory() as temp:
            p = Path(temp)/"motion.npz"
            p.touch()
            result = dict(index=5225, error=None, rating="PASS", hardware_screen="CANDIDATE_TIER2",
                          tracking_motion_frames_complete=True, npz_file=str(p))
            with patch.object(gt_slow, "get_result", return_value=result):
                _, sim, deploy = run_generated.build_commands(5225, True, gt=True, slow=True)
            self.assertIsNone(sim)
            self.assertIn(str(p), deploy)
            self.assertNotIn("--auto-start", deploy)
            self.assertNotIn("--publish-reference", deploy)

    def test_slow_review_stays_blocked(self):
        for rating, screen in (("PASS", "REVIEW"), ("FALL", "NOT_RECOMMENDED"), ("ERROR", "NOT_RECOMMENDED")):
            result = dict(index=5225, error=None, rating=rating, hardware_screen=screen,
                          tracking_motion_frames_complete=True, reasons=[])
            with patch.object(gt_slow, "get_result", return_value=result), self.assertRaises(ValueError):
                run_generated.build_commands(5225, True, gt=True, slow=True)


if __name__ == "__main__":
    unittest.main()
