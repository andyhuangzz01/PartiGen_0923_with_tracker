import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.tracking_metrics import BODY_POINT_NAMES, compute_mpjpe, resolve_body_points


class TrackingMetricTests(unittest.TestCase):
    def test_identical(self):
        reference = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3) / 10
        result = compute_mpjpe(reference, reference, reference[:, 0], reference[:, 0])
        self.assertAlmostEqual(result["global_mpjpe_mm"], 0.0)
        self.assertAlmostEqual(result["root_relative_mpjpe_mm"], 0.0)

    def test_global_translation(self):
        reference = np.zeros((3, 5, 3))
        shift = np.array([0.1, 0.0, 0.0])
        actual = reference + shift
        result = compute_mpjpe(actual, reference, actual[:, 0], reference[:, 0])
        self.assertAlmostEqual(result["global_mpjpe_mm"], 100.0)
        self.assertAlmostEqual(result["root_relative_mpjpe_mm"], 0.0)

    def test_one_non_root_point_offset(self):
        k = 5
        reference = np.zeros((1, k, 3))
        actual = reference.copy()
        actual[0, 2, 0] = 0.1
        result = compute_mpjpe(actual, reference, actual[:, 0], reference[:, 0])
        self.assertAlmostEqual(result["global_mpjpe_mm"], 100.0 / k)
        self.assertAlmostEqual(result["root_relative_mpjpe_mm"], 100.0 / k)

    def test_root_yaw_is_not_removed(self):
        reference = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
        actual = np.array([[[0, 0, 0], [0, 1, 0], [-1, 0, 0]]], dtype=float)
        result = compute_mpjpe(actual, reference, actual[:, 0], reference[:, 0])
        self.assertGreater(result["root_relative_mpjpe_mm"], 900.0)

    def test_time_shift_is_visible(self):
        t = np.arange(5, dtype=float)[:, None, None]
        reference = np.broadcast_to(t * 0.1, (5, 3, 3)).copy()
        aligned = compute_mpjpe(reference, reference, reference[:, 0], reference[:, 0])
        shifted = compute_mpjpe(reference[1:], reference[:-1], reference[1:, 0], reference[:-1, 0])
        self.assertAlmostEqual(aligned["global_mpjpe_mm"], 0.0)
        self.assertGreater(shifted["global_mpjpe_mm"], 100.0)

    def test_fk_actual_and_independent_reference_paths_match(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "config/g1/assets/g1.xml"))
        ids, _ = resolve_body_points(model)
        self.assertEqual(len(ids), len(BODY_POINT_NAMES))
        actual = mujoco.MjData(model)
        reference = mujoco.MjData(model)
        qpos = model.qpos0.copy()
        qpos[:3] = [0.2, -0.1, 0.81]
        qpos[3:7] = [np.cos(0.2), 0.0, 0.0, np.sin(0.2)]  # MuJoCo wxyz
        qpos[7:] = np.linspace(-0.2, 0.2, model.nq - 7)
        actual.qpos[:] = qpos
        reference.qpos[:] = qpos
        mujoco.mj_forward(model, actual)
        mujoco.mj_forward(model, reference)
        np.testing.assert_allclose(actual.xpos[ids], reference.xpos[ids], atol=1e-12)
        np.testing.assert_allclose(actual.xpos[ids[0]], actual.qpos[:3], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
