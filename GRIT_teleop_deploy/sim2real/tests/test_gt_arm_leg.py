import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gt_arm_leg as gt
import run_generated as launcher


class GTArmLegTest(unittest.TestCase):
    def test_gt_id_is_not_generated_index(self):
        self.assertEqual(gt.gt_id(5225), "005225")
        with self.assertRaises(ValueError):
            gt.gt_id(1183)

    def test_sources_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            launcher.build_commands(5225, False, baseline=True, gt=True)

    def test_review_never_starts_hardware(self):
        result = dict(index=5225, error=None, rating="PASS", hardware_screen="REVIEW",
                      tracking_motion_frames_complete=True, reasons=["speed"])
        with patch.object(gt, "get_result", return_value=result):
            with self.assertRaisesRegex(ValueError, "未进入"):
                launcher.build_commands(5225, True, gt=True)

    def test_candidate_hardware_is_manual_and_uses_gt_npz(self):
        with TemporaryDirectory() as temp:
            motion = Path(temp) / "motion.npz"
            motion.touch()
            result = dict(index=5225, error=None, rating="PASS", hardware_screen="CANDIDATE_TIER2",
                          tracking_motion_frames_complete=True, npz_file=str(motion))
            with patch.object(gt, "get_result", return_value=result):
                _, sim, deploy = launcher.build_commands(5225, True, gt=True)
            self.assertIsNone(sim)
            self.assertIn(str(motion), deploy)
            self.assertNotIn("--auto-start", deploy)
            self.assertNotIn("--publish-reference", deploy)

    def test_fk_validation_and_wrist_preservation(self):
        model = mujoco.MjModel.from_xml_path(str(gt.XML))
        data = mujoco.MjData(model)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, model.nbody)]
        joints = np.zeros((3, 29))
        wrist = [j - 1 for j in range(1, model.njnt)
                 if "wrist" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        joints[:, wrist] = .1
        local = []
        for row in joints:
            data.qpos[:] = np.r_[0, 0, 0, 1, 0, 0, 0, row]
            mujoco.mj_forward(model, data)
            local.append(data.xpos[1:].copy())
        record = dict(root_pos=np.tile([0, 0, .78], (3, 1)),
                      root_rot=np.tile([0, 0, 0, 1], (3, 1)), dof_pos=joints,
                      local_body_pos=np.array(local), link_body_list=names, fps=30)
        pos, quat, values, fps, error = gt.validate_source(record, model)
        self.assertLess(error, 1e-10)
        with TemporaryDirectory() as temp:
            output = Path(temp) / "motion.npz"
            gt.save_reference(output, model, pos, quat, values, fps)
            with np.load(output) as clip:
                np.testing.assert_allclose(clip["joint_pos"][:, wrist], .1)
                np.testing.assert_allclose(clip["body_pos_w"][:, 0], pos)
                self.assertTrue(all(np.isfinite(clip[k]).all() for k in clip.files))
        record["dof_pos"] = joints.copy()
        record["dof_pos"][:, 0] += .2
        with self.assertRaisesRegex(ValueError, "FK mismatch"):
            gt.validate_source(record, model)


if __name__ == "__main__":
    unittest.main()
