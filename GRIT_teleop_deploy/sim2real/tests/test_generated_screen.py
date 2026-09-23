import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from screen_generated_motions import LIMITS, classify, extract


class GeneratedScreenTests(unittest.TestCase):
    def record(self):
        q = np.zeros((4, 30))
        q[:, 2] = 0.75
        q[:, 6] = 1  # xyzw identity
        return {"qpos": q, "fps": 30}

    def passing(self):
        r = {k: v * 0.5 for k, v in LIMITS.items()}
        r.update(root_z_min=0.75, source_xy_path_m=0.0,
                 source_leg_excursion_rad=0.1, text="waves one hand")
        return r

    def test_xyzw_conversion_and_no_source_mutation(self):
        record = self.record()
        q, fps = extract(record, 2, "xyzw")
        self.assertEqual(q.shape, (2, 30))
        np.testing.assert_array_equal(q[:, 3:7], [[1, 0, 0, 0]] * 2)
        self.assertEqual(record["qpos"][0, 6], 1)
        self.assertEqual(fps, 30)

    def test_invalid_inputs(self):
        for kind in ("short", "nan", "quat", "fps"):
            r = self.record()
            if kind == "short":
                r["qpos"] = r["qpos"][:1]
            elif kind == "nan":
                r["qpos"][0, 7] = np.nan
            elif kind == "quat":
                r["qpos"][:, 3:7] = 0
            else:
                r["fps"] = 0
            with self.assertRaises(ValueError):
                extract(r, 0, "xyzw")

    def test_error_never_passes(self):
        r = self.passing()
        r["error"] = "timeout"
        self.assertEqual(classify(r)[:2], ("ERROR", "NOT_RECOMMENDED"))

    def test_missing_or_nan_metric_never_passes(self):
        for value in (None, float("nan")):
            r = self.passing()
            r["reference_joint_rmse_rad"] = value
            self.assertEqual(classify(r)[0], "ERROR")

    def test_fall_and_marginal(self):
        r = self.passing()
        r["root_z_min"] = 0.4
        self.assertEqual(classify(r)[0], "FALL")
        r["root_z_min"] = 0.55
        self.assertEqual(classify(r)[:2], ("MARGINAL", "NOT_RECOMMENDED"))

    def test_sim_pass_not_automatic_hardware_candidate(self):
        r = self.passing()
        r["reference_lower_body_rmse_rad"] = 0.3
        self.assertEqual(classify(r)[:2], ("PASS", "REVIEW"))

    def test_candidate_tiers_and_description_veto(self):
        r = self.passing()
        self.assertEqual(classify(r)[1], "CANDIDATE_TIER1")
        r["source_xy_path_m"] = 0.5
        self.assertEqual(classify(r)[1], "CANDIDATE_TIER2")
        r["text"] = "a person is jumping"
        self.assertEqual(classify(r)[1], "REVIEW")


if __name__ == "__main__":
    unittest.main()
