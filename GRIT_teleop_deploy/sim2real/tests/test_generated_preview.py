"""Simulation preview routing must never expand hardware authorization."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import generated_preview
import run_generated


class GeneratedPreviewTest(unittest.TestCase):
    def test_new_generated_hardware_rejected_before_preparation(self):
        with patch.object(generated_preview, "prepare") as prepare:
            with self.assertRaisesRegex(ValueError, "真机入口"):
                run_generated.build_commands(1183, hardware=True)
            prepare.assert_not_called()

    def test_baseline_hardware_rejected(self):
        with self.assertRaisesRegex(ValueError, "仅支持仿真"):
            run_generated.build_commands(222, hardware=True, baseline=True)

    def test_indices_rejected(self):
        for index in (-1, 2576):
            with self.subTest(index=index), self.assertRaises(ValueError):
                run_generated.build_commands(index, hardware=False)

    def test_dry_run_preserves_source_and_creates_nothing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "source/sample_1183"
            sample.mkdir(parents=True)
            qpos = np.zeros((4, 30))
            qpos[:, 2] = 0.78
            qpos[:, 6] = 1.0
            np.save(sample / "qpos.npy", qpos)
            (sample / "prompt.txt").write_text(
                "sample_idx: 1183\nsrc_idx: 2197\nfps: 30\ngen_frames: 4\nprompt: waves\n")
            previews = root / "previews"
            with patch.object(generated_preview, "SOURCE", root / "source"), \
                    patch.object(generated_preview, "PREVIEWS", previews), \
                    patch.object(generated_preview.subprocess, "run") as convert:
                result, sim, deploy = run_generated.build_commands(1183, False, dry_run=True)
            convert.assert_not_called()
            self.assertFalse(previews.exists())
            self.assertEqual(result["source_file"], str(sample / "qpos.npy"))
            self.assertEqual(result["reference_frames"], 6)
            self.assertEqual(result["hardware_screen"], "SIM_ONLY")
            self.assertEqual(result["trimmed_frames"], 0)
            self.assertIn("--auto-start", sim)
            self.assertIn("--max-policy-steps", deploy)
            np.testing.assert_array_equal(np.load(sample / "qpos.npy"), qpos)


if __name__ == "__main__":
    unittest.main()
