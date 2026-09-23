import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from sim2sim_diagnostics import DiagnosticSim
import sim2sim


class DiagnosticTest(unittest.TestCase):
    def test_observation_is_copied_and_once_per_physics_step(self):
        obj = DiagnosticSim.__new__(DiagnosticSim)
        obj._diagnostic_rows = []
        obj._diagnostic_last_count = 0
        obj._control_sample_count = 1
        obj.low_level_dt = .005
        obj._diagnostic_reference = dict(evaluation_phase="motion", motion_frame=453)
        obj._reference_q_policy = np.zeros(29)
        obj._policy_to_mujoco = lambda q: q
        qpos = np.zeros(36); qpos[2] = .78; qpos[3] = 1
        qpos[7] = .62
        obj.data = SimpleNamespace(qpos=qpos, qvel=np.zeros(35), time=1.2)
        with patch.object(sim2sim.Sim2Sim, "_viewer_sync", return_value=True):
            obj._viewer_sync(); obj._viewer_sync()
        self.assertEqual(len(obj._diagnostic_rows), 1)
        self.assertEqual(obj._diagnostic_rows[0][2:4], ("motion", 453))
        obj.data.qpos[7] = 0
        self.assertEqual(obj._diagnostic_rows[0][4][7], .62)

    def test_peak_joint_and_phase_are_saved(self):
        with TemporaryDirectory() as tmp:
            obj = DiagnosticSim.__new__(DiagnosticSim)
            obj._summary_written = False
            obj.metrics_out = Path(tmp)/"metrics.json"
            obj.mujoco_joint_names = [f"joint_{i}" for i in range(29)]
            q = np.zeros(36); q[3] = 1; q[10] = -.62
            obj._diagnostic_rows = [(2., 2., "motion", 42, q, np.zeros(35), np.zeros(29), 20.5)]
            def summary(instance):
                instance.metrics_out.write_text(json.dumps(dict(reference_joint_error_max_rad=.62, root_tilt_max_deg=20.5)))
                instance._summary_written = True
            with patch.object(sim2sim.Sim2Sim, "_write_summary", summary):
                obj._write_summary()
            result = json.loads((Path(tmp)/"diagnostics.json").read_text())
            self.assertTrue(result["matches_aggregate_peaks"])
            self.assertEqual(result["peak_joint_error"]["joint"], "joint_3")
            self.assertEqual(result["peak_joint_error"]["motion_frame"], 42)


if __name__ == "__main__":
    unittest.main()
