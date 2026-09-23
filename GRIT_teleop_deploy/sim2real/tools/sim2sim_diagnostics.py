"""Observational 200 Hz diagnostics; original dynamics and metric rules unchanged."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import sim2sim


class DiagnosticSim(sim2sim.Sim2Sim):
    def __init__(self, args, config):
        self._diagnostic_rows = []
        self._diagnostic_last_count = 0
        self._diagnostic_reference = None
        super().__init__(args, config)

    def _read_command_snapshot(self):
        snapshot = super()._read_command_snapshot()
        self._diagnostic_reference = snapshot[3]
        return snapshot

    def _viewer_sync(self):
        count = self._control_sample_count
        if count > self._diagnostic_last_count:
            self._diagnostic_last_count = count
            ref = self._diagnostic_reference or {}
            reference = (self._policy_to_mujoco(self._reference_q_policy).copy()
                         if self._reference_q_policy is not None else np.full(29, np.nan))
            q = self.data.qpos.copy()
            tilt = np.degrees(np.arccos(np.clip(1-2*(q[4]**2+q[5]**2), -1, 1)))
            self._diagnostic_rows.append((float(self.data.time), count*self.low_level_dt,
                str(ref.get("evaluation_phase", "unknown")), int(ref.get("motion_frame", -1)),
                q, self.data.qvel.copy(), reference, float(tilt)))
        return super()._viewer_sync()

    def _write_summary(self):
        if self._summary_written:
            return
        super()._write_summary()
        if not self._diagnostic_rows or self.metrics_out is None:
            return
        rows = self._diagnostic_rows
        path = self.metrics_out.with_name("diagnostics.npz")
        actual = np.stack([r[4] for r in rows])
        reference = np.stack([r[6] for r in rows])
        errors = actual[:, 7:] - reference
        tilts = np.array([r[7] for r in rows])
        np.savez_compressed(path, joint_names=np.array(self.mujoco_joint_names),
            sim_time_s=np.array([r[0] for r in rows]), control_time_s=np.array([r[1] for r in rows]),
            phase=np.array([r[2] for r in rows]), motion_frame=np.array([r[3] for r in rows]),
            actual_qpos=actual, actual_qvel=np.stack([r[5] for r in rows]),
            reference_joints=reference, joint_error_rad=errors, root_tilt_deg=tilts)
        peak = int(np.nanargmax(np.abs(errors)))
        frame, joint = np.unravel_index(peak, errors.shape)
        tilt_frame = int(tilts.argmax())
        def where(i):
            return dict(control_time_s=rows[i][1], sim_time_s=rows[i][0],
                        phase=rows[i][2], motion_frame=rows[i][3])
        summary = dict(diagnostics_file=str(path), samples=len(rows),
            peak_joint_error=dict(**where(frame), joint=self.mujoco_joint_names[joint],
                error_rad=float(errors[frame, joint]), actual_rad=float(actual[frame, 7+joint]),
                reference_rad=float(reference[frame, joint])),
            peak_root_tilt=dict(**where(tilt_frame), tilt_deg=float(tilts[tilt_frame])))
        metrics = json.loads(self.metrics_out.read_text())
        summary["matches_aggregate_peaks"] = bool(
            abs(abs(errors[frame, joint])-metrics["reference_joint_error_max_rad"]) < 1e-10
            and abs(tilts[tilt_frame]-metrics["root_tilt_max_deg"]) < 1e-10)
        self.metrics_out.with_name("diagnostics.json").write_text(json.dumps(summary, indent=2))
        print("[Diagnostics] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    # Reuse the exact original CLI, initialization and run loop with an
    # observational subclass. No old runtime files/signatures are modified.
    sim2sim.Sim2Sim = DiagnosticSim
    sim2sim.main()
