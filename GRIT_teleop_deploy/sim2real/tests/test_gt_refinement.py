import sys
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from refine_gt_005225 import transform, SPECS, gt_slow
from gt_refined import confirmed_result, qualified
from screen_generated_motions import LIMITS, classify


class RefinementTest(unittest.TestCase):
    def test_three_complete_runs_required_and_worst_limits_used(self):
        prepared=dict(signature={"version":"test"},artifacts={})
        result=dict(prepared,index=5225,text="steps and waves",error=None,
                    source_xy_path_m=.3,source_leg_excursion_rad=.3,
                    tracking_motion_frames_complete=True,tracking_recorded_frames=100,
                    expected_motion_frames=100,global_mpjpe_mm=100,root_relative_mpjpe_mm=30)
        for key in LIMITS: result[key]=.75 if key=="root_z_min" else 0.
        result["rating"],result["hardware_screen"],result["reasons"]=classify(result)
        self.assertTrue(qualified(result))
        with TemporaryDirectory() as temp:
            paths=[]
            for i in range(3):
                r=dict(result,root_tilt_max_deg=10+i)
                p=Path(temp)/f"{i}.json";p.write_text(json.dumps(r));paths.append(p)
            confirmed=confirmed_result(prepared,paths)
            self.assertEqual(confirmed["root_tilt_max_deg"],12)
            self.assertEqual(confirmed["confirmed_runs"],3)
            with self.assertRaises(ValueError): confirmed_result(prepared,paths[:2])
            bad=dict(result,tracking_motion_frames_complete=False)
            paths[-1].write_text(json.dumps(bad))
            with self.assertRaises(ValueError): confirmed_result(prepared,paths)

    def test_transform_keeps_gesture_and_minimum_toe_height(self):
        model=mujoco.MjModel.from_xml_path(str(gt_slow.gt.XML))
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(1,model.njnt)]
        pos=np.array([[0.,0.,.78],[.05,.1,.78],[.1,0.,.78]])
        quat=np.tile([1.,0,0,0],(3,1))
        joints=np.zeros((3,29));default=np.zeros(29)
        joints[:,names.index("left_hip_roll_joint")]=[0,.1,0]
        joints[:,names.index("left_shoulder_pitch_joint")]=[.1,.2,.3]
        joints[:,names.index("left_wrist_pitch_joint")]=-.3
        originals=[a.copy() for a in (pos,quat,joints)]
        p,q,j,_=transform(model,pos,quat,joints,default,SPECS["gentle90"])
        for a,b in zip((pos,quat,joints),originals): np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(j[:,names.index("left_shoulder_pitch_joint")],[.1,.2,.3])
        np.testing.assert_allclose(j[:,names.index("left_hip_roll_joint")],[0,.09,0])
        np.testing.assert_allclose(j[:,names.index("left_wrist_pitch_joint")],-.18)
        toes=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ("left_toe_link","right_toe_link")]
        data=mujoco.MjData(model)
        for i in range(3):
            data.qpos[:]=np.r_[pos[i],quat[i],joints[i]];mujoco.mj_forward(model,data)
            height=data.xpos[toes,2].min()
            data.qpos[:]=np.r_[p[i],q[i],j[i]];mujoco.mj_forward(model,data)
            self.assertAlmostEqual(height,data.xpos[toes,2].min(),places=9)


if __name__=="__main__": unittest.main()
