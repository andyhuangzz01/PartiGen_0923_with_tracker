"""Asset-free FK math tests; these do not validate a real G1 skeleton."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from robotmdar.skeleton.kinematics_math import (
    forward_difference_velocity,
    quaternion_xyzw_to_axis_angle,
    validate_time_delta,
)


def axis_angle_matrix(vector):
    """Independent rotation exponential for checking converted axis angles."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    return torch.matrix_exp(skew.reshape(*vector.shape[:-1], 3, 3))


def composite_root_rotation():
    roll, pitch, yaw = torch.tensor([0.4, 0.3, 1.0], dtype=torch.double)
    cr, cp, cy = torch.cos(torch.stack((roll, pitch, yaw)) / 2)
    sr, sp, sy = torch.sin(torch.stack((roll, pitch, yaw)) / 2)
    quaternion = torch.stack((sr * cp * cy - cr * sp * sy,
                              cr * sp * cy + sr * cp * sy,
                              cr * cp * sy - sr * sp * cy,
                              cr * cp * cy + sr * sp * sy))
    rotations = axis_angle_matrix(torch.diag(torch.stack((roll, pitch, yaw))))
    matrix = rotations[2] @ rotations[1] @ rotations[0]
    return quaternion, matrix


class QuaternionAxisAngleTests(unittest.TestCase):
    def test_composite_rotation_matrix_and_child_position(self):
        quaternion, expected = composite_root_rotation()
        actual = axis_angle_matrix(quaternion_xyzw_to_axis_angle(quaternion))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        child_offset = torch.tensor([0.0, 0.0, -0.8], dtype=torch.double)
        torch.testing.assert_close(actual @ child_offset, expected @ child_offset)

    def test_identity_small_angles_and_signs(self):
        quaternions = torch.tensor([[0, 0, 0, 1], [1e-10, -2e-10, 3e-10, 1],
                                    [0.2, -0.3, 0.4, 0.8], [0, -1, 0, 0]],
                                   dtype=torch.double)
        converted = quaternion_xyzw_to_axis_angle(quaternions)
        self.assertTrue(torch.isfinite(converted).all())
        torch.testing.assert_close(converted[0], torch.zeros(3, dtype=torch.double))
        torch.testing.assert_close(converted[1], 2 * quaternions[1, :3], atol=1e-20, rtol=1e-10)
        torch.testing.assert_close(converted, quaternion_xyzw_to_axis_angle(-quaternions))
        torch.testing.assert_close(converted, quaternion_xyzw_to_axis_angle(3 * quaternions))

    def test_identity_small_and_composite_gradients(self):
        composite, _ = composite_root_rotation()
        for quaternion in (torch.tensor([0., 0., 0., 1.], dtype=torch.double),
                           torch.tensor([1e-8, -2e-8, 3e-8, 1.], dtype=torch.double),
                           composite):
            quaternion.requires_grad_()
            self.assertTrue(torch.autograd.gradcheck(quaternion_xyzw_to_axis_angle,
                                                    (quaternion,), atol=1e-6, rtol=1e-4))
        identity = torch.tensor([0., 0., 0., 1.], requires_grad=True)
        quaternion_xyzw_to_axis_angle(identity).sum().backward()
        torch.testing.assert_close(identity.grad, torch.tensor([2., 2., 2., 0.]))

    def test_invalid_quaternions(self):
        for quaternion in (torch.zeros(4), torch.tensor([0., 0., 0., float('nan')]),
                           torch.ones(3), torch.ones(4, dtype=torch.long)):
            with self.assertRaises(ValueError):
                quaternion_xyzw_to_axis_angle(quaternion)

    def test_actual_robot_wrapper_passes_axis_angle_and_gradients(self):
        # Run the actual wrapper method without importing Isaac or constructing
        # robot assets. The stub records the pose sent to FK; it is not G1 FK.
        source = Path(__file__).resolve().parents[1] / 'robotmdar/skeleton/robot.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        robot = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == 'RobotSkeleton')
        method = next(node for node in robot.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'forward_kinematics')
        namespace = {'torch': torch, 'MotionDict': dict,
                     'quaternion_xyzw_to_axis_angle': quaternion_xyzw_to_axis_angle}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)

        class RecordingFK:
            def dof_to_axis_angle(self, dof):
                return dof[..., None] * dof.new_tensor([0., 0., 1.])

            def fk_batch(self, pose, translation, return_full):
                self.pose, self.translation = pose, translation
                return {}

        quaternion, expected = composite_root_rotation()
        for batched in (False, True):
            fk = RecordingFK()
            skeleton = SimpleNamespace(fk=fk, num_extend_dof=1)
            shape = (2, 3) if batched else (3,)
            root = quaternion.expand(*shape, 4).clone().requires_grad_()
            motion = {'root_rot': root, 'dof': torch.zeros(*shape, 1, dtype=torch.double),
                      'root_trans_offset': torch.zeros(*shape, 3, dtype=torch.double)}
            namespace['forward_kinematics'](skeleton, motion)
            matrix = axis_angle_matrix(fk.pose[..., 0, :])
            torch.testing.assert_close(matrix, expected.expand_as(matrix))
            self.assertEqual(fk.pose.dtype, root.dtype)
            matrix[..., 0, 1].sum().backward()
            self.assertTrue(torch.isfinite(root.grad).all())
            self.assertGreater(root.grad.abs().sum().item(), 0)


class ForwardDifferenceTests(unittest.TestCase):
    def test_last_interval_is_extended(self):
        position = torch.tensor([[[0.], [1.], [3.], [6.]]])
        torch.testing.assert_close(forward_difference_velocity(position, 1 / 30),
                                   torch.tensor([[[30.], [60.], [90.], [90.]]]))

    def test_one_and_two_frame_shape_and_gradients(self):
        for length in (1, 2):
            position = torch.arange(2 * length * 3, dtype=torch.double).reshape(2, length, 3)
            position.requires_grad_()
            velocity = forward_difference_velocity(position, 0.5)
            self.assertEqual(velocity.shape, position.shape)
            self.assertEqual(velocity.dtype, position.dtype)
            if length == 1:
                torch.testing.assert_close(velocity, torch.zeros_like(position))
            else:
                torch.testing.assert_close(velocity, torch.full_like(position, 6))
            velocity.sum().backward()
            self.assertTrue(torch.isfinite(position.grad).all())

    def test_time_delta_and_empty_sequence_rejected(self):
        for time_delta in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                forward_difference_velocity(torch.zeros(2, 1, 3), time_delta)
        with self.assertRaises(ValueError):
            forward_difference_velocity(torch.zeros(2, 0, 3), 1)


class VelocityFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the source methods without loading Isaac/assets. Only the
        # single-frame angular path is tested here; it needs no Isaac rotation.
        source = Path(__file__).resolve().parents[1] / 'robotmdar/skeleton/forward_kinematics.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        fk = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == 'ForwardKinematics')
        names = {'_gaussian_smooth1d', '_compute_velocity', '_compute_angular_velocity'}
        fk.body = [node for node in fk.body if isinstance(node, ast.FunctionDef)
                   and node.name in names]
        namespace = {'torch': torch, 'validate_time_delta': validate_time_delta}
        exec(compile(ast.Module(body=[fk], type_ignores=[]), str(source), 'exec'), namespace)
        cls.fk = namespace['ForwardKinematics']

    def test_smoothing_impulse_stays_in_its_joint_coordinate(self):
        trajectory = torch.zeros(2, 5, 3, 3, dtype=torch.double)
        trajectory[0, 2, 1, 2] = 1
        trajectory.requires_grad_()
        expected = torch.zeros_like(trajectory)
        expected[0, 1:4, 1, 2] = torch.tensor([0.25, 0.5, 0.25], dtype=torch.double)
        filtered = self.fk._gaussian_smooth1d(trajectory)
        torch.testing.assert_close(filtered, expected)
        filtered[0, 2, 1, 2].backward()
        torch.testing.assert_close(trajectory.grad, expected)

    def test_smoothing_preserves_independent_constant_channels(self):
        for shape in ((1, 3, 3), (5, 3, 3), (2, 2, 5, 3, 3)):
            channels = torch.arange(9, dtype=torch.double).reshape(3, 3)
            trajectory = channels.expand(*shape)
            torch.testing.assert_close(self.fk._gaussian_smooth1d(trajectory), trajectory)

    def test_linear_velocity_filter_does_not_mix_coordinates(self):
        position = torch.zeros(2, 5, 3, 3, dtype=torch.double)
        position[0, 2, 1, 2] = 1
        expected = torch.zeros_like(position)
        expected[0, :, 1, 2] = torch.tensor([0.125, 0.25, 0, -0.25, -0.125], dtype=torch.double)
        torch.testing.assert_close(self.fk._compute_velocity(position, 1), expected)

    def test_single_frame_angular_velocity_has_three_coordinates(self):
        for shape in ((2, 1, 3, 4), (1, 3, 4)):
            rotation = torch.zeros(shape, dtype=torch.double)
            rotation[..., 3] = 1
            velocity = self.fk._compute_angular_velocity(rotation, 1 / 30)
            self.assertEqual(velocity.shape, (*shape[:-1], 3))
            torch.testing.assert_close(velocity, torch.zeros_like(rotation[..., :3]))

    def test_invalid_time_delta_rejected_by_velocity_methods(self):
        for time_delta in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.fk._compute_velocity(torch.zeros(2, 1, 3, 3), time_delta)
            with self.assertRaises(ValueError):
                self.fk._compute_angular_velocity(torch.zeros(2, 1, 3, 4), time_delta)


if __name__ == '__main__':
    unittest.main()
