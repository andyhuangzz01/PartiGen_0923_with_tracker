"""Differentiable kinematics helpers independent of robot assets and Isaac."""
import math

import torch


def quaternion_xyzw_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert (..., 4) xyzw quaternions to principal axis-angle vectors.

    Normalization permits non-unit inputs. The sinc formulation is well-defined
    at the identity and preserves gradients through small rotations. Quaternion
    signs are canonicalized because q and -q encode the same orientation.
    """
    if quaternion.shape[-1] != 4 or not quaternion.is_floating_point():
        raise ValueError("quaternion must be a floating tensor with final dimension 4")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if not bool(torch.all(torch.isfinite(norm) & (norm > 0))):
        raise ValueError("quaternion must be finite and nonzero")
    quaternion = quaternion / norm
    vector, scalar = quaternion[..., :3], quaternion[..., 3:]
    # At exactly pi, resolve the otherwise ambiguous sign by the dominant axis.
    dominant = vector.gather(-1, vector.abs().argmax(dim=-1, keepdim=True))
    sign_source = torch.where(scalar == 0, dominant, scalar)
    quaternion = torch.where(sign_source < 0, -quaternion, quaternion)
    vector, scalar = quaternion[..., :3], quaternion[..., 3:]
    half_angle = torch.atan2(torch.linalg.vector_norm(vector, dim=-1, keepdim=True), scalar)
    return 2 * vector / torch.sinc(half_angle / math.pi)


def validate_time_delta(time_delta: float) -> None:
    """Reject undefined or nonphysical finite-difference time intervals."""
    if not math.isfinite(time_delta) or time_delta <= 0:
        raise ValueError("time_delta must be finite and strictly positive")


def forward_difference_velocity(position: torch.Tensor, time_delta: float) -> torch.Tensor:
    """Return [B, T, ...] forward differences, extending the final interval.

    The paper defines (q[t+1] - q[t]) / dt but leaves the last frame unspecified.
    This implementation repeats the last available forward difference to retain
    T frames; a one-frame sequence has no interval and returns zero velocity.
    """
    validate_time_delta(time_delta)
    if position.ndim < 2 or position.shape[1] == 0:
        raise ValueError("position must have a nonempty time dimension at axis 1")
    if position.shape[1] == 1:
        return position * 0
    velocity = (position[:, 1:] - position[:, :-1]) / time_delta
    return torch.cat((velocity, velocity[:, -1:]), dim=1)
