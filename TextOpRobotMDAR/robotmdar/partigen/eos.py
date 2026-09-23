"""Frame-level termination targets, loss and post-generation endpoint selection."""
import torch
from torch.nn import functional as F


def primitive_targets(start, end, block_index, future_len, history_len=2, horizon=320,
                      *, sequence_len):
    """Annotation ends are exclusive; horizon truncation is not a positive EOS.

    Return raw frame indices (including one look-ahead frame for differences),
    valid future frames, EOS labels and one-based frame positions in the action.
    ``sequence_len`` is the physical motion length, independent of the annotation
    endpoint. Preserve real adjacent states for Table S3 increments, including
    after the annotation; only hold the final state at the physical sequence end.
    Frames outside the annotation or horizon remain excluded by ``valid``.
    """
    if sequence_len <= 0 or not 0 <= start < end <= sequence_len:
        raise ValueError("Annotation must be nonempty and lie within the physical sequence")
    if future_len <= 0 or horizon <= 0 or history_len < 0:
        raise ValueError("Positive block size and horizon and nonnegative history are required")
    length = min(end - start, horizon)
    offset = block_index * future_len
    if offset < 0 or offset >= length:
        raise ValueError("Primitive must contain at least one valid future frame")
    future = start + offset + torch.arange(future_len)
    valid = future < start + length
    eos = (future == end - 1) & valid
    raw = start + offset + torch.arange(-history_len, future_len + 1)
    raw = raw.clamp(min=0, max=sequence_len - 1)
    return raw.long(), valid, eos.float(), offset + torch.arange(1, future_len + 1)


def eos_loss(logits, targets, valid_mask, positive_weight):
    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("EOS logits, labels and valid mask must have identical [B, T] shapes")
    loss = F.binary_cross_entropy_with_logits(logits, targets.to(logits), reduction="none",
                                              pos_weight=logits.new_tensor(positive_weight))
    valid = valid_mask.bool()
    return loss.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


def endpoint_lengths(probabilities, threshold=0.9):
    """First STRICT threshold crossing, inclusive; no crossing retains all frames."""
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("Expected nonempty [batch, candidate_frames] probabilities")
    if not 0 <= threshold <= 1:
        raise ValueError("EOS threshold must be a probability")
    crossed = probabilities > threshold
    first = crossed.int().argmax(dim=1) + 1
    return torch.where(crossed.any(dim=1), first, torch.full_like(first, probabilities.shape[1]))
