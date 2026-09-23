"""Mean Huber reconstruction and the paper's hierarchical kinematic objective."""
import math

import torch
from torch.nn import functional as F


def learning_rate_at(step, max_steps, learning_rate, schedule):
    fraction = min(max(step / max_steps, 0.0), 1.0)
    if schedule == 'cosine':
        factor = 0.5 * (1 + math.cos(math.pi * fraction))
    elif schedule == 'linear':
        factor = 1 - fraction
    else:
        raise ValueError(f"Unsupported annealing schedule: {schedule}")
    return learning_rate * factor


def stage_at(step, stages):
    boundary = 0
    for index, count in enumerate(stages):
        boundary += count
        if step < boundary:
            return index
    return len(stages) - 1


def masked_huber(pred, target, valid):
    loss = F.huber_loss(pred, target, reduction='none', delta=1.0)
    mask = valid.bool().reshape(*valid.shape, *((1,) * (loss.ndim - valid.ndim))).expand_as(loss)
    return loss.masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1)


def geometry_terms(dataset, pred, target, history, valid, temporal=False):
    """Reconstruct history+future jointly; padding never enters loss denominators."""
    history_len = history.shape[1]
    # FK uses temporal differences; unconstrained padded predictions must not
    # affect the last valid frame's velocity or receive gradients through it.
    final_index = (valid.sum(1).long() - 1)[:, None, None].expand(-1, 1, pred.shape[-1])
    pred = torch.where(valid[..., None], pred, pred.gather(1, final_index))
    target = torch.where(valid[..., None], target, target.gather(1, final_index))
    joined_pred = torch.cat((history, pred), dim=1)
    joined_gt = torch.cat((history, target), dim=1)
    pred_fk = dataset.reconstruct_motion(joined_pred, need_denormalize=True, ret_fk=True)
    with torch.no_grad():
        gt_fk = dataset.reconstruct_motion(joined_gt, need_denormalize=True, ret_fk=True)
    fields = {'body_trans': 'global_translation_extend', 'body_rot': 'global_rotation',
              'dof_pos': 'dof_pos', 'dof_vel': 'dof_vel'}
    terms = {name: masked_huber(pred_fk[key][:, history_len:], gt_fk[key][:, history_len:], valid)
             for name, key in fields.items()}
    # Keep the server FK's 30 Hz forward-difference convention (Appendix S2.5).
    pq, gq = pred_fk['dof_pos'], gt_fk['dof_pos']
    feet = dataset.skeleton.foot_id
    contact = gt_fk['contact_mask'][:, history_len:].unsqueeze(-1)
    foot_pred = pred_fk['global_translation_extend'][:, history_len:, feet] * contact
    foot_gt = gt_fk['global_translation_extend'][:, history_len:, feet] * contact
    terms['foot_contact'] = masked_huber(foot_pred, foot_gt, valid)
    last = history_len + valid.sum(dim=1).long() - 1
    rows = torch.arange(len(pred), device=pred.device)
    terms['drift_xy'] = F.huber_loss(pred_fk['global_translation_extend'][rows, last, 0, :2],
                                    gt_fk['global_translation_extend'][rows, last, 0, :2], delta=1.0)
    # Feature channel 4 is a yaw increment, so wrapped cumulative differences
    # give heading drift without depending on quaternion sign conventions.
    raw_pred, raw_gt = dataset.denormalize(pred), dataset.denormalize(target)
    # The last feature's increment leads to an unobserved next frame.
    yaw_valid = valid & (torch.arange(pred.shape[1], device=pred.device)[None] < valid.sum(1)[:, None] - 1)
    yaw_error = ((raw_pred[..., 4] - raw_gt[..., 4]) * yaw_valid).sum(1)
    yaw_error = torch.atan2(yaw_error.sin(), yaw_error.cos())
    terms['drift_yaw'] = F.huber_loss(yaw_error, torch.zeros_like(yaw_error), delta=1.0)
    if temporal:
        for name, key in [('joints_delta', 'global_translation_extend'), ('dof_delta', 'dof_pos')]:
            p, g = pred_fk[key], gt_fk[key]
            terms[name] = masked_huber(p[:, history_len:] - p[:, history_len - 1:-1],
                                       g[:, history_len:] - g[:, history_len - 1:-1], valid)
        # Table S4 defines orientation increments for the root, not every body.
        p, g = pred_fk['global_rotation'][:, :, 0], gt_fk['global_rotation'][:, :, 0]
        terms['orient_delta'] = masked_huber(p[:, history_len:] - p[:, history_len - 1:-1],
                                            g[:, history_len:] - g[:, history_len - 1:-1], valid)
        p, g = pred_fk['global_translation_extend'][:, :, 0], gt_fk['global_translation_extend'][:, :, 0]
        terms['trans_delta'] = masked_huber(p[:, history_len:] - p[:, history_len - 1:-1],
                                           g[:, history_len:] - g[:, history_len - 1:-1], valid)
        # Second-order joint-angle smoothness, rather than legacy feature L1.
        second = pq[:, history_len:] - 2 * pq[:, history_len - 1:-1] + pq[:, history_len - 2:-2]
        terms['smooth'] = masked_huber(second, torch.zeros_like(second), valid)
    return terms


def weighted_geometry(terms, weights):
    return sum(weights.get(name, 0.0) * value for name, value in terms.items())
