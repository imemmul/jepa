import torch
from torch.functional import F
from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation
import torch

def rotmat_to_quat(R_mat):
    """
    Convert rotation matrices to quaternions using scipy.

    Args:
        R_mat: Rotation matrices, shape [batch_size, frames, 3, 3]

    Returns:
        quaternions: Quaternions, shape [batch_size, frames, 4] (w, x, y, z)
    """
    batch_size, frames, _, _ = R_mat.shape
    R_mat_flat = R_mat.view(-1, 3, 3).cpu().numpy()
    rot = Rotation.from_matrix(R_mat_flat)
    quaternions = rot.as_quat()
    quaternions = torch.from_numpy(quaternions).to(R_mat.device)
    quaternions = quaternions[:, [3, 0, 1, 2]]
    quaternions = quaternions.view(batch_size, frames, 4)
    return quaternions


def quaternion_loss(q_pred, q_gt):
    """
    Compute the loss between predicted and ground truth quaternions.

    Args:
        q_pred: Predicted quaternions, shape [B, T, Q, 4]
        q_gt: Ground truth quaternions, shape [B, T, Q, 4]

    Returns:
        loss_rot: Scalar tensor representing the rotation loss
    """
    # Normalize quaternions
    q_pred = F.normalize(q_pred, p=2, dim=-1)
    q_gt = F.normalize(q_gt, p=2, dim=-1)

    # Compute the absolute dot product to account for double-cover
    dot_product = torch.abs(torch.sum(q_pred * q_gt, dim=-1))  # [B, T, Q]
    dot_product = torch.clamp(dot_product, min=-1.0 + 1e-6, max=1.0 - 1e-6)

    # Compute angular difference
    theta = 2 * torch.acos(dot_product)  # [B, T, Q]

    # Compute mean loss
    loss_rot = torch.mean(theta)
    return loss_rot

def compute_loss(predictions, ground_truth):
    t_pred = predictions[..., :3] # 3
    dims_pred = predictions[..., 3:6] # 3
    R_pred = predictions[..., 6:] # 4

    t_gt = ground_truth[..., :3]
    dims_gt = ground_truth[..., 3:6]
    R_gt = ground_truth[..., 6:]

    # Translation loss
    loss_trans = F.l1_loss(t_pred, t_gt)

    # Dimension loss
    loss_dims = F.l1_loss(dims_pred, dims_gt)

    loss_rot = quaternion_loss(R_pred, R_gt)

    # Total loss with weights
    total_loss = 1.0 * loss_trans + 1.0 * loss_dims + 10.0* loss_rot
    return total_loss
