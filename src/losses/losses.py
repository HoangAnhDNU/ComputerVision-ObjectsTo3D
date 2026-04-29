"""
Loss functions cho 3D point cloud generation.
Tự implement (không dùng pytorch3d).
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def chamfer_distance(pred_xyz, gt_xyz, return_nn=False):
    """
    Chamfer Distance giữa 2 point clouds.

    Args:
        pred_xyz: [batch, N, 3]
        gt_xyz: [batch, M, 3]
        return_nn: nếu True, trả thêm nearest-neighbor indices (dùng cho color alignment)
    Returns:
        cd: scalar, chamfer distance trung bình trên batch
        nn_idx: [batch, N] (chỉ khi return_nn=True) - index GT gần nhất cho mỗi pred point
    """
    # Pairwise distances: [batch, N, M]
    diff = pred_xyz.unsqueeze(2) - gt_xyz.unsqueeze(1)  # [B, N, 1, 3] - [B, 1, M, 3]
    dist_matrix = (diff ** 2).sum(-1)  # [B, N, M]

    # Min distance từ mỗi pred point đến gt
    min_dist_pred_to_gt, nn_idx = dist_matrix.min(dim=2)  # [B, N]
    # Min distance từ mỗi gt point đến pred
    min_dist_gt_to_pred = dist_matrix.min(dim=1)[0]  # [B, M]

    cd = min_dist_pred_to_gt.mean(dim=1) + min_dist_gt_to_pred.mean(dim=1)  # [B]

    if return_nn:
        return cd.mean(), nn_idx
    return cd.mean()


def earth_mover_distance(pred_xyz, gt_xyz):
    """
    Earth Mover's Distance (approximation) dùng scipy linear_sum_assignment.
    Chạy trên CPU, batch từng sample.

    Args:
        pred_xyz: [batch, N, 3]
        gt_xyz: [batch, M, 3]
    Returns:
        emd: scalar
    """
    batch = pred_xyz.size(0)
    total_emd = 0.0

    pred_np = pred_xyz.detach().cpu().numpy()
    gt_np = gt_xyz.detach().cpu().numpy()

    for i in range(batch):
        # Cost matrix: pairwise distances
        diff = pred_np[i][:, None, :] - gt_np[i][None, :, :]
        cost = (diff ** 2).sum(-1)  # [N, M]

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)
        emd_i = cost[row_ind, col_ind].mean()
        total_emd += emd_i

    return torch.tensor(total_emd / batch, device=pred_xyz.device, dtype=pred_xyz.dtype)


def color_loss(pred_rgb, gt_rgb, nn_idx=None):
    """
    MSE loss cho màu sắc, dùng nearest-neighbor alignment.

    Args:
        pred_rgb: [batch, N, 3]
        gt_rgb: [batch, M, 3]
        nn_idx: [batch, N] - index GT gần nhất cho mỗi pred point (từ chamfer_distance)
    Returns:
        loss: scalar
    """
    if nn_idx is not None:
        # Align GT colors theo nearest-neighbor matching
        nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, 3)  # [B, N, 3]
        matched_gt_rgb = torch.gather(gt_rgb, 1, nn_idx_exp)  # [B, N, 3]
        return ((pred_rgb - matched_gt_rgb) ** 2).mean()
    return ((pred_rgb - gt_rgb) ** 2).mean()


class MultiModal3DLoss(nn.Module):
    """
    Combined loss: L = λ1·CD + λ2·EMD + λ3·Color
    """

    def __init__(self, lambda_cd=1.0, lambda_emd=0.5, lambda_color=0.2, use_emd=True):
        super().__init__()
        self.lambda_cd = lambda_cd
        self.lambda_emd = lambda_emd
        self.lambda_color = lambda_color
        self.use_emd = use_emd

    def forward(self, pred_outputs, gt_point_cloud):
        """
        Args:
            pred_outputs: dict từ decoder, chứa:
                - fine_xyz: [batch, N, 3]
                - colors: [batch, N, 3]
                - coarse_xyz: [batch, M, 3] (optional)
            gt_point_cloud: [batch, N, 6] (XYZ + RGB)
        Returns:
            dict các loss values
        """
        pred_xyz = pred_outputs["fine_xyz"]
        pred_rgb = pred_outputs["colors"]

        gt_xyz = gt_point_cloud[:, :, :3]
        gt_rgb = gt_point_cloud[:, :, 3:]

        # Chamfer Distance (tính thêm NN index cho color alignment)
        cd, nn_idx = chamfer_distance(pred_xyz, gt_xyz, return_nn=True)

        # Coarse Chamfer (nếu có)
        coarse_cd = torch.tensor(0.0, device=cd.device)
        if "coarse_xyz" in pred_outputs:
            # Subsample GT cho coarse comparison
            n_coarse = pred_outputs["coarse_xyz"].size(1)
            indices = torch.randperm(gt_xyz.size(1), device=gt_xyz.device)[:n_coarse]
            gt_coarse = gt_xyz[:, indices, :]
            coarse_cd = chamfer_distance(pred_outputs["coarse_xyz"], gt_coarse)

        # EMD (tùy chọn, chậm)
        emd = torch.tensor(0.0, device=cd.device)
        if self.use_emd and self.lambda_emd > 0:
            emd = earth_mover_distance(pred_xyz, gt_xyz)

        # Color loss (dùng NN alignment từ CD ở trên)
        c_loss = color_loss(pred_rgb, gt_rgb, nn_idx=nn_idx)

        # Total
        total = self.lambda_cd * (cd + 0.5 * coarse_cd) + self.lambda_emd * emd + self.lambda_color * c_loss

        return {
            "total": total,
            "chamfer_distance": cd,
            "coarse_cd": coarse_cd,
            "emd": emd,
            "color_loss": c_loss,
        }
