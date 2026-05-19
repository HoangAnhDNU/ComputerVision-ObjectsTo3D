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
    # Dùng |a-b|² = |a|² + |b|² - 2a·b để tránh tạo tensor [B,N,M,3] tốn 3GB
    pred_sq = (pred_xyz ** 2).sum(-1, keepdim=True)          # [B, N, 1]
    gt_sq   = (gt_xyz   ** 2).sum(-1, keepdim=True)          # [B, M, 1]
    cross   = torch.bmm(pred_xyz, gt_xyz.transpose(1, 2))    # [B, N, M]
    dist_matrix = pred_sq + gt_sq.transpose(1, 2) - 2.0 * cross  # [B, N, M]
    dist_matrix = dist_matrix.clamp(min=0.0)

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


def repulsion_loss(pred_xyz, k=8, h=0.03, max_points=512):
    """
    Repulsion Loss: đẩy các điểm quá gần nhau ra xa.
    Giảm hiện tượng tụ cụm (clustering), giúp phủ bề mặt đều hơn.

    Với mỗi điểm, tìm k nearest neighbors, phạt nếu khoảng cách < h.
    Loss = mean(max(0, h² - dist²)²) cho tất cả k-NN pairs.

    Tiết kiệm VRAM: nếu N > max_points, random subsample để tránh O(N²) memory.

    Args:
        pred_xyz: [batch, N, 3]
        k: số nearest neighbors để xét
        h: ngưỡng khoảng cách (points gần hơn h bị phạt)
        max_points: giới hạn số points để tính (tiết kiệm VRAM)
    Returns:
        loss: scalar
    """
    N = pred_xyz.size(1)

    # Subsample nếu quá nhiều points (tránh O(N²) memory blowup)
    if N > max_points:
        idx = torch.randperm(N, device=pred_xyz.device)[:max_points]
        pred_xyz = pred_xyz[:, idx, :]

    # Pairwise distances: [B, N_sub, N_sub]
    diff = pred_xyz.unsqueeze(2) - pred_xyz.unsqueeze(1)
    dist_sq = (diff ** 2).sum(-1)

    # Loại bỏ self-distance (set diagonal = inf)
    diag_mask = torch.eye(dist_sq.size(1), device=dist_sq.device, dtype=torch.bool)
    dist_sq = dist_sq.masked_fill(diag_mask.unsqueeze(0), float('inf'))

    # k-nearest neighbors
    k_actual = min(k, dist_sq.size(1) - 1)
    knn_dist_sq, _ = dist_sq.topk(k_actual, dim=-1, largest=False)  # [B, N_sub, k]

    # Repulsion: phạt nếu dist < h
    h_sq = h * h
    penalty = torch.clamp(h_sq - knn_dist_sq, min=0.0) ** 2

    return penalty.mean()


def normal_consistency_loss(pred_xyz, k=8, max_points=512):
    """
    Differentiable normal consistency loss (Laplacian smoothness).
    Phạt các điểm có vị trí không nhất quán với neighbors → bề mặt mượt hơn.

    Cách tiếp cận: Laplacian deviation
        L(i) = mean_j(x_j - x_i) cho k nearest neighbors
        Loss = ||L||^2 → nhỏ khi bề mặt mượt, lớn khi có gai/lỗi
    Hoàn toàn differentiable, không cần numpy.

    Args:
        pred_xyz: [batch, N, 3]
        k: số nearest neighbors
        max_points: giới hạn số points để tránh O(N²) memory
    Returns:
        loss: scalar
    """
    B, N, _ = pred_xyz.shape

    if N > max_points:
        idx = torch.randperm(N, device=pred_xyz.device)[:max_points]
        sub = pred_xyz[:, idx, :]
    else:
        sub = pred_xyz
    N_sub = sub.size(1)

    # Pairwise distances [B, N_sub, N_sub]
    diff_all = sub.unsqueeze(2) - sub.unsqueeze(1)  # [B, N, N, 3]
    dist_sq = (diff_all ** 2).sum(-1)  # [B, N, N]

    # Loại bỏ self-distance
    diag = torch.eye(N_sub, device=pred_xyz.device, dtype=torch.bool).unsqueeze(0)
    dist_sq = dist_sq.masked_fill(diag, float('inf'))

    # K nearest neighbors
    k_actual = min(k, N_sub - 1)
    _, knn_idx = dist_sq.topk(k_actual, dim=-1, largest=False)  # [B, N, k]

    # Gather neighbor positions: [B, N_sub, k, 3]
    flat_idx = knn_idx.reshape(B, -1)  # [B, N_sub*k]
    flat_idx_exp = flat_idx.unsqueeze(-1).expand(-1, -1, 3)
    neighbors = torch.gather(sub, 1, flat_idx_exp).reshape(B, N_sub, k_actual, 3)

    # Laplacian: độ lệch so với vị trí trung bình của neighbors
    laplacian = (neighbors - sub.unsqueeze(2)).mean(dim=2)  # [B, N_sub, 3]
    return (laplacian ** 2).mean()


def uniformity_loss(pred_xyz, gt_xyz, nn_idx=None):
    """
    Uniformity Loss: phạt phân bố không đều bằng cách so sánh
    local density giữa predicted và ground-truth point clouds.

    Ý tưởng: Với mỗi pred point, xét GT nearest neighbor.
    Nếu nhiều pred points cùng map đến 1 vùng GT → vùng khác bị trống.
    Phạt variance của coverage counts.

    Tiết kiệm VRAM: tái sử dụng nn_idx từ chamfer_distance (không cần tính lại [B,N,M]).

    Args:
        pred_xyz: [batch, N, 3]
        gt_xyz: [batch, M, 3]
        nn_idx: [batch, N] - index GT gần nhất (từ chamfer_distance, tiết kiệm VRAM)
    Returns:
        loss: scalar
    """
    if nn_idx is None:
        # Fallback: tính lại (tốn VRAM)
        diff = pred_xyz.unsqueeze(2) - gt_xyz.unsqueeze(1)
        dist_sq = (diff ** 2).sum(-1)  # [B, N, M]
        nn_idx = dist_sq.argmin(dim=2)  # [B, N]

    batch, N = nn_idx.shape
    M = gt_xyz.size(1)

    # Đếm bao nhiêu pred points map đến mỗi GT point
    counts = torch.zeros(batch, M, device=pred_xyz.device, dtype=pred_xyz.dtype)
    ones = torch.ones(batch, N, device=pred_xyz.device, dtype=pred_xyz.dtype)
    counts.scatter_add_(1, nn_idx, ones)  # [B, M]

    # Lý tưởng: mỗi GT point có ~N/M pred points
    # Phạt variance → buộc phân bố đều
    mean_count = counts.mean(dim=1, keepdim=True)  # [B, 1]
    variance = ((counts - mean_count) ** 2).mean(dim=1)  # [B]

    return variance.mean()


class MultiModal3DLoss(nn.Module):
    """
    Combined loss: L = λ1·CD + λ2·EMD + λ3·Color + λ4·Repulsion + λ5·Uniform + λ6·Normal
    """

    def __init__(self, lambda_cd=1.0, lambda_emd=0.5, lambda_color=0.2,
                 lambda_repulsion=0.1, lambda_uniform=0.05, lambda_normal=0.0,
                 use_emd=True, repulsion_k=8, repulsion_h=0.03, normal_k=8):
        super().__init__()
        self.lambda_cd = lambda_cd
        self.lambda_emd = lambda_emd
        self.lambda_color = lambda_color
        self.lambda_repulsion = lambda_repulsion
        self.lambda_uniform = lambda_uniform
        self.lambda_normal = lambda_normal
        self.use_emd = use_emd
        self.repulsion_k = repulsion_k
        self.repulsion_h = repulsion_h
        self.normal_k = normal_k

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

        # Repulsion loss (chống tụ cụm, subsample để tiết kiệm VRAM)
        rep_loss = torch.tensor(0.0, device=cd.device)
        if self.lambda_repulsion > 0:
            rep_loss = repulsion_loss(pred_xyz, k=self.repulsion_k, h=self.repulsion_h,
                                     max_points=512)

        # Uniformity loss (phân bố đều, tái dùng nn_idx từ CD)
        uni_loss = torch.tensor(0.0, device=cd.device)
        if self.lambda_uniform > 0:
            uni_loss = uniformity_loss(pred_xyz, gt_xyz, nn_idx=nn_idx)

        # Normal Consistency Loss (Laplacian smoothness → bề mặt mượt hơn)
        norm_loss = torch.tensor(0.0, device=cd.device)
        if self.lambda_normal > 0:
            norm_loss = normal_consistency_loss(pred_xyz, k=self.normal_k, max_points=512)

        # Total
        total = (self.lambda_cd * (cd + 0.5 * coarse_cd)
                 + self.lambda_emd * emd
                 + self.lambda_color * c_loss
                 + self.lambda_repulsion * rep_loss
                 + self.lambda_uniform * uni_loss
                 + self.lambda_normal * norm_loss)

        return {
            "total": total,
            "chamfer_distance": cd,
            "coarse_cd": coarse_cd,
            "emd": emd,
            "color_loss": c_loss,
            "repulsion_loss": rep_loss,
            "uniformity_loss": uni_loss,
            "normal_loss": norm_loss,
        }
