"""
Point Cloud Decoder: 2-stage (coarse → fine) với color prediction.
"""

import torch
import torch.nn as nn


class CoarseDecoder(nn.Module):
    """
    Stage 1: Fused features → coarse point cloud [batch, coarse_points, 3].
    """

    def __init__(self, input_dim=512, coarse_points=256):
        super().__init__()

        self.coarse_points = coarse_points

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, coarse_points * 3),
        )

    def forward(self, fused_features):
        """
        Args:
            fused_features: [batch, input_dim]
        Returns:
            coarse_xyz: [batch, coarse_points, 3]
        """
        batch = fused_features.size(0)
        out = self.mlp(fused_features)
        coarse_xyz = out.view(batch, self.coarse_points, 3)
        return coarse_xyz


class FineDecoder(nn.Module):
    """
    Stage 2: Mỗi coarse point → 8 fine points (FoldingNet-style).
    Output: [batch, fine_points, 3]
    """

    def __init__(self, input_dim=512, coarse_points=256, fold_ratio=8):
        super().__init__()

        self.coarse_points = coarse_points
        self.fold_ratio = fold_ratio
        self.fine_points = coarse_points * fold_ratio

        per_point_dim = 64

        # Global feature → per-point features
        self.global_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, coarse_points * per_point_dim),
        )

        # Folding MLP: từ (coarse_xyz + per_point_feat) → 3D offset
        self.fold_mlp = nn.Sequential(
            nn.Linear(3 + per_point_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * fold_ratio),
        )

    def forward(self, fused_features, coarse_xyz):
        """
        Args:
            fused_features: [batch, input_dim]
            coarse_xyz: [batch, coarse_points, 3]
        Returns:
            fine_xyz: [batch, fine_points, 3]
        """
        batch = fused_features.size(0)

        # Global feature → per-point features
        global_feat = self.global_proj(fused_features)
        global_feat = global_feat.view(batch, self.coarse_points, 64)

        # Concat coarse_xyz + per-point features
        local_feat = torch.cat([coarse_xyz, global_feat], dim=-1)  # [B, coarse, 3+64]

        # Folding: mỗi điểm → fold_ratio điểm
        offsets = self.fold_mlp(local_feat)  # [B, coarse, 3*fold_ratio]
        offsets = offsets.view(batch, self.coarse_points, self.fold_ratio, 3)

        # Expand coarse points và cộng offset
        coarse_expanded = coarse_xyz.unsqueeze(2).expand(-1, -1, self.fold_ratio, -1)
        fine_xyz = coarse_expanded + offsets  # [B, coarse, fold_ratio, 3]
        fine_xyz = fine_xyz.reshape(batch, self.fine_points, 3)

        return fine_xyz


class ColorHead(nn.Module):
    """
    Dự đoán màu RGB cho mỗi điểm.
    Input: fine_xyz + per-point features → RGB [0, 1].
    """

    def __init__(self, input_dim=512, num_points=2048):
        super().__init__()

        # Per-point feature từ global
        self.feat_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 64),
        )

        # Color MLP
        self.color_mlp = nn.Sequential(
            nn.Linear(3 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )

    def forward(self, fused_features, fine_xyz):
        """
        Args:
            fused_features: [batch, input_dim]
            fine_xyz: [batch, num_points, 3]
        Returns:
            colors: [batch, num_points, 3] (RGB trong [0, 1])
        """
        batch = fused_features.size(0)
        num_points = fine_xyz.size(1)

        # Global → per-point features
        feat = self.feat_proj(fused_features)
        feat = feat.view(batch, num_points, 64)

        # Concat xyz + features → predict color
        point_feat = torch.cat([fine_xyz, feat], dim=-1)  # [B, N, 3+64]
        colors = self.color_mlp(point_feat)  # [B, N, 3]

        return colors


class PointCloudDecoder(nn.Module):
    """
    Full 2-stage decoder: Coarse → Fine + Color.
    """

    def __init__(self, input_dim=512, coarse_points=256, fine_points=2048):
        super().__init__()

        self.coarse_points = coarse_points
        self.fine_points = fine_points
        self.fold_ratio = fine_points // coarse_points

        self.coarse_decoder = CoarseDecoder(input_dim, coarse_points)
        self.fine_decoder = FineDecoder(input_dim, coarse_points, self.fold_ratio)
        self.color_head = ColorHead(input_dim, fine_points)

    def forward(self, fused_features):
        """
        Args:
            fused_features: [batch, input_dim]
        Returns:
            dict với:
                - coarse_xyz: [batch, coarse_points, 3]
                - fine_xyz: [batch, fine_points, 3]
                - colors: [batch, fine_points, 3]
                - point_cloud: [batch, fine_points, 6] (XYZ + RGB)
        """
        coarse_xyz = self.coarse_decoder(fused_features)
        fine_xyz = self.fine_decoder(fused_features, coarse_xyz)
        colors = self.color_head(fused_features, fine_xyz)

        point_cloud = torch.cat([fine_xyz, colors], dim=-1)  # [B, N, 6]

        return {
            "coarse_xyz": coarse_xyz,
            "fine_xyz": fine_xyz,
            "colors": colors,
            "point_cloud": point_cloud,
        }
