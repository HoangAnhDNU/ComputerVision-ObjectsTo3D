"""
Transformer-based Point Cloud Decoder.
Lấy cảm hứng từ Point-E (OpenAI) nhưng đơn giản hơn, phù hợp dataset nhỏ.

Kiến trúc:
- Coarse stage: Learnable queries + Self-Attention + Cross-Attention to CLIP grid features
- Fine stage: FoldingNet upsample + Cross-Attention refinement (no self-attn → tiết kiệm VRAM)

So với MLP decoder cũ:
- CLIP grid features (49 spatial tokens) thay vì 1 vector → giữ thông tin spatial
- Self-attention giữa coarse points → points biết nhau → hình thù rõ hơn
- Cross-attention → mỗi point attend đúng vùng ảnh liên quan
"""

import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    """
    Transformer decoder block.
    Hỗ trợ tùy chọn self-attention và cross-attention.
    """

    def __init__(self, width, heads, use_self_attn=True, use_cross_attn=True):
        super().__init__()

        self.use_self_attn = use_self_attn
        self.use_cross_attn = use_cross_attn

        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
            self.ln_self = nn.LayerNorm(width)

        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
            self.ln_cross = nn.LayerNorm(width)
            self.ln_cond = nn.LayerNorm(width)

        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )
        self.ln_ffn = nn.LayerNorm(width)

    def forward(self, x, cond=None):
        """
        Args:
            x: [B, N, width] - point queries
            cond: [B, M, width] - conditioning tokens (CLIP features)
        Returns:
            x: [B, N, width]
        """
        # Self-attention among points
        if self.use_self_attn:
            x_norm = self.ln_self(x)
            x = x + self.self_attn(x_norm, x_norm, x_norm)[0]

        # Cross-attention to conditioning
        if self.use_cross_attn and cond is not None:
            x_norm = self.ln_cross(x)
            c_norm = self.ln_cond(cond)
            x = x + self.cross_attn(x_norm, c_norm, c_norm)[0]

        # FFN
        x = x + self.ffn(self.ln_ffn(x))

        return x


class TransformerPointDecoder(nn.Module):
    """
    Transformer-based 2-stage point cloud decoder.

    Stage 1 (Coarse): Learnable queries → Self-Attn + Cross-Attn → coarse XYZ
    Stage 2 (Fine):   FoldingNet upsample → Cross-Attn refinement → fine XYZ + RGB

    Args:
        cond_dim: Conditioning dimension (CLIP embed_dim, thường = 512)
        width: Transformer hidden width
        coarse_points: Số coarse points (mặc định 256)
        fine_points: Số fine output points (mặc định 2048)
        coarse_layers: Số Transformer blocks cho coarse stage
        fine_layers: Số cross-attention blocks cho fine stage
        heads: Số attention heads
    """

    def __init__(self, cond_dim=512, width=256, coarse_points=256, fine_points=2048,
                 coarse_layers=4, fine_layers=2, heads=4):
        super().__init__()

        self.coarse_points = coarse_points
        self.fine_points = fine_points
        self.fold_ratio = fine_points // coarse_points
        self.width = width

        # === CONDITIONING ===
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, width),
        )

        # === COARSE STAGE ===
        # Learnable point queries (tương tự DETR object queries)
        self.coarse_queries = nn.Parameter(
            torch.randn(1, coarse_points, width) * 0.02
        )

        self.coarse_blocks = nn.ModuleList([
            DecoderBlock(width, heads, use_self_attn=True, use_cross_attn=True)
            for _ in range(coarse_layers)
        ])
        self.coarse_ln = nn.LayerNorm(width)
        self.coarse_head = nn.Linear(width, 3)

        # === FINE STAGE ===
        # Upsample: mỗi coarse point → fold_ratio fine points
        self.upsample = nn.Sequential(
            nn.Linear(width + 3, width),
            nn.GELU(),
            nn.Linear(width, width * self.fold_ratio),
        )

        # Cross-attention refinement (KHÔNG self-attn → O(N*M) thay vì O(N²))
        self.fine_blocks = nn.ModuleList([
            DecoderBlock(width, heads, use_self_attn=False, use_cross_attn=True)
            for _ in range(fine_layers)
        ])
        self.fine_ln = nn.LayerNorm(width)

        # Output heads
        self.xyz_head = nn.Linear(width, 3)
        self.color_head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 3),
            nn.Sigmoid(),
        )

        self._init_output_heads()

    def _init_output_heads(self):
        """Init nhỏ cho output heads → bắt đầu gần origin, refine dần khi train."""
        nn.init.normal_(self.coarse_head.weight, std=0.01)
        nn.init.zeros_(self.coarse_head.bias)
        nn.init.normal_(self.xyz_head.weight, std=0.01)
        nn.init.zeros_(self.xyz_head.bias)

    def forward(self, image_grid_features, text_features):
        """
        Args:
            image_grid_features: [B, G, cond_dim] - CLIP grid features (G=49 cho ViT-B-32)
            text_features: [B, cond_dim] - CLIP text CLS features
        Returns:
            dict:
                - coarse_xyz: [B, coarse_points, 3]
                - fine_xyz: [B, fine_points, 3]
                - colors: [B, fine_points, 3]
                - point_cloud: [B, fine_points, 6]
        """
        batch = image_grid_features.size(0)

        # Conditioning: [image_grid; text_token] → project to width
        text_token = text_features.unsqueeze(1)  # [B, 1, cond_dim]
        cond_raw = torch.cat([image_grid_features, text_token], dim=1)  # [B, G+1, cond_dim]
        cond = self.cond_proj(cond_raw)  # [B, G+1, width]

        # === COARSE STAGE ===
        queries = self.coarse_queries.expand(batch, -1, -1)  # [B, C, width]

        for block in self.coarse_blocks:
            queries = block(queries, cond)

        coarse_feat = self.coarse_ln(queries)  # [B, C, width]
        coarse_xyz = self.coarse_head(coarse_feat)  # [B, C, 3]

        # === FINE STAGE ===
        # Upsample features
        upsample_in = torch.cat([coarse_feat, coarse_xyz], dim=-1)  # [B, C, width+3]
        fine_feat = self.upsample(upsample_in)  # [B, C, width*fold_ratio]
        fine_feat = fine_feat.view(batch, self.coarse_points, self.fold_ratio, self.width)
        fine_feat = fine_feat.reshape(batch, self.fine_points, self.width)  # [B, N, width]

        # Base positions: expand coarse XYZ
        base_xyz = coarse_xyz.unsqueeze(2).expand(-1, -1, self.fold_ratio, -1)
        base_xyz = base_xyz.reshape(batch, self.fine_points, 3)  # [B, N, 3]

        # Cross-attention refinement
        for block in self.fine_blocks:
            fine_feat = block(fine_feat, cond)

        fine_feat = self.fine_ln(fine_feat)  # [B, N, width]

        # Output: residual XYZ + colors
        xyz_offset = self.xyz_head(fine_feat)  # [B, N, 3]
        fine_xyz = base_xyz + xyz_offset  # [B, N, 3]
        colors = self.color_head(fine_feat)  # [B, N, 3]

        point_cloud = torch.cat([fine_xyz, colors], dim=-1)  # [B, N, 6]

        return {
            "coarse_xyz": coarse_xyz,
            "fine_xyz": fine_xyz,
            "colors": colors,
            "point_cloud": point_cloud,
        }
