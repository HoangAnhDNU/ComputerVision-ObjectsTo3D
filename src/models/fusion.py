"""
Multimodal Fusion module.
Kết hợp image features và text features.
"""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    Fusion đơn giản: concatenate image + text features → MLP projection.
    """

    def __init__(self, image_dim=512, text_dim=512, fused_dim=512):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, image_features, text_features):
        """
        Args:
            image_features: [batch, image_dim]
            text_features: [batch, text_dim]
        Returns:
            fused_features: [batch, fused_dim]
        """
        combined = torch.cat([image_features, text_features], dim=-1)
        return self.fusion(combined)


class CrossAttentionFusion(nn.Module):
    """
    Fusion nâng cao: Cross-attention giữa image tokens và text tokens.
    Kết hợp với concat fusion cho feature vector cuối.
    """

    def __init__(self, image_dim=512, text_dim=512, fused_dim=512, num_heads=8, num_layers=2):
        super().__init__()

        # Project để đồng nhất dimension
        self.image_proj = nn.Linear(image_dim, fused_dim)
        self.text_proj = nn.Linear(text_dim, fused_dim)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(fused_dim) for _ in range(num_layers)
        ])

        # FFN sau mỗi cross-attn
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fused_dim, fused_dim * 2),
                nn.GELU(),
                nn.Linear(fused_dim * 2, fused_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(fused_dim) for _ in range(num_layers)
        ])

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, image_features, text_features):
        """
        Args:
            image_features: [batch, image_dim]
            text_features: [batch, text_dim]
        Returns:
            fused_features: [batch, fused_dim]
        """
        batch = image_features.size(0)

        # Project
        img = self.image_proj(image_features).unsqueeze(1)  # [B, 1, D]
        txt = self.text_proj(text_features).unsqueeze(1)    # [B, 1, D]

        # Cross-attention: image attend to text, text attend to image
        for cross_attn, ln, ffn, ffn_ln in zip(
            self.cross_attn_layers, self.layer_norms, self.ffn_layers, self.ffn_norms
        ):
            # Stack thành sequence [B, 2, D]
            seq = torch.cat([img, txt], dim=1)

            # Self-attention trên 2 tokens
            attn_out, _ = cross_attn(seq, seq, seq)
            seq = ln(seq + attn_out)

            # FFN
            ffn_out = ffn(seq)
            seq = ffn_ln(seq + ffn_out)

            img = seq[:, 0:1, :]
            txt = seq[:, 1:2, :]

        img = img.squeeze(1)  # [B, D]
        txt = txt.squeeze(1)  # [B, D]

        # Kết hợp
        fused = self.output_proj(torch.cat([img, txt], dim=-1))
        return fused
