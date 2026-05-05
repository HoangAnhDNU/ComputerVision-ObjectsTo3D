"""
Full Multimodal 3D Model.
Kết nối: CLIP Encoders → Fusion → Point Cloud Decoder.
"""

import torch
import torch.nn as nn

from src.models.encoders import CLIPEncoders
from src.models.fusion import ConcatFusion, CrossAttentionFusion
from src.models.decoder import PointCloudDecoder


class Multimodal3DModel(nn.Module):
    """
    Input: (image, text) → Output: point cloud [batch, num_points, 6]

    Hỗ trợ 2 loại decoder:
    - "mlp": ConcatFusion + MLP decoder (cũ, nhẹ)
    - "transformer": Cross-Attention Transformer decoder (mới, chất lượng cao)
    """

    def __init__(
        self,
        clip_model="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        freeze_clip=True,
        # Decoder selection
        decoder_type="mlp",
        # Transformer decoder params
        decoder_width=256,
        decoder_layers=4,
        decoder_heads=4,
        fine_layers=2,
        # MLP decoder params (legacy)
        fusion_type="concat",
        fused_dim=512,
        # Shared
        coarse_points=256,
        fine_points=2048,
    ):
        super().__init__()

        self.decoder_type = decoder_type

        # Encoders
        self.encoders = CLIPEncoders(
            clip_model=clip_model,
            pretrained=pretrained,
            freeze_clip=freeze_clip,
        )
        embed_dim = self.encoders.embed_dim

        if decoder_type == "transformer":
            # Transformer decoder: dùng grid features + cross-attention
            from src.models.transformer_decoder import TransformerPointDecoder
            self.decoder = TransformerPointDecoder(
                cond_dim=embed_dim,
                width=decoder_width,
                coarse_points=coarse_points,
                fine_points=fine_points,
                coarse_layers=decoder_layers,
                fine_layers=fine_layers,
                heads=decoder_heads,
            )
        else:
            # Legacy MLP decoder: ConcatFusion + PointCloudDecoder
            if fusion_type == "cross_attention":
                self.fusion = CrossAttentionFusion(
                    image_dim=embed_dim,
                    text_dim=embed_dim,
                    fused_dim=fused_dim,
                )
            else:
                self.fusion = ConcatFusion(
                    image_dim=embed_dim,
                    text_dim=embed_dim,
                    fused_dim=fused_dim,
                )
            self.decoder = PointCloudDecoder(
                input_dim=fused_dim,
                coarse_points=coarse_points,
                fine_points=fine_points,
            )

        self.fusion_type = fusion_type

    def forward(self, image, text_tokens):
        """
        Args:
            image: [batch, 3, 224, 224]        — single view
                   [batch, K, 3, 224, 224]     — multi-view (K views)
            text_tokens: [batch, seq_len] (CLIP tokenized)
        Returns:
            dict với point_cloud [batch, fine_points, 6] và intermediate outputs
        """
        is_multi_view = image.dim() == 5  # [B, K, 3, 224, 224]

        if self.decoder_type == "transformer":
            if is_multi_view:
                # Multi-view: concat K*49 patches → richer cross-attention context
                image_grid = self.encoders.encode_image_grid_multi(image)  # [B, K*49, D]
            else:
                image_grid = self.encoders.encode_image_grid(image)        # [B, 49, D]
            text_features = self.encoders.encode_text(text_tokens)
            outputs = self.decoder(image_grid, text_features)
        else:
            if is_multi_view:
                # Multi-view: mean pool K views → [B, D]
                image_features = self.encoders.encode_multi_view(image)
            else:
                image_features = self.encoders.encode_image(image)
            text_features = self.encoders.encode_text(text_tokens)
            fused_features = self.fusion(image_features, text_features)
            outputs = self.decoder(fused_features)

        return outputs

    def tokenize(self, texts):
        """Helper để tokenize text."""
        return self.encoders.tokenize(texts)

    def get_num_params(self):
        """Đếm số parameters trainable."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_model(config):
    """
    Tạo model từ config dict.

    Args:
        config: dict với keys như trong configs/default.yaml
    Returns:
        Multimodal3DModel
    """
    model_cfg = config.get("model", {})
    return Multimodal3DModel(
        clip_model=model_cfg.get("clip_model", "ViT-B-32"),
        pretrained=model_cfg.get("pretrained", "laion2b_s34b_b79k"),
        freeze_clip=model_cfg.get("freeze_clip", True),
        decoder_type=model_cfg.get("decoder_type", "mlp"),
        decoder_width=model_cfg.get("decoder_width", 256),
        decoder_layers=model_cfg.get("decoder_layers", 4),
        decoder_heads=model_cfg.get("decoder_heads", 4),
        fine_layers=model_cfg.get("fine_layers", 2),
        fusion_type=model_cfg.get("fusion_type", "concat"),
        fused_dim=model_cfg.get("fused_dim", 512),
        coarse_points=model_cfg.get("coarse_points", 256),
        fine_points=model_cfg.get("fine_points", 2048),
    )
