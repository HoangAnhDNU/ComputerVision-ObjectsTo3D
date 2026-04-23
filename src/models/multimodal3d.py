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
    """

    def __init__(
        self,
        clip_model="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        freeze_clip=True,
        fusion_type="concat",
        fused_dim=512,
        coarse_points=256,
        fine_points=2048,
    ):
        super().__init__()

        # Encoders
        self.encoders = CLIPEncoders(
            clip_model=clip_model,
            pretrained=pretrained,
            freeze_clip=freeze_clip,
        )
        embed_dim = self.encoders.embed_dim

        # Fusion
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

        # Decoder
        self.decoder = PointCloudDecoder(
            input_dim=fused_dim,
            coarse_points=coarse_points,
            fine_points=fine_points,
        )

        self.fusion_type = fusion_type

    def forward(self, image, text_tokens):
        """
        Args:
            image: [batch, 3, 224, 224]
            text_tokens: [batch, seq_len] (CLIP tokenized)
        Returns:
            dict với point_cloud [batch, fine_points, 6] và intermediate outputs
        """
        # Encode
        image_features = self.encoders.encode_image(image)
        text_features = self.encoders.encode_text(text_tokens)

        # Fuse
        fused_features = self.fusion(image_features, text_features)

        # Decode
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
        fusion_type=model_cfg.get("fusion_type", "concat"),
        fused_dim=model_cfg.get("fused_dim", 512),
        coarse_points=model_cfg.get("coarse_points", 256),
        fine_points=model_cfg.get("fine_points", 2048),
    )
