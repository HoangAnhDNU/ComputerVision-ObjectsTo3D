"""
CLIP-based Image và Text Encoders.
Sử dụng open-clip-torch pretrained.
"""

import contextlib

import torch
import torch.nn as nn
import open_clip


class CLIPEncoders(nn.Module):
    """
    Wrapper cho CLIP ViT-B/32 pretrained.
    Cung cấp image encoder và text encoder.
    """

    def __init__(self, clip_model="ViT-B-32", pretrained="laion2b_s34b_b79k", freeze_clip=True):
        super().__init__()

        self.clip_model_name = clip_model
        self.freeze_clip = freeze_clip

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model)

        # Embedding dimension
        self.embed_dim = self.model.text_projection.shape[1] if hasattr(self.model, 'text_projection') and self.model.text_projection is not None else 512

        # Freeze CLIP nếu cần
        if freeze_clip:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def encode_image(self, image):
        """
        Encode ảnh thành feature vector.

        Args:
            image: [batch, 3, 224, 224]
        Returns:
            image_features: [batch, embed_dim]
        """
        if self.freeze_clip:
            with torch.no_grad():
                features = self.model.encode_image(image)
        else:
            features = self.model.encode_image(image)
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # L2 normalize
        return features

    def encode_image_grid(self, image):
        """
        Extract spatial grid features từ CLIP ViT.
        Trả về per-patch features thay vì single CLS token.
        Dùng cho Transformer decoder (cross-attention).

        Args:
            image: [batch, 3, 224, 224]
        Returns:
            grid_features: [batch, num_patches, embed_dim]
                           num_patches = (224/patch_size)² = 49 cho ViT-B-32
        """
        try:
            return self._extract_grid_features(image)
        except Exception:
            # Fallback: repeat CLS token nếu internal API thay đổi
            cls = self.encode_image(image)
            return cls.unsqueeze(1).expand(-1, 49, -1)

    def _extract_grid_features(self, image):
        """Extract grid features bằng cách chạy visual transformer thủ công."""
        visual = self.model.visual

        ctx = torch.no_grad() if self.freeze_clip else contextlib.nullcontext()
        with ctx:
            # Patch embedding
            x = visual.conv1(image)  # [B, width, H/P, W/P]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, N_patches]
            x = x.permute(0, 2, 1)  # [B, N_patches, width]

            # Add CLS token
            cls_emb = visual.class_embedding.to(x.dtype)
            cls_tokens = cls_emb.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+N, width]

            # Position embedding
            x = x + visual.positional_embedding.to(x.dtype)

            # Patch dropout (nếu có trong phiên bản mới)
            if hasattr(visual, 'patch_dropout') and visual.patch_dropout is not None:
                x = visual.patch_dropout(x)

            x = visual.ln_pre(x)
            x = visual.transformer(x)  # [B, 1+N, width_internal]
            x = visual.ln_post(x)

            # Grid tokens (bỏ CLS ở index 0)
            grid = x[:, 1:, :]  # [B, N_patches, width_internal]

            # Project xuống embed_dim (cùng projection như CLS token)
            if hasattr(visual, 'proj') and visual.proj is not None:
                if isinstance(visual.proj, nn.Linear):
                    grid = visual.proj(grid)
                else:
                    grid = grid @ visual.proj  # [B, N_patches, embed_dim]

        return grid.float()

    def encode_text(self, text_tokens):
        """
        Encode text tokens thành feature vector.

        Args:
            text_tokens: [batch, seq_len] (đã tokenize)
        Returns:
            text_features: [batch, embed_dim]
        """
        if self.freeze_clip:
            with torch.no_grad():
                features = self.model.encode_text(text_tokens)
        else:
            features = self.model.encode_text(text_tokens)
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # L2 normalize
        return features

    def tokenize(self, texts):
        """
        Tokenize danh sách text strings.

        Args:
            texts: list of str
        Returns:
            tokens: [batch, seq_len] tensor
        """
        return self.tokenizer(texts)

    def unfreeze_last_layers(self, num_layers=2):
        """Unfreeze N layer cuối của CLIP để fine-tune."""
        # Unfreeze visual transformer blocks cuối
        visual_blocks = self.model.visual.transformer.resblocks
        for block in list(visual_blocks)[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze text transformer blocks cuối
        text_blocks = self.model.transformer.resblocks
        for block in list(text_blocks)[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True

        self.freeze_clip = False
        self.model.train()
