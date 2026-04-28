"""
CLIP-based Image và Text Encoders.
Sử dụng open-clip-torch pretrained.
"""

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
