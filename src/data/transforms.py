"""
Augmentation transforms cho image, point cloud, text.
"""

import random

import numpy as np
import torch
import torchvision.transforms as T


class ImageTransform:
    """Transform cho ảnh 2D đầu vào."""

    def __init__(self, image_size=224, is_train=True):
        self.image_size = image_size
        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])

    def __call__(self, image):
        return self.transform(image)


class PointCloudTransform:
    """Augmentation cho point cloud 3D."""

    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, point_cloud):
        """
        Args:
            point_cloud: numpy array [N, 6] (XYZ + RGB)
        Returns:
            point_cloud: numpy array [N, 6]
        """
        if not self.is_train:
            return point_cloud

        xyz = point_cloud[:, :3].copy()
        rgb = point_cloud[:, 3:].copy()

        # NOTE: Không random rotation vì ảnh rendered từ góc cố định
        # Xoay GT point cloud sẽ phá vỡ alignment image↔3D

        # Random scale
        scale = random.uniform(0.9, 1.1)
        xyz *= scale

        # Random jitter
        jitter = np.random.randn(*xyz.shape) * 0.02
        xyz += jitter

        # Clip RGB về [0, 1]
        rgb = np.clip(rgb, 0.0, 1.0)

        return np.concatenate([xyz, rgb], axis=1)


def subsample_point_cloud(point_cloud, num_points=2048):
    """
    Subsample point cloud từ N điểm → num_points điểm.
    Nếu N > num_points: random sample.
    Nếu N < num_points: oversample với replacement.

    Args:
        point_cloud: numpy array [N, 6]
        num_points: số điểm đầu ra
    Returns:
        numpy array [num_points, 6]
    """
    n = point_cloud.shape[0]
    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
    else:
        indices = np.random.choice(n, num_points, replace=True)
    return point_cloud[indices]


class TextTransform:
    """Chọn ngẫu nhiên caption (general hoặc compositional)."""

    def __call__(self, captions_dict):
        """
        Args:
            captions_dict: {"general": str, "compositional": str} hoặc tương tự
        Returns:
            caption: str
        """
        if not captions_dict:
            return ""

        available = list(captions_dict.values())
        if len(available) == 1:
            return available[0]
        return random.choice(available)
