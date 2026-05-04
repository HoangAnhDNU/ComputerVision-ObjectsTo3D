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
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
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
        # Xoay GT point cloud sẽ phá vỡ alignment image↔️3D

        # Random scale (hơi mạnh hơn)
        scale = random.uniform(0.85, 1.15)
        xyz *= scale

        # Random jitter (hơi mạnh hơn)
        jitter = np.random.randn(*xyz.shape) * 0.03
        xyz += jitter

        # Random point dropout: bỏ ngẫu nhiên 5-15% points
        # Buộc model học tạo point cloud đầy đủ dù GT bị thiếu
        if random.random() < 0.3:
            drop_rate = random.uniform(0.05, 0.15)
            n = len(xyz)
            keep = max(int(n * (1 - drop_rate)), 1)
            indices = np.random.choice(n, keep, replace=False)
            xyz = xyz[indices]
            rgb = rgb[indices]
            # Pad lại số điểm bằng oversample
            if len(xyz) < n:
                pad_idx = np.random.choice(len(xyz), n - len(xyz), replace=True)
                xyz = np.concatenate([xyz, xyz[pad_idx]], axis=0)
                rgb = np.concatenate([rgb, rgb[pad_idx]], axis=0)

        # Random color noise
        if random.random() < 0.3:
            rgb += np.random.randn(*rgb.shape).astype(np.float32) * 0.05

        # Clip RGB về [0, 1]
        rgb = np.clip(rgb, 0.0, 1.0)

        return np.concatenate([xyz, rgb], axis=1)


def farthest_point_sampling(xyz, num_points):
    """
    Farthest Point Sampling: chọn điểm xa nhất lần lượt.
    Giữ coverage tốt hơn random sample → ít lỗ chỗ hơn.

    Args:
        xyz: numpy array [N, 3]
        num_points: số điểm cần chọn
    Returns:
        indices: numpy array [num_points]
    """
    N = len(xyz)
    if N <= num_points:
        return np.arange(N)

    indices = np.zeros(num_points, dtype=np.int64)
    distances = np.full(N, np.inf)
    # Bắt đầu từ điểm ngẫu nhiên
    current = np.random.randint(N)
    indices[0] = current

    for i in range(1, num_points):
        dist = np.sum((xyz - xyz[current]) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        current = np.argmax(distances)
        indices[i] = current

    return indices


def subsample_point_cloud(point_cloud, num_points=2048, use_fps=True):
    """
    Subsample point cloud từ N điểm → num_points điểm.
    Nếu N > num_points: dùng FPS (phân bố đều) hoặc random sample.
    Nếu N < num_points: oversample với replacement.

    Args:
        point_cloud: numpy array [N, 6]
        num_points: số điểm đầu ra
        use_fps: dùng Farthest Point Sampling (chậm hơn nhưng coverage tốt hơn)
    Returns:
        numpy array [num_points, 6]
    """
    n = point_cloud.shape[0]
    if n >= num_points:
        if use_fps and n <= 50000:
            indices = farthest_point_sampling(point_cloud[:, :3], num_points)
        else:
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
