"""
PyTorch Dataset cho Cap3D ABO.
Load ảnh, text, point cloud cho mỗi object.
"""

import json
import os
import random

import numpy as np
import torch
import trimesh
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import ImageTransform, PointCloudTransform, TextTransform, subsample_point_cloud


class Cap3DDataset(Dataset):
    """
    Dataset trả về (image_tensor, text_tokens, point_cloud_gt) cho mỗi object.
    """

    def __init__(
        self,
        data_dir,
        split="train",
        num_points=2048,
        image_size=224,
    ):
        """
        Args:
            data_dir: đường dẫn Cap3D_data/
            split: "train", "val", hoặc "test"
            num_points: số điểm point cloud đầu ra
            image_size: kích thước ảnh đầu vào
        """
        self.data_dir = data_dir
        self.split = split
        self.num_points = num_points
        self.is_train = split == "train"

        # Load object IDs từ split file
        split_path = os.path.join(data_dir, "splits", f"{split}.txt")
        with open(split_path, "r") as f:
            self.object_ids = [line.strip() for line in f if line.strip()]

        # Load captions
        captions_path = os.path.join(data_dir, "captions.json")
        with open(captions_path, "r", encoding="utf-8") as f:
            self.captions = json.load(f)

        # Transforms
        self.image_transform = ImageTransform(image_size=image_size, is_train=self.is_train)
        self.pc_transform = PointCloudTransform(is_train=self.is_train)
        self.text_transform = TextTransform()

        # Đường dẫn
        self.pc_dir = os.path.join(data_dir, "pointclouds")
        self.img_dir = os.path.join(data_dir, "rendered_images")

        print(f"[Cap3DDataset] Split={split}, Objects={len(self.object_ids)}")

    def __len__(self):
        return len(self.object_ids)

    def _load_point_cloud(self, obj_id):
        """Load point cloud từ .ply file, trả về [N, 6] (XYZ + RGB)."""
        ply_path = os.path.join(self.pc_dir, f"{obj_id}.ply")
        mesh = trimesh.load(ply_path)

        # Trích xuất vertices làm point cloud
        if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
            xyz = np.array(mesh.vertices, dtype=np.float32)
            # Lấy màu từ vertex colors nếu có
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                vc = mesh.visual.vertex_colors
                if vc is not None and len(vc) == len(xyz):
                    rgb = np.array(vc[:, :3], dtype=np.float32) / 255.0
                else:
                    rgb = np.ones_like(xyz) * 0.5
            else:
                rgb = np.ones_like(xyz) * 0.5
        else:
            # Fallback: thử đọc như PointCloud
            xyz = np.zeros((1, 3), dtype=np.float32)
            rgb = np.ones((1, 3), dtype=np.float32) * 0.5

        point_cloud = np.concatenate([xyz, rgb], axis=1)  # [N, 6]

        # Normalize XYZ về [-1, 1]
        xyz_min = point_cloud[:, :3].min(axis=0)
        xyz_max = point_cloud[:, :3].max(axis=0)
        center = (xyz_min + xyz_max) / 2.0
        scale = (xyz_max - xyz_min).max() / 2.0
        if scale < 1e-8:
            scale = 1.0
        point_cloud[:, :3] = (point_cloud[:, :3] - center) / scale

        return point_cloud

    def _load_image(self, obj_id):
        """Load 1 ảnh ngẫu nhiên từ 20 góc nhìn, trả về PIL Image."""
        obj_img_dir = os.path.join(self.img_dir, obj_id)
        if not os.path.exists(obj_img_dir):
            # Trả về ảnh đen nếu không có ảnh
            return Image.new("RGB", (224, 224), (0, 0, 0))

        # Liệt kê các file ảnh .png (loại bỏ _depth.png và _MatAlpha.png)
        image_files = sorted([
            f for f in os.listdir(obj_img_dir)
            if f.endswith(".png") and not f.startswith(".")
            and "_depth" not in f and "_MatAlpha" not in f
        ])

        if len(image_files) == 0:
            return Image.new("RGB", (224, 224), (0, 0, 0))

        # Chọn ngẫu nhiên 1 ảnh (train) hoặc ảnh đầu tiên (val/test)
        if self.is_train:
            chosen = random.choice(image_files)
        else:
            chosen = image_files[0]

        img_path = os.path.join(obj_img_dir, chosen)
        return Image.open(img_path).convert("RGB")

    def _get_caption(self, obj_id):
        """Lấy caption cho object."""
        caps = self.captions.get(obj_id, {})
        return self.text_transform(caps)

    def __getitem__(self, idx):
        try:
            return self._load_sample(idx)
        except Exception as e:
            print(f"[WARNING] Lỗi load sample {idx} ({self.object_ids[idx]}): {e}")
            # Fallback: thử sample khác
            return self._load_sample(random.randint(0, len(self) - 1))

    def _load_sample(self, idx):
        obj_id = self.object_ids[idx]

        # Load point cloud
        point_cloud = self._load_point_cloud(obj_id)
        point_cloud = subsample_point_cloud(point_cloud, self.num_points)
        point_cloud = self.pc_transform(point_cloud)
        point_cloud = torch.from_numpy(point_cloud).float()  # [num_points, 6]

        # Load image
        image = self._load_image(obj_id)
        image = self.image_transform(image)  # [3, 224, 224]

        # Get caption
        caption = self._get_caption(obj_id)

        return {
            "image": image,
            "caption": caption,
            "point_cloud": point_cloud,
            "object_id": obj_id,
        }


def build_dataloader(data_dir, split, batch_size=16, num_workers=4, num_points=2048, image_size=224):
    """Tạo DataLoader cho split tương ứng."""
    from torch.utils.data import DataLoader

    dataset = Cap3DDataset(
        data_dir=data_dir,
        split=split,
        num_points=num_points,
        image_size=image_size,
    )

    shuffle = split == "train"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )

    return dataloader
