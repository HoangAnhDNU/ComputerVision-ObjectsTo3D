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
        num_views=1,
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
        self.num_views = num_views

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

        # Đường dẫn ABO
        self.pc_dir = os.path.join(data_dir, "pointclouds")
        self.pt_dir = os.path.join(data_dir, "pointclouds_pt")
        # Đường dẫn ShapeNet (.ply nằm trong ShapeNet_pcs/, .pt nằm trong pointclouds_pt/)
        self.sn_pc_dir = os.path.join(data_dir, "shapenet", "ShapeNet_pcs")
        self.sn_pt_dir = os.path.join(data_dir, "shapenet", "pointclouds_pt")
        # Ưu tiên ảnh đã lọc background (MatAlpha mask), fallback ảnh gốc
        masked_dir = os.path.join(data_dir, "rendered_images_masked")
        raw_dir = os.path.join(data_dir, "rendered_images")
        if os.path.exists(masked_dir) and len(os.listdir(masked_dir)) > 0:
            self.img_dir = masked_dir
        else:
            self.img_dir = raw_dir

        # Load DiffuRank best views cache (nếu có)
        cache_path = os.path.join(data_dir, "best_views_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.best_views_cache = json.load(f)
            print(f"[Cap3DDataset] DiffuRank cache: {len(self.best_views_cache)} objects")
        else:
            self.best_views_cache = {}

        print(f"[Cap3DDataset] Split={split}, Objects={len(self.object_ids)}, Views={num_views}")

    def __len__(self):
        return len(self.object_ids)

    def _load_point_cloud(self, obj_id):
        """Load point cloud từ .pt (nhanh) hoặc .ply (fallback), trả về [N, 6] (XYZ + RGB).
        Tìm lần lượt: ABO .pt → ShapeNet .pt → ABO .ply → ShapeNet .ply
        """
        # Ưu tiên .pt format (nhanh hơn 5-10x) - kiểm tra ABO rồi ShapeNet
        for pt_dir in [self.pt_dir, self.sn_pt_dir]:
            pt_path = os.path.join(pt_dir, f"{obj_id}.pt")
            if os.path.exists(pt_path):
                pc = torch.load(pt_path, map_location='cpu', weights_only=True).numpy()
                # .pt format: [:, :3] = XYZ, [:, 3:] = RGB normalized [-1,1] → convert về [0,1]
                pc[:, 3:] = (pc[:, 3:] + 1.0) / 2.0
                return pc

        # Fallback: đọc .ply bằng trimesh - kiểm tra ABO rồi ShapeNet
        ply_path = None
        for pc_dir in [self.pc_dir, self.sn_pc_dir]:
            candidate = os.path.join(pc_dir, f"{obj_id}.ply")
            if os.path.exists(candidate):
                ply_path = candidate
                break

        if ply_path is None:
            raise FileNotFoundError(f"Không tìm thấy point cloud cho: {obj_id}")
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

    def _get_image_files(self, obj_img_dir):
        """Liệt kê file ảnh RGB (bỏ depth + MatAlpha)."""
        if not os.path.exists(obj_img_dir):
            return []
        return sorted([
            f for f in os.listdir(obj_img_dir)
            if f.endswith(".png") and not f.startswith(".")
            and "_depth" not in f and "_MatAlpha" not in f
        ])

    def _load_image(self, obj_id):
        """Load 1 ảnh (DiffuRank-inspired: best view nếu có cache, ngẫu nhiên nếu không)."""
        obj_img_dir = os.path.join(self.img_dir, obj_id)
        image_files = self._get_image_files(obj_img_dir)

        if not image_files:
            return Image.new("RGB", (224, 224), (0, 0, 0))

        if self.is_train:
            # DiffuRank: dùng best view nếu có cache
            if obj_id in self.best_views_cache:
                best = self.best_views_cache[obj_id]
                chosen = best[0] if best[0] in set(image_files) else random.choice(image_files)
            else:
                chosen = random.choice(image_files)
        else:
            chosen = image_files[0]

        return Image.open(os.path.join(obj_img_dir, chosen)).convert("RGB")

    def _load_multi_view(self, obj_id):
        """Load K views (DiffuRank-inspired), trả về list K PIL Images."""
        obj_img_dir = os.path.join(self.img_dir, obj_id)
        image_files = self._get_image_files(obj_img_dir)

        if not image_files:
            return [Image.new("RGB", (224, 224), (0, 0, 0))] * self.num_views

        if self.is_train:
            if obj_id in self.best_views_cache:
                # DiffuRank: ưu tiên các views có CLIP-caption score cao nhất
                files_set = set(image_files)
                best = [f for f in self.best_views_cache[obj_id] if f in files_set]
                extra = [f for f in image_files if f not in set(best)]
                random.shuffle(extra)
                ordered = best + extra
                chosen = ordered[:self.num_views]
            else:
                chosen = random.sample(image_files, min(self.num_views, len(image_files)))
        else:
            # Val/test: chọn đều theo bước
            step = max(1, len(image_files) // self.num_views)
            chosen = image_files[::step][:self.num_views]

        # Pad nếu chưa đủ K views
        while len(chosen) < self.num_views:
            chosen.append(chosen[-1])

        return [Image.open(os.path.join(obj_img_dir, f)).convert("RGB") for f in chosen]

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

        # Load image(s)
        if self.num_views > 1:
            images = self._load_multi_view(obj_id)
            image = torch.stack([self.image_transform(img) for img in images])  # [K, 3, 224, 224]
        else:
            image = self.image_transform(self._load_image(obj_id))  # [3, 224, 224]

        # Get caption
        caption = self._get_caption(obj_id)

        return {
            "image": image,
            "caption": caption,
            "point_cloud": point_cloud,
            "object_id": obj_id,
        }


def build_dataloader(data_dir, split, batch_size=16, num_workers=4, num_points=2048, image_size=224, max_objects=0, num_views=1):
    """Tạo DataLoader cho split tương ứng."""
    from torch.utils.data import DataLoader

    dataset = Cap3DDataset(
        data_dir=data_dir,
        split=split,
        num_points=num_points,
        image_size=image_size,
        num_views=num_views,
    )

    # Giới hạn số objects nếu max_objects > 0 (cho CPU training)
    if max_objects > 0 and len(dataset.object_ids) > max_objects:
        dataset.object_ids = dataset.object_ids[:max_objects]
        print(f"[Cap3DDataset] Limited to {max_objects} objects (CPU mode)")

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
