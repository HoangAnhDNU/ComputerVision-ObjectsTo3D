# Gợi ý cải thiện dự án - Tạo mô hình 3D từ ảnh và mô tả văn bản

> Tài liệu này phân tích dữ liệu có sẵn từ Cap3D dataset và gợi ý cách cải thiện dự án
> với các mức độ khó khác nhau. Mỗi gợi ý đều có giải thích lý do và hướng dẫn thực hiện.

---

## 1. Tổng quan dữ liệu có sẵn từ Cap3D (HuggingFace)

### 1.1 Dữ liệu đang sử dụng: ABO subset

| Dữ liệu | Mô tả | Dung lượng |
|---|---|---|
| `Cap3D_automated_ABO.csv` | Captions cho 7,953 objects ABO (Amazon) | 3.39 MB |
| `PointCloud_zips_ABO/` | Point clouds 16,384 điểm/object (.ply) | ~2 GB |
| `RenderedImage_perobj_zips_ABO/` | Rendered images, 20 views/object (gồm depth, MatAlpha, camera JSON) | ~5 GB |

### 1.2 Dữ liệu có thể lấy thêm

| Dữ liệu | Mô tả | Dung lượng | Khả thi? |
|---|---|---|---|
| `PointCloud_zips_ShapeNet/` | 52,472 objects ShapeNet (.ply, 16,384 điểm) | ~15 GB | ★★★ Rất tốt |
| `Cap3D_automated_ShapeNet.csv` | Captions ShapeNet | 19.5 MB | ★★★ |
| `RenderedImage_perobj_zips_ShapeNet/` | Rendered images ShapeNet | ~30 GB | ★★☆ |
| `PointCloud_pt_zips/` | 1,552,357 objects Objaverse (.pt, load nhanh hơn .ply) | ~40 GB | ★☆☆ Quá lớn |
| `Cap3D_automated_Objaverse_full.csv` | 1,604,291 captions Objaverse+XL | 255 MB | ★☆☆ |
| `misc/ShapELatentCode_zips/` | ShapE latent codes (3D descriptors) | ~20 GB | ★★☆ |

### 1.3 Cấu trúc rendered images (mỗi object)

Dataset cung cấp **nhiều hơn chỉ là ảnh RGB**. Mỗi object có 20 views, mỗi view gồm:

```
{object_id}/
├── 00000.png              # Ảnh RGB rendered
├── 00000_depth.png        # Depth map
├── 00000_MatAlpha.png     # Material Alpha mask (segmentation)
├── 00000.json             # Camera parameters (intrinsic + extrinsic)
├── 00001.png
├── 00001_depth.png
├── 00001_MatAlpha.png
├── 00001.json
├── ...                    # 20 views x 4 files = 80 files/object
└── 00019.json
```

**Hiện tại dự án chỉ dùng `00000.png` (1 view RGB), bỏ qua depth, mask, camera và các view khác.**

---

## 2. Các gợi ý cải thiện

### ======================================================
### GỢI Ý 1: Sử dụng Depth Map làm input bổ sung
### ======================================================

**Độ khó**: ★★☆☆☆ (Dễ)
**Tác động**: ★★★★☆ (Cao)
**Cần train lại**: Có

**Lý do**:
- Depth map chứa thông tin 3D trực tiếp (khoảng cách từ camera đến object)
- CLIP chỉ hiểu 2D appearance, không hiểu chiều sau
- Thêm depth = thêm "gợi ý 3D" cho model, giảm ambiguity (uncertainty)

**Cách thực hiện**:

**Bước 1**: Sửa `src/data/dataset.py` - load thêm depth map

```python
# Trong _load_image(), thay vì chỉ load 1 ảnh RGB:
def _load_image_and_depth(self, obj_id):
    obj_img_dir = os.path.join(self.img_dir, obj_id)
    image_files = sorted([
        f for f in os.listdir(obj_img_dir)
        if f.endswith(".png") and "_depth" not in f and "_MatAlpha" not in f
    ])

    if self.is_train:
        chosen = random.choice(image_files)
    else:
        chosen = image_files[0]

    # Load RGB
    rgb_path = os.path.join(obj_img_dir, chosen)
    rgb = Image.open(rgb_path).convert("RGB")

    # Load depth tương ứng
    depth_name = chosen.replace(".png", "_depth.png")
    depth_path = os.path.join(obj_img_dir, depth_name)
    if os.path.exists(depth_path):
        depth = Image.open(depth_path).convert("L")  # Grayscale
    else:
        depth = Image.new("L", (224, 224), 128)

    return rgb, depth
```

**Bước 2**: Sửa `src/models/encoders.py` - thêm depth encoder

```python
class DepthEncoder(nn.Module):
    """Encoder đơn giản cho depth map."""
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),  # 112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # 56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                       # 1x1
            nn.Flatten(),
            nn.Linear(256, out_dim),
        )

    def forward(self, depth):
        """depth: [B, 1, 224, 224] -> [B, out_dim]"""
        return self.conv(depth)
```

**Bước 3**: Sửa `src/models/multimodal3d.py` - fusion thêm depth

```python
# Trong forward():
image_feat = self.clip.encode_image(image)       # [B, 512]
text_feat = self.clip.encode_text(text)           # [B, 512]
depth_feat = self.depth_encoder(depth)            # [B, 512]

# Fusion: concat 3 modalities
fused = torch.cat([image_feat, text_feat, depth_feat], dim=-1)  # [B, 1536]
fused = self.fusion_proj(fused)  # [B, fused_dim]
```

**Bước 4**: Sửa dataset __getitem__ để trả về depth

```python
return {
    "image": image_tensor,
    "depth": depth_tensor,      # Thêm dòng này
    "caption": caption,
    "point_cloud": point_cloud,
    "object_id": obj_id,
}
```

---

### ======================================================
### GỢI Ý 2: Multi-View Input (Nhiều góc nhìn)
### ======================================================

**Độ khó**: ★★★☆☆ (Trung bình)
**Tác động**: ★★★★★ (Rất cao)
**Cần train lại**: Có

**Lý do**:
- 1 ảnh 2D chỉ thể hiện 1 mặt của object -> mất nửa phía sau
- Dataset đã có 20 views/object -> chỉ cần load thêm
- Multi-view + attention = model hiểu được cấu trúc 3D tốt hơn nhiều

**Cách thực hiện**:

**Bước 1**: Sửa `src/data/dataset.py` - load K views

```python
def _load_multi_view(self, obj_id, num_views=4):
    """Load nhiều góc nhìn cho 1 object."""
    obj_img_dir = os.path.join(self.img_dir, obj_id)
    image_files = sorted([
        f for f in os.listdir(obj_img_dir)
        if f.endswith(".png") and "_depth" not in f and "_MatAlpha" not in f
    ])

    if len(image_files) == 0:
        return [Image.new("RGB", (224, 224), (0, 0, 0))] * num_views

    if self.is_train:
        # Random chọn K views
        chosen = random.sample(image_files, min(num_views, len(image_files)))
    else:
        # Chọn đều (views 0, 5, 10, 15 nếu có 20 views)
        step = max(1, len(image_files) // num_views)
        chosen = image_files[::step][:num_views]

    # Padding nếu thiếu
    while len(chosen) < num_views:
        chosen.append(chosen[-1])

    images = []
    for fname in chosen:
        img_path = os.path.join(obj_img_dir, fname)
        images.append(Image.open(img_path).convert("RGB"))

    return images
```

**Bước 2**: Sửa `src/models/encoders.py` - encode nhiều views

```python
def encode_multi_view(self, images):
    """
    Encode nhiều views và aggregate.
    Args:
        images: [B, K, 3, 224, 224] - K views
    Returns:
        aggregated: [B, embed_dim]
        grid_features: [B, K*num_patches, embed_dim]
    """
    B, K = images.shape[:2]
    # Flatten -> encode tất cả views cùng lúc
    flat = images.view(B * K, *images.shape[2:])  # [B*K, 3, 224, 224]
    features = self.encode_image(flat)              # [B*K, embed_dim]
    features = features.view(B, K, -1)              # [B, K, embed_dim]

    # Aggregate: attention-weighted average
    # Hoặc đơn giản: mean pooling
    aggregated = features.mean(dim=1)               # [B, embed_dim]

    # Grid features cho transformer decoder
    grid = self._extract_grid_features(flat)         # [B*K, 49, embed_dim]
    grid = grid.view(B, K * 49, -1)                  # [B, K*49, embed_dim]

    return aggregated, grid
```

**Bước 3**: Sửa `__getitem__` để trả về multi-view tensor

```python
# Trong _load_sample():
images = self._load_multi_view(obj_id, num_views=4)
image_tensors = torch.stack([self.image_transform(img) for img in images])  # [K, 3, 224, 224]

return {
    "images": image_tensors,      # [K, 3, 224, 224] thay vì [3, 224, 224]
    "caption": caption,
    "point_cloud": point_cloud,
    "object_id": obj_id,
}
```

**Lưu ý**:
- VRAM tăng K lần cho image encoding -> giảm batch_size hoặc giảm K
- K=4 views là đủ tốt, K=8 tốt hơn nhưng cần nhiều VRAM
- Demo vẫn dùng 1 ảnh (tự động repeat thành K views)

---

### ======================================================
### GỢI Ý 3: Sử dụng Camera Parameters cho View-Aware Encoding
### ======================================================

**Độ khó**: ★★★★☆ (Khó)
**Tác động**: ★★★★☆ (Cao)
**Cần train lại**: Có

**Lý do**:
- Dataset cung cấp camera intrinsic + extrinsic cho mỗi view (file .json)
- Model biết camera ở đâu -> hiểu được hình học 3D tốt hơn
- Kết hợp với multi-view -> biết chính xác góc nhìn của mỗi ảnh

**Cấu trúc camera JSON** (từ example_zipfile):

```json
{
    "camera_angle_x": 0.8575,
    "camera_angle_y": 0.8575,
    "transform_matrix": [
        [0.707, -0.408, 0.577, 2.0],
        [0.707, 0.408, -0.577, -2.0],
        [0.0, 0.816, 0.577, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
}
```

**Cách thực hiện**:

```python
# Thêm camera encoding vào model
class CameraPoseEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        # Input: 4x4 transform matrix flatten = 16 values
        self.mlp = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, camera_matrix):
        """camera_matrix: [B, 4, 4] -> [B, out_dim]"""
        flat = camera_matrix.view(camera_matrix.size(0), -1)  # [B, 16]
        return self.mlp(flat)

# Trong forward:
cam_feat = self.camera_encoder(camera_matrix)  # [B, 128]
# Concat với image features trước khi đưa vào decoder
```

---

### ======================================================
### GỢI Ý 4: Tăng dữ liệu training với ShapeNet subset
### ======================================================

**Độ khó**: ★★☆☆☆ (Dễ)
**Tác động**: ★★★★☆ (Cao)
**Cần train lại**: Có

**Lý do**:
- ABO chỉ có ~7,953 objects -> ít cho deep learning
- ShapeNet có 52,472 objects -> gấp 7 lần
- ShapeNet đa dạng categories: chair, car, plane, lamp, table, ...
- Dữ liệu cùng định dạng (.ply, rendered images, captions) -> gộp chung dễ dàng

**Cách thực hiện**:

**Bước 1**: Tải ShapeNet data

```bash
# Tải captions
hf download tiange/Cap3D Cap3D_automated_ShapeNet.csv --repo-type dataset --local-dir ./Cap3D_data

# Tải point clouds
hf download tiange/Cap3D PointCloud_zips_ShapeNet/compressed_pcs_00.zip --repo-type dataset --local-dir ./Cap3D_data
hf download tiange/Cap3D PointCloud_zips_ShapeNet/compressed_pcs_01.zip --repo-type dataset --local-dir ./Cap3D_data

# Tải rendered images
hf download tiange/Cap3D RenderedImage_perobj_zips_ShapeNet/compressed_imgs_perobj_00.zip --repo-type dataset --local-dir ./Cap3D_data
hf download tiange/Cap3D RenderedImage_perobj_zips_ShapeNet/compressed_imgs_perobj_01.zip --repo-type dataset --local-dir ./Cap3D_data
hf download tiange/Cap3D RenderedImage_perobj_zips_ShapeNet/compressed_imgs_perobj_02.zip --repo-type dataset --local-dir ./Cap3D_data
```

**Bước 2**: Sửa `src/data/prepare_data.py` - thêm ShapeNet parsing

```python
# Parse ShapeNet captions (tương tự ABO)
shapenet_csv = os.path.join(data_dir, "Cap3D_automated_ShapeNet.csv")
if os.path.exists(shapenet_csv):
    df = pd.read_csv(shapenet_csv, header=None)
    for _, row in df.iterrows():
        obj_id = row[0]
        caption = row[1]
        captions[obj_id] = {"general": caption}
```

**Bước 3**: Tạo splits mới gồm cả ABO + ShapeNet

```python
# Gộp tất cả object IDs
all_ids = abo_ids + shapenet_ids  # ~60,000 objects
random.shuffle(all_ids)
train = all_ids[:int(0.8 * len(all_ids))]
val = all_ids[int(0.8 * len(all_ids)):int(0.9 * len(all_ids))]
test = all_ids[int(0.9 * len(all_ids)):]
```

**Lưu ý**:
- ShapeNet dung lượng lớn (~45 GB tổng) -> cần nhiều disk trên Colab
- Có thể chỉ tải 1 phần (ví dụ 1/3 ShapeNet)
- Rendered images ShapeNet cũng có 20 views + depth + camera

---

### ======================================================
### GỢI Ý 5: Sử dụng Mask (MatAlpha) để lọc background
### ======================================================

**Độ khó**: ★☆☆☆☆ (Rất dễ)
**Tác động**: ★★★☆☆ (Trung bình)
**Cần train lại**: Có (nhưng thay đổi nhỏ)

**Lý do**:
- Ảnh rendered có background -> CLIP có thể bị nhiễu bởi background
- MatAlpha mask chỉ rõ vùng nào là object, vùng nào là background
- Lọc background giúp model tập trung vào object

**Cách thực hiện**:

```python
# Trong _load_image():
def _load_image(self, obj_id):
    # ... load RGB image như bình thường ...

    # Load mask
    mask_name = chosen.replace(".png", "_MatAlpha.png")
    mask_path = os.path.join(obj_img_dir, mask_name)
    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask) > 128  # Binary mask

        # Áp dụng mask: set background = white
        img_array = np.array(rgb)
        img_array[~mask] = [255, 255, 255]
        rgb = Image.fromarray(img_array)

    return rgb
```

**Lưu ý**: Background trắng phù hợp với CLIP (pretrained trên ảnh có background trắng/trống)

---

### ======================================================
### GỢI Ý 6: Cải thiện Loss với Normal Consistency
### ======================================================

**Độ khó**: ★★★☆☆ (Trung bình)
**Tác động**: ★★★☆☆ (Trung bình)
**Cần train lại**: Có

**Lý do**:
- CD chỉ đo khoảng cách điểm -> không quan tâm surface smoothness
- Normal consistency loss bắt điểm gần nhau có normal giống nhau
- Kết quả: surface mượt mà hơn, ít gai/lỗi

**Cách thực hiện**:

```python
# Trong src/losses/losses.py:
def normal_consistency_loss(pred_xyz, k=10):
    """
    Bắt các điểm gần nhau có normal tương tự.
    Estimate normals từ k-nearest neighbors.
    """
    from scipy.spatial import cKDTree
    import numpy as np

    B, N, _ = pred_xyz.shape
    total_loss = 0.0

    for b in range(B):
        pts = pred_xyz[b].detach().cpu().numpy()
        tree = cKDTree(pts)
        _, idx = tree.query(pts, k=k+1)

        # Estimate normals bằng PCA trên neighbors
        normals = []
        for i in range(N):
            neighbors = pts[idx[i, 1:]]
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]  # Smallest eigenvalue
            normals.append(normal)
        normals = np.array(normals)

        # Consistency: neighbors nên có normal tương tự
        for i in range(N):
            nn_normals = normals[idx[i, 1:]]
            dot = np.abs((nn_normals * normals[i]).sum(axis=1))
            total_loss += (1 - dot.mean())

    return total_loss / (B * N)
```

---

### ======================================================
### GỢI Ý 7: Point Cloud format .pt (load nhanh hơn)
### ======================================================

**Độ khó**: ★☆☆☆☆ (Rất dễ)
**Tác động**: ★★☆☆☆ (Thấp-Trung bình, chỉ tăng tốc độ load)
**Cần train lại**: Không (chỉ preprocessing)

**Lý do**:
- Hiện tại đang đọc .ply bằng trimesh -> chậm
- Cap3D có sẵn format .pt (PyTorch tensor) -> load nhanh gấp 5-10 lần
- Format: `torch.Tensor [16384, 6]` với `[:,:3]` = XYZ, `[:,3:]` = RGB (normalized [-1,1])

**Cách thực hiện**:

**Bước 1**: Convert .ply -> .pt (chạy 1 lần)

```python
import torch, trimesh, os, glob

ply_dir = "Cap3D_data/pointclouds"
pt_dir = "Cap3D_data/pointclouds_pt"
os.makedirs(pt_dir, exist_ok=True)

for ply_file in glob.glob(f"{ply_dir}/*.ply"):
    mesh = trimesh.load(ply_file)
    xyz = torch.tensor(mesh.vertices, dtype=torch.float32)
    rgb = torch.tensor(mesh.visual.vertex_colors[:, :3], dtype=torch.float32) / 127.5 - 1.0
    pc = torch.cat([xyz, rgb], dim=1)  # [N, 6]
    obj_id = os.path.basename(ply_file).replace(".ply", "")
    torch.save(pc, os.path.join(pt_dir, f"{obj_id}.pt"))
```

**Bước 2**: Sửa dataset.py để đọc .pt

```python
def _load_point_cloud(self, obj_id):
    pt_path = os.path.join(self.pt_dir, f"{obj_id}.pt")
    if os.path.exists(pt_path):
        pc = torch.load(pt_path).numpy()
        # Convert RGB từ [-1,1] về [0,1]
        pc[:, 3:] = (pc[:, 3:] + 1.0) / 2.0
        return pc
    else:
        # Fallback: đọc .ply
        return self._load_point_cloud_ply(obj_id)
```

---

## 3. Bảng tổng hợp và thứ tự ưu tiên

| # | Gợi ý | Độ khó | Tác động | Thời gian | Ưu tiên |
|---|---|---|---|---|---|
| 5 | Mask background (MatAlpha) | ★☆☆☆☆ | ★★★☆☆ | 30 phút | **1** |
| 7 | Convert .pt format | ★☆☆☆☆ | ★★☆☆☆ | 1 giờ | **2** |
| 1 | Depth map input | ★★☆☆☆ | ★★★★☆ | 3-4 giờ | **3** |
| 4 | Tăng data ShapeNet | ★★☆☆☆ | ★★★★☆ | 2-3 giờ (code) + download | **4** |
| 2 | Multi-view input | ★★★☆☆ | ★★★★★ | 4-6 giờ | **5** |
| 6 | Normal consistency loss | ★★★☆☆ | ★★★☆☆ | 2-3 giờ | **6** |
| 3 | Camera parameters | ★★★★☆ | ★★★★☆ | 6-8 giờ | **7** |

## 4. Kế hoạch thực hiện khuyến nghị

### Phase 1 - Cải thiện nhanh (1-2 ngày, không cần train lại nhiều)
1. Áp dụng **Mask background** (Gợi ý 5) -> lọc ảnh sạch hơn
2. Convert **point cloud .pt** (Gợi ý 7) -> tăng tốc training

### Phase 2 - Cải thiện trung bình (3-5 ngày, cần train lại)
3. Thêm **Depth map input** (Gợi ý 1) -> model hiểu chiều sau
4. Tăng data **ShapeNet** (Gợi ý 4) -> đa dạng hơn, model generalise tốt hơn

### Phase 3 - Cải thiện sâu (1-2 tuần, cần train lại từ đầu)
5. **Multi-view input** (Gợi ý 2) -> cải thiện lớn nhất
6. Thêm **Normal consistency loss** (Gợi ý 6) -> surface mượt hơn
7. **Camera parameters** (Gợi ý 3) -> view-aware encoding

---

## 5. Lưu ý quan trọng

- **Mỗi thay đổi cần train lại** model từ đầu (checkpoint cũ không tương thích)
- **VRAM**: Multi-view và depth tăng VRAM -> giảm batch_size
- **Disk**: ShapeNet cần thêm ~45 GB disk trên Colab
- **Thời gian train**: Tăng data = tăng thời gian/epoch
- **Demo**: Các gợi ý trên chỉ ảnh hưởng training, demo UI giữ nguyên
