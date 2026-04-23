# Kế hoạch dự án: Xây dựng hệ thống tạo mô hình 3D từ ảnh và mô tả văn bản (AI đa phương thức)

## Kiến trúc tổng quan

```
Input: Ảnh 2D + Text mô tả
         │            │
    Image Encoder  Text Encoder    (CLIP pretrained)
         │            │
         └────┬───────┘
              │
        Multimodal Fusion          (Cross-attention)
              │
        3D Point Cloud Decoder     (Transformer decoder, 2-stage)
              │
Output: Point Cloud 3D (2,048 điểm có màu)
              │
        Mesh Reconstruction        (Poisson)
              │
Output: 3D Mesh (.ply / .obj)
```

## Dataset: Cap3D - ABO subset

- **7,930 objects** (sản phẩm thực tế từ Amazon)
- **14,330 captions** (2 loại: general + compositional)
- **Point clouds**: 16,384 điểm/object (XYZ + RGB)
- **Rendered images**: 20 góc nhìn/object + camera params

---

## Bước 1: Chuẩn bị dữ liệu ✅

- [x] Giải nén `PointCloud_zips_ABO/compressed_pcs_00.zip` → `pointclouds/` (7,953 files)
- [x] Giải nén `RenderedImage_perobj_zips_ABO/compressed_imgs_perobj_00.zip` → `rendered_images/` (7,953 folders)
- [x] Parse `Cap3D_automated_ABO.csv` → tạo mapping `{object_id → [general_caption, compositional_caption]}` (7,930 objects)
- [x] Kiểm tra object nào có đủ cả 3 modality (text + images + point cloud) → 7,930 valid
- [x] Loại bỏ objects thiếu dữ liệu (23 objects chỉ có PC+images, không có caption)
- [x] Chia tập dữ liệu: train (6,344) / val (793) / test (793)
- [x] Lưu splits vào `splits/train.txt`, `splits/val.txt`, `splits/test.txt` + `captions.json`
- [x] Viết script `src/data/prepare_data.py` thực hiện tất cả các bước trên

**Cấu trúc sau khi hoàn thành:**
```
Cap3D_data/
├── Cap3D_automated_ABO.csv
├── pointclouds/
│   ├── B07QFB4XCQ.ply
│   └── ...
├── rendered_images/
│   ├── B07QFB4XCQ/
│   │   ├── 00.png ... 19.png
│   │   └── camera.json
│   └── ...
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

---

## Bước 2: Data Pipeline ✅

- [x] Cài đặt dependencies: `torch`, `torchvision`, `open-clip-torch`, `open3d`, `pandas`
- [x] Viết `src/data/transforms.py` — augmentation cho image, point cloud, text
  - [x] Image: resize 224×224, normalize, random horizontal flip, color jitter
  - [x] Point cloud: random rotation (trục Y), random scale (0.9-1.1), random jitter
  - [x] Text: chọn ngẫu nhiên giữa general và compositional caption
- [x] Viết `src/data/dataset.py` — PyTorch Dataset class
  - [x] `__getitem__` trả về: `(image, text, point_cloud_gt)`
  - [x] Image: 1 ảnh ngẫu nhiên (trong 20 góc) → tensor [3, 224, 224]
  - [x] Text: caption string → CLIP tokenize
  - [x] Point cloud: subsample từ 16,384 → 2,048 điểm → tensor [2048, 6]
- [x] Viết DataLoader với batch_size=16, num_workers=4, shuffle=True (train)
- [ ] Test thử: load 1 batch, in shapes, visualize 1 sample

---

## Bước 3: Xây dựng Model ✅

### 3A. Encoders (`src/models/encoders.py`)
- [x] Load CLIP ViT-B/32 pretrained
- [x] Image encoder: ảnh → `image_features` [batch, 512]
- [x] Text encoder: text → `text_features` [batch, 512]
- [x] Chọn chiến lược: freeze CLIP hoặc fine-tune last 2 layers

### 3B. Multimodal Fusion (`src/models/fusion.py`)
- [x] Concatenate image_features + text_features → [batch, 1024]
- [x] MLP projection: 1024 → 512 (ReLU, LayerNorm)
- [x] (Nâng cao) Cross-attention giữa image tokens và text tokens

### 3C. Point Cloud Decoder (`src/models/decoder.py`)
- [x] **Coarse stage**: MLP (512 → 1024 → 256×3) → coarse points [batch, 256, 3]
- [x] **Fine stage**: FoldingNet / MLPs mỗi điểm thô → 8 điểm chi tiết
  - [x] Output: fine points [batch, 2048, 3] (XYZ)
  - [x] Color head: MLP → predicted colors [batch, 2048, 3] (RGB)

### 3D. Full Model (`src/models/multimodal3d.py`)
- [x] Kết nối: Encoders → Fusion → Decoder
- [x] Forward pass: `(image, text) → point_cloud [batch, 2048, 6]`
- [ ] Test: tạo dummy input, chạy forward, kiểm tra output shapes

---

## Bước 4: Training Pipeline ✅

### 4A. Loss Functions (`src/losses/losses.py`)
- [x] Chamfer Distance (CD) — tự implement (không dùng pytorch3d, khó cài Windows)
- [x] Earth Mover's Distance (EMD) — dùng scipy.optimize.linear_sum_assignment (approximation)
- [x] Color Loss (MSE) — sai số màu sắc
- [x] Total loss: `L = λ1·CD + λ2·EMD + λ3·Color`

### 4B. Training Script (`src/train.py`)
- [x] Optimizer: AdamW (lr=1e-4, weight_decay=0.05)
- [x] Scheduler: CosineAnnealingLR (warmup 5 epochs)
- [x] Training loop: forward → loss → backward → step
- [x] Validation mỗi epoch: tính CD trên val set
- [x] Logging: TensorBoard hoặc WandB
- [x] Save best model checkpoint theo val CD
- [x] Early stopping (patience=20)

### 4C. Config (`configs/default.yaml`)
- [x] Viết file config chứa tất cả hyperparameters
  ```yaml
  data:
    num_points: 2048
    image_size: 224
    batch_size: 16
    num_workers: 4
    train_split: 0.8
    val_split: 0.1
  model:
    clip_model: "ViT-B-32"
    freeze_clip: true
    fused_dim: 512
    coarse_points: 256
    fine_points: 2048
  training:
    epochs: 200
    lr: 1e-4
    weight_decay: 0.05
    warmup_epochs: 5
    lambda_cd: 1.0
    lambda_emd: 0.5
    lambda_color: 0.2
  ```

---

## Bước 5: Đánh giá ✅ (code sẵn, chờ training xong)

### 5A. Metrics (`src/evaluate.py`)
- [x] Chamfer Distance (CD) ↓ — càng nhỏ càng tốt
- [x] F-Score@0.01 ↑ — % điểm nằm trong bán kính τ=0.01
- [x] CLIP Similarity ↑ — render predicted → so sánh với text qua CLIP

### 5B. Evaluation
- [ ] Chạy model trên test set, tính mean CD, F-Score, CLIP-Sim
- [ ] So sánh visual: predicted vs ground truth (chọn 10-20 mẫu đẹp)
- [ ] Lưu kết quả vào bảng

### 5C. Ablation Studies
- [ ] Text-only input vs Image-only vs Text+Image → bảng so sánh CD
- [ ] Số điểm output: 1024 vs 2048 vs 4096
- [ ] Freeze CLIP vs Fine-tune last 2 layers
- [ ] Có/không color loss

---

## Bước 6: Demo & Visualization ✅ (code sẵn, chờ training xong)

### 6A. Visualization Utils
- [x] Viết hàm visualize point cloud 3D (plotly hoặc open3d)
- [x] Viết hàm render point cloud thành ảnh 2D từ nhiều góc
- [x] Point cloud → Mesh bằng Poisson reconstruction (open3d)
- [x] Export sang `.ply` và `.obj`

### 6B. Demo App (`src/demo.py`)
- [x] Cài Gradio: `pip install gradio`
- [x] Giao diện:
  - [x] Upload ảnh (hoặc chọn ảnh mẫu)
  - [x] Nhập mô tả text
  - [x] Nút "Generate 3D"
  - [x] Hiển thị 3D point cloud (interactive viewer)
  - [x] Nút download .ply / .obj
- [x] Load model checkpoint, chạy inference real-time
- [ ] Test demo end-to-end

---

## Bước 7: Báo cáo & Hoàn thiện

- [ ] Viết báo cáo bài tập lớn:
  - [ ] Giới thiệu đề tài, motivation
  - [ ] Tổng quan phương pháp (kiến trúc model + sơ đồ)
  - [ ] Dataset mô tả
  - [ ] Kết quả thí nghiệm (bảng metrics + hình ảnh so sánh)
  - [ ] Ablation studies
  - [ ] Demo screenshots
  - [ ] Kết luận & hướng phát triển
- [ ] Chuẩn bị slides thuyết trình
- [ ] Viết `README.md` cho project (hướng dẫn cài đặt, chạy)
- [ ] Clean code, thêm `requirements.txt`
- [ ] Test lại toàn bộ pipeline từ đầu đến cuối

---

## Cấu trúc thư mục dự án

```
ThiGiacMayTinh_BaiTapLon/
├── Cap3D_data/                    # Dữ liệu
│   ├── Cap3D_automated_ABO.csv
│   ├── pointclouds/
│   ├── rendered_images/
│   └── splits/
├── src/
│   ├── data/
│   │   ├── prepare_data.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── encoders.py
│   │   ├── fusion.py
│   │   ├── decoder.py
│   │   └── multimodal3d.py
│   ├── losses/
│   │   └── losses.py
│   ├── train.py
│   ├── evaluate.py
│   └── demo.py
├── configs/
│   └── default.yaml
├── checkpoints/
├── results/
├── requirements.txt
├── PLAN.md                        # ← File này
└── README.md
```

---

## Dependencies

```
torch>=2.0
torchvision
open-clip-torch
trimesh
gradio
pandas
numpy
Pillow
matplotlib
plotly
tensorboard
pyyaml
tqdm
scipy
```
