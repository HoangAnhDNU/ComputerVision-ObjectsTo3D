# Xây dựng hệ thống tạo mô hình 3D từ ảnh và mô tả văn bản (AI đa phương thức)

Đồ án môn **Thị giác máy tính** — Xây dựng hệ thống sử dụng AI đa phương thức để tạo mô hình 3D (point cloud + mesh) từ ảnh 2D và mô tả văn bản.

## Kiến trúc

```
Input: Ảnh 2D + Text mô tả
        │            │
   Image Encoder  Text Encoder    (CLIP ViT-B/32 pretrained)
        │            │
        └────┬───────┘
             │
       Multimodal Fusion          (Concat MLP / Cross-Attention)
             │
       3D Point Cloud Decoder     (2-stage: Coarse → Fine + Color)
             │
Output: Point Cloud 3D (2,048 điểm có màu)
             │
       Mesh Reconstruction        (Poisson)
             │
Output: 3D Mesh (.ply / .obj)
```

## Hướng dẫn chạy dự án (từ A → Z)

> **Yêu cầu**: Python 3.10+ (khuyến nghị 3.10 hoặc 3.11; Python 3.13 có thể gặp vấn đề với một số package)

### Bước 1: Tạo virtual environment

```bash
python -m venv venv

# Kích hoạt:
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac
```

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 3: Tải dữ liệu từ HuggingFace

Dataset: **Cap3D - ABO + ShapeNet** (~7,930 ABO + 52,472 ShapeNet objects, tổng ~55GB)

> **Lưu ý**: Dùng lệnh `hf` (không phải `huggingface-cli`, đã deprecated)

#### 3.1 ABO subset (bắt buộc, ~40GB)

```bash
pip install huggingface_hub

# Tải captions ABO
hf download tiange/Cap3D Cap3D_automated_ABO.csv --repo-type dataset --local-dir ./Cap3D_data

# Tải point clouds ABO (~1.5 GB)
hf download tiange/Cap3D PointCloud_zips_ABO/compressed_pcs_00.zip --repo-type dataset --local-dir ./Cap3D_data

# Tải rendered images ABO (~37 GB, gồm RGB, depth, MatAlpha mask, camera params)
hf download tiange/Cap3D RenderedImage_perobj_zips_ABO/compressed_imgs_perobj_00.zip --repo-type dataset --local-dir ./Cap3D_data
```

#### 3.2 ShapeNet subset (khuyến nghị, ~15 GB thêm)

Thêm 52,472 objects đa dạng (chair, car, plane, lamp, ...) → model generalise tốt hơn.

```bash
# Tải captions ShapeNet (~19.5 MB)
hf download tiange/Cap3D Cap3D_automated_ShapeNet.csv --repo-type dataset --local-dir ./Cap3D_data

# Tải point clouds ShapeNet (~15 GB, 1 file)
hf download tiange/Cap3D PointCloud_zips_ShapeNet/compressed_pcs_00.zip --repo-type dataset --local-dir ./Cap3D_data
```

> **Disk cần thiết**: ABO ~40 GB + ShapeNet point clouds ~15 GB = **~55 GB** tổng

### Bước 4: Chuẩn bị dữ liệu (giải nén + tạo splits)

```bash
# Chỉ ABO
python src/data/prepare_data.py --data-dir Cap3D_data

# ABO + ShapeNet (nếu đã tải shapenet/pointclouds/)
python src/data/prepare_data.py --data-dir Cap3D_data --include-shapenet
```

Script này sẽ:
- Giải nén file zip point clouds và rendered images
- Parse captions từ CSV → `captions.json` (gộp ABO + ShapeNet nếu có)
- Chia train/val/test splits → `splits/train.txt`, `splits/val.txt`, `splits/test.txt`

### Bước 5: Training

```bash
# Train với config (CPU-only, tối ưu cho Colab)
python src/train.py --config configs/colab.yaml

# Override tham số:
python src/train.py --config configs/colab.yaml --epochs 100 --batch-size 4

# Resume từ checkpoint:
python src/train.py --config configs/colab.yaml --resume checkpoints/last_model.pth
```

**Lưu ý cho Windows**: Config đã đặt `num_workers: 0` để tránh lỗi multiprocessing trên Windows.

**Lưu ý cho CPU**: EMD loss đã tắt (`use_emd: false`) vì thuật toán Hungarian rất chậm trên CPU. Cấu hình đầy đủ trong `configs/colab.yaml`.

### Bước 6: Theo dõi training bằng TensorBoard

```bash
tensorboard --logdir runs
# Mở http://localhost:6006 trên trình duyệt
```

### Bước 7: Đánh giá

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

### Bước 8: Demo (Gradio)

```bash
python src/demo.py --checkpoint checkpoints/best_model.pth
# Mở http://localhost:7860 trên trình duyệt
```

Demo hỗ trợ **tiếng Việt**: nhập mô tả tiếng Việt trực tiếp, ví dụ:
`một chiếc ghế gỗ có bốn chân`, `một chiếc đèn bàn màu trắng`

## Cấu trúc thư mục

```
ThiGiacMayTinh_BaiTapLon/
├── Cap3D_data/                       # Dữ liệu
│   ├── Cap3D_automated_ABO.csv       # Captions ABO (~7,930 objects)
│   ├── Cap3D_automated_ShapeNet.csv  # Captions ShapeNet (~52,472 objects)
│   ├── captions.json                 # Captions đã merge (ABO + ShapeNet)
│   ├── captions_vi.json              # Captions dịch tiếng Việt
│   ├── pointclouds/                  # ABO point clouds (.ply)
│   ├── pointclouds_pt/               # ABO point clouds (.pt, load nhanh hơn)
│   ├── rendered_images/              # ABO rendered images (RGB + depth + mask)
│   ├── rendered_images_masked/       # ABO images sau MatAlpha mask
│   ├── shapenet/
│   │   ├── pointclouds/              # ShapeNet point clouds (.ply)
│   │   └── pointclouds_pt/           # ShapeNet point clouds (.pt)
│   └── splits/                       # train.txt, val.txt, test.txt
├── src/
│   ├── data/
│   │   ├── prepare_data.py            # Chuẩn bị dữ liệu (ABO + ShapeNet)
│   │   ├── dataset.py                 # PyTorch Dataset + DataLoader
│   │   └── transforms.py             # Augmentation
│   ├── models/
│   │   ├── encoders.py                # CLIP Image/Text Encoders
│   │   ├── fusion.py                  # Multimodal Fusion
│   │   ├── decoder.py                 # 2-stage Point Cloud Decoder (MLP)
│   │   ├── transformer_decoder.py     # Transformer Decoder (alternative)
│   │   └── multimodal3d.py            # Full Model
│   ├── losses/
│   │   └── losses.py                  # CD, EMD, Color, Repulsion Loss
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation script
│   └── demo.py                        # Gradio Demo (tiếng Việt)
├── configs/
│   └── colab.yaml                     # Config CPU-only (Colab)
├── PLAN/
│   ├── Plan.md                        # Kế hoạch cải thiện 7 ngày
│   ├── IMPROVEMENTS.md               # Phân tích và gợi ý cải thiện
│   ├── README_Cap3D.md               # Tài liệu Cap3D dataset
│   └── note.txt                       # Yêu cầu bổ sung
├── images/                            # Ảnh mẫu cho demo
├── checkpoints/                       # Model checkpoints
├── results/                           # Evaluation results
├── colab_training.ipynb               # Notebook Colab (CPU, auto-resume)
├── requirements.txt
└── README.md
```

## Loss Functions

- **Chamfer Distance (CD)**: Khoảng cách hình dạng giữa 2 point clouds
- **Earth Mover's Distance (EMD)**: Phân bố đều điểm (approximation via Hungarian)
- **Color Loss (MSE)**: Sai số màu sắc
- **Total**: `L = λ₁·CD + λ₂·EMD + λ₃·Color`

## Evaluation Metrics

- **Chamfer Distance (CD)** ↓ — càng nhỏ càng tốt
- **F-Score@0.01** ↑ — % điểm nằm trong bán kính τ=0.01
- **CLIP Similarity** ↑ — tương quan semantic giữa render và text
