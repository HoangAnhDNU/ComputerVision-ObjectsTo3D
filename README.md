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

Dataset: **Cap3D - ABO subset** (~7,930 objects từ Amazon, tổng ~40GB)

```bash
pip install huggingface_hub

# Tải captions
huggingface-cli download tiange/Cap3D Cap3D_automated_ABO.csv --repo-type dataset --local-dir ./Cap3D_data

# Tải point clouds ABO
huggingface-cli download tiange/Cap3D PointCloud_zips_ABO/compressed_pcs_00.zip --repo-type dataset --local-dir ./Cap3D_data

# Tải rendered images ABO
huggingface-cli download tiange/Cap3D RenderedImage_perobj_zips_ABO/compressed_imgs_perobj_00.zip --repo-type dataset --local-dir ./Cap3D_data
```

### Bước 4: Chuẩn bị dữ liệu (giải nén + tạo splits)

```bash
python src/data/prepare_data.py --data-dir Cap3D_data
```

Script này sẽ:
- Giải nén file zip point clouds và rendered images
- Parse captions từ CSV → `captions.json`
- Chia train/val/test splits → `splits/train.txt`, `splits/val.txt`, `splits/test.txt`

### Bước 5: Training

```bash
# Train với config mặc định (50 epochs, batch_size=8, lr=0.0001)
python src/train.py

# Hoặc chỉ định config:
python src/train.py --config configs/default.yaml

# Override tham số:
python src/train.py --epochs 100 --batch-size 4 --lr 0.00005

# Resume từ checkpoint:
python src/train.py --resume checkpoints/last_model.pth
```

**Lưu ý cho Windows**: Config đã đặt `num_workers: 0` để tránh lỗi multiprocessing trên Windows.

**Lưu ý cho CPU**: EMD loss đã tắt (`use_emd: false`) vì thuật toán Hungarian rất chậm trên CPU. Nếu có GPU, có thể bật lại trong `configs/default.yaml`.

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

## Cấu trúc thư mục

```
ThiGiacMayTinh_BaiTapLon/
├── Cap3D_data/                    # Dữ liệu
│   ├── Cap3D_automated_ABO.csv
│   ├── captions.json
│   ├── pointclouds/
│   ├── rendered_images/
│   └── splits/
├── src/
│   ├── data/
│   │   ├── prepare_data.py        # Chuẩn bị dữ liệu
│   │   ├── dataset.py             # PyTorch Dataset + DataLoader
│   │   └── transforms.py         # Augmentation
│   ├── models/
│   │   ├── encoders.py            # CLIP Image/Text Encoders
│   │   ├── fusion.py              # Multimodal Fusion
│   │   ├── decoder.py             # 2-stage Point Cloud Decoder
│   │   └── multimodal3d.py        # Full Model
│   ├── losses/
│   │   └── losses.py              # CD, EMD, Color Loss
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── demo.py                    # Gradio Demo
├── configs/
│   └── default.yaml               # Hyperparameters
├── checkpoints/                   # Model checkpoints
├── results/                       # Evaluation results
├── requirements.txt
├── PLAN.md
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
