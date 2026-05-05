# Kế hoạch Cải thiện Dự án - Tạo Mô hình 3D từ Ảnh và Mô tả Văn bản

> **Dựa trên**: IMPROVEMENTS.md, note.txt, README_Cap3D.md, Cap3D (NeurIPS 2023), DiffuRank (ECCV 2024)
> **Mục tiêu**: Cải thiện chất lượng tạo mô hình 3D, tận dụng tối đa dataset có sẵn
> **⏰ Deadline**: 1 tuần (7 ngày)
> **💻 Môi trường**: Google Colab CPU — RAM 12.7 GB, Disk 225.8 GB

---

## Tổng quan hiện trạng

### Kiến trúc hiện tại
- **Encoder**: OpenCLIP ViT-B-32 encode ảnh RGB + text caption
- **Decoder**: Transformer decoder với learnable queries (coarse → fine)
- **Loss**: Chamfer Distance + EMD + Color + Repulsion + Uniform
- **Dữ liệu**: ABO subset (~7,953 objects), 1 view RGB mỗi object

### Tài nguyên chưa khai thác
- Depth map (20 views/object) → thông tin chiều sâu 3D
- MatAlpha mask → segmentation object/background
- Camera parameters (intrinsic + extrinsic) → geometry-aware
- 20 multi-view images/object → chỉ đang dùng 1
- ShapeNet subset (52,472 objects) → gấp 7x data hiện tại

### Bài báo liên quan cần tham khảo
| Bài báo | Venue | Đóng góp chính |
|---|---|---|
| Cap3D | NeurIPS 2023 | Multi-view captioning bằng CLIP/BLIP2 + LLM |
| DiffuRank | ECCV 2024 | Chọn view tốt nhất bằng diffusion model alignment |
| PointNet++ | NeurIPS 2017 | Hierarchical point cloud learning |
| Point-E | - | CLIP-conditioned 3D generation |
| ShapE | - | ShapE latent codes cho 3D descriptors |

---

## Phase 1 - Hoàn thiện nền tảng (Ngày 1) ✅ Ưu tiên cao nhất

### Nhiệm vụ 1.1: Thống kê và phân tích dataset (từ note.txt)
**Mục tiêu**: Hiểu rõ phân bố dữ liệu trước khi cải thiện

**Bước thực hiện**:
1. Đọc `Cap3D_automated_ABO.csv` → phân tích phân bố categories
2. Chuẩn hóa caption: translate sang tiếng Việt bằng API (Google Translate hoặc deepl)
3. Thống kê số nhãn: đếm từ khóa phổ biến, nhóm category

```python
# Phân tích caption distribution
import pandas as pd
from collections import Counter

df = pd.read_csv("Cap3D_data/Cap3D_automated_ABO.csv", header=None)
print(f"Tổng số objects: {len(df)}")

# Trích xuất từ khóa phổ biến
words = []
for caption in df[1]:
    words.extend(caption.lower().split())
counter = Counter(words)
print("Top 30 từ phổ biến:", counter.most_common(30))
```

### Nhiệm vụ 1.2: Hiển thị độ đo đánh giá (từ note.txt)
**Mục tiêu**: Trực quan hóa metrics training trong demo

**Bước thực hiện**:
1. Đọc checkpoint và in Chamfer Distance (CD) theo epoch từ log
2. Thêm panel hiển thị metrics trong Gradio demo
3. So sánh CD của các checkpoint khác nhau

```python
# Trong src/demo.py - thêm tab metrics
def show_metrics():
    """Đọc training log và hiển thị metrics."""
    log_path = os.path.join("runs", "train_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            logs = json.load(f)
        # Plot CD theo epoch
        epochs = [l["epoch"] for l in logs]
        train_cd = [l.get("train_cd", None) for l in logs]
        val_cd = [l.get("val_cd", None) for l in logs]
        return epochs, train_cd, val_cd
```

### Nhiệm vụ 1.3: Áp dụng MatAlpha Mask lọc background
**Độ khó**: ★☆☆☆☆ | **Tác động**: ★★★☆☆

**Bước thực hiện**:
1. Sửa `src/data/dataset.py` → load `_MatAlpha.png` cùng RGB
2. Set background = white (255,255,255) trước khi encode CLIP
3. Retrain hoặc fine-tune model

**Lý do quan trọng**: DiffuRank paper cho thấy background gây noise cho CLIP features, MatAlpha giúp isolate object

---

## Phase 2 - Cải thiện Chất lượng Model (Ngày 2-4)

### Nhiệm vụ 2.1: Tìm kiếm bài báo cụ thể (từ note.txt)
**Mục tiêu**: Xác định các phương pháp SOTA để áp dụng

**Danh sách bài báo cần đọc**:
1. **Cap3D** (NeurIPS 2023) - `arxiv.org/abs/2306.07279`
   - Phương pháp captioning đa view → học cách tổng hợp multi-view info
2. **DiffuRank** (ECCV 2024) - `arxiv.org/abs/2404.07984`
   - View selection bằng diffusion alignment → chọn view tốt nhất cho training
3. **Point-E** (OpenAI) - CLIP → 3D point cloud pipeline
4. **ShapE** (OpenAI) - Latent 3D generation
5. **PointNet++** - Hierarchical 3D feature extraction

**Ứng dụng từ DiffuRank**:
- DiffuRank chọn "key views" có alignment cao nhất với 3D descriptor
- Có thể áp dụng: thay vì random chọn 1 view, **chọn view có CLIP score cao nhất với caption**

```python
def select_best_view(obj_id, caption, clip_model):
    """Chọn view có CLIP similarity cao nhất với caption."""
    views = load_all_views(obj_id)  # 20 views
    text_feat = clip_model.encode_text(caption)
    
    best_score, best_view = -1, views[0]
    for view in views:
        img_feat = clip_model.encode_image(view)
        score = (img_feat @ text_feat.T).item()
        if score > best_score:
            best_score, best_view = score, view
    return best_view
```

### Nhiệm vụ 2.2: Thêm Depth Map Input
**Độ khó**: ★★☆☆☆ | **Tác động**: ★★★★☆

- Load `_depth.png` tương ứng với mỗi RGB view
- Thêm `DepthEncoder` (CNN nhỏ) song song với CLIP encoder
- Concat depth features với image+text features trước decoder

### Nhiệm vụ 2.3: Tăng dữ liệu với ShapeNet
**Độ khó**: ★★☆☆☆ | **Tác động**: ★★★★☆

> **Disk 225.8 GB cho phép tải ShapeNet** — Point clouds ~15 GB hoàn toàn khả thi

- Tải `PointCloud_zips_ShapeNet/` (~15 GB, 52,472 objects)
- Tải `Cap3D_automated_ShapeNet.csv` (captions, chỉ ~19.5 MB)
- Gộp ABO + ShapeNet → ~60,000 objects
- Tạo lại splits: 80% train / 10% val / 10% test
- ⚠️ **Không tải Rendered Images ShapeNet (~30 GB)** — tốn thời gian, chưa cần thiết

---

## Phase 3 - Cải thiện Nâng cao (Ngày 5-7) — Chọn 1 trong 2

### Nhiệm vụ 3.1: Multi-View Input với DiffuRank-inspired View Selection
**Độ khó**: ★★★☆☆ | **Tác động**: ★★★★★

**Kết hợp hai ý tưởng**:
1. Load K=4 views thay vì 1 view
2. Áp dụng **DiffuRank approach**: chọn K views có CLIP-caption alignment cao nhất

```python
def load_diffurank_views(obj_id, caption, clip_model, k=4):
    """
    Chọn K views tốt nhất theo DiffuRank alignment.
    Thay vì random → chọn views align nhất với caption.
    """
    all_views = load_all_views(obj_id)
    text_feat = clip_model.encode_text(caption).float()
    
    scores = []
    for view in all_views:
        img_feat = clip_model.encode_image(view).float()
        score = F.cosine_similarity(img_feat, text_feat).item()
        scores.append(score)
    
    # Top-K views
    top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [all_views[i] for i in top_k_idx]
```

### Nhiệm vụ 3.2 (Phương án A): Normal Consistency Loss
**Độ khó**: ★★★☆☆ | **Tác động**: ★★★☆☆ | **Thời gian**: ~3 giờ code

- Bổ sung loss function để điểm gần nhau có normal vector tương tự
- Không cần thay đổi architecture → dễ implement trong 1 ngày
- Kết quả: bề mặt mượt mà hơn, ít lỗi hơn

### Nhiệm vụ 3.3 (Phương án B): Depth Map Input
**Độ khó**: ★★☆☆☆ | **Tác động**: ★★★★☆ | **Thời gian**: ~4 giờ code

- Load `_depth.png` song song với RGB (data đã có sẵn, không cần download thêm)
- Thêm `DepthEncoder` CNN nhỏ → concat với CLIP features
- Train lại 20-30 epoch để thấy cải thiện

> ❌ **Loại bỏ khỏi deadline 1 tuần**:
> - ShapE latent codes (tải ~20 GB + complex integration)
> - Camera-aware encoding (phức tạp, cần nhiều thời gian debug)

---

## Bảng tổng hợp kế hoạch (1 tuần)

| Ngày | Nhiệm vụ | Độ khó | Tác động | Thời gian |
|---|---|---|---|---|
| **1** | Thống kê & phân tích dataset | ★☆☆☆☆ | ★★☆☆☆ | 2 giờ |
| **1** | Hiển thị độ đo đánh giá trong demo | ★★☆☆☆ | ★★★☆☆ | 3 giờ |
| **1** | MatAlpha mask lọc background | ★☆☆☆☆ | ★★★☆☆ | 2 giờ |
| **2** | Đọc & tóm tắt bài báo | ★★☆☆☆ | ★★★★☆ | 4 giờ |
| **2** | DiffuRank-inspired best view selection | ★★☆☆☆ | ★★★★☆ | 3 giờ |
| **3-4** | Tải ShapeNet + gộp data (~15 GB) | ★★☆☆☆ | ★★★★☆ | 2 giờ code + download |
| **4-5** | Train lại với data mới + đánh giá | ★★☆☆☆ | ★★★★☆ | training time |
| **5** | Multi-view input (K=4) | ★★★☆☆ | ★★★★★ | 4 giờ |
| **6** | Normal consistency loss (3A) **hoặc** Depth map (3B) | ★★★☆☆ | ★★★☆☆ | 3-4 giờ |
| **7** | Train cuối + đánh giá tổng hợp + báo cáo | - | - | cả ngày |

---

## Lịch thực hiện 7 ngày

```
📅 Ngày 1 — Hoàn thiện nền tảng
  ├── Thống kê dataset (phân bố category, từ khóa)
  ├── Thêm panel metrics vào Gradio demo
  └── Implement MatAlpha mask → lọc background

📅 Ngày 2 — Nghiên cứu & cải thiện view selection
  ├── Đọc Cap3D + DiffuRank paper (tóm tắt ý chính)
  └── Implement chọn view tốt nhất theo CLIP-caption alignment

📅 Ngày 3 — Tải & tích hợp ShapeNet
  ├── Download PointCloud_zips_ShapeNet (~15 GB) — Disk còn 200+ GB
  ├── Download Cap3D_automated_ShapeNet.csv
  └── Sửa dataset.py + tạo lại splits

📅 Ngày 4 — Train với data mới
  ├── Chạy training với ABO + ShapeNet (~60k objects)
  └── Theo dõi CD loss, điều chỉnh nếu cần

📅 Ngày 5 — Multi-view input
  ├── Sửa dataset.py → load K=4 views
  ├── Sửa encoders.py → encode_multi_view()
  └── Bắt đầu train với multi-view

📅 Ngày 6 — Cải thiện loss / depth
  ├── Chọn: Normal consistency loss (nhanh hơn)
  │         hoặc Depth map input (tác động cao hơn)
  └── Implement + train thêm

📅 Ngày 7 — Đánh giá & hoàn thiện
  ├── Đánh giá CD, F-Score trên test set
  ├── So sánh baseline vs cải thiện
  ├── Chạy demo với ảnh đa dạng
  └── Viết báo cáo tổng hợp
```

## Lưu ý cấu hình Colab CPU (RAM 12.7 GB, Disk 225.8 GB)

| Tài nguyên | Giá trị | Tác động đến kế hoạch |
|---|---|---|
| **RAM 12.7 GB** | Đủ dùng | batch_size=8, max_objects=5000, num_workers=4 |
| **Disk 225.8 GB** | Rộng rãi | Tải được ShapeNet (~15 GB) + rendered images nếu cần |
| **CPU only** | Chậm | Dùng `mlp` decoder, tắt EMD, giới hạn max_objects |

> ⚡ Dùng `configs/colab_cpu.yaml` đã được tối ưu cho cấu hình này

---

## Tài liệu tham khảo

| Tên | Link |
|---|---|
| Cap3D Paper | https://arxiv.org/abs/2306.07279 |
| DiffuRank Paper | https://arxiv.org/abs/2404.07984 |
| Cap3D GitHub | https://github.com/crockwell/Cap3D |
| DiffuRank GitHub | https://github.com/tiangeluo/DiffuRank |
| Cap3D Dataset | https://huggingface.co/datasets/tiange/Cap3D |
| PointNet++ | https://arxiv.org/abs/1706.02413 |
| ShapE | https://arxiv.org/abs/2305.02463 |
| Point-E | https://arxiv.org/abs/2212.08751 |
