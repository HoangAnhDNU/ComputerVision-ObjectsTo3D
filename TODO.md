# TODO — Hoàn thành dự án

## Trạng thái hiện tại

| Bước | Trạng thái | Ghi chú |
|------|-----------|---------|
| Code pipeline | ✅ Xong | Encoder → Fusion → Decoder → Demo |
| Data preparation | ✅ Xong | Cap3D ABO, splits, captions |
| Bug fixes | ✅ Xong | L2 normalize, tắt rotation, tuning LR |
| Training | ⏳ Cần train lại | Checkpoint cũ không dùng được |
| Evaluation | ❌ Chưa chạy | Cần checkpoint tốt |
| Demo | ❌ Chưa có kết quả tốt | Cần checkpoint tốt |

---

## Việc cần làm (theo thứ tự)

### 1. Train lại model (BẮT BUỘC)

Checkpoint cũ train với bug (random rotation + warmup sai) → phải train lại từ đầu.

```bash
python src/train.py --epochs 15
```

- Thời gian ước tính: **6-10 giờ** trên CPU, **1-2 giờ** trên GPU
- Sau khi xong sẽ có `checkpoints/best_model.pth` và `checkpoints/last_model.pth`
- Có thể tắt máy giữa chừng rồi resume:
  ```bash
  python src/train.py --resume checkpoints/last_model.pth --epochs 15
  ```

### 2. Kiểm tra kết quả training

Sau khi train xong, xem TensorBoard để xác nhận loss giảm:

```bash
tensorboard --logdir runs
```

Kiểm tra:
- [ ] `train/chamfer_distance` giảm dần
- [ ] `val/chamfer_distance` giảm dần
- [ ] `train/color_loss` giảm dần

### 3. Chạy evaluation

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

Ghi lại kết quả:
- [ ] Chamfer Distance (CD) — mong đợi < 0.05
- [ ] F-Score@0.01 — mong đợi > 0.3

### 4. Chạy demo và chụp screenshot

```bash
python src/demo.py --checkpoint checkpoints/best_model.pth
```

Mở `http://localhost:7860`, test với 2-3 ảnh từ dataset:

- [ ] **Demo 1**: `Cap3D_data/rendered_images/B002CM3J2A/00000.png` + text `"a white wooden table with two matching chairs"`
- [ ] **Demo 2**: `Cap3D_data/rendered_images/B003DQ5B2U/00000.png` + text `"a set of three nesting tables"`
- [ ] **Demo 3**: `Cap3D_data/rendered_images/B003LSTD38/00000.png` + text `"a sleek rectangular wallet"`

Chụp screenshot kết quả để đưa vào báo cáo.

### 5. Cập nhật README.md

- [ ] Cập nhật lại hyperparameters trong README (lr=0.0005, warmup=2, weight_decay=0.01)
- [ ] Thêm screenshot demo vào README

### 6. (Tùy chọn) Viết báo cáo đồ án

Nội dung cần có:
- [ ] Giới thiệu đề tài
- [ ] Kiến trúc hệ thống (sơ đồ trong README)
- [ ] Dataset (Cap3D ABO, 7930 objects)
- [ ] Phương pháp (CLIP encoder, Fusion, 2-stage decoder)
- [ ] Loss functions (CD, EMD, Color)
- [ ] Kết quả training (biểu đồ loss từ TensorBoard)
- [ ] Kết quả evaluation (bảng metrics)
- [ ] Demo screenshots
- [ ] Kết luận

### 7. Push lên GitHub

```bash
git add -A
git commit -m "Final: complete training and demo"
git push origin main
```

Lưu ý: `.gitignore` đã loại trừ `Cap3D_data/`, `checkpoints/`, `runs/` — chỉ push code.

---

## Lưu ý quan trọng

- **Ảnh demo**: Dùng ảnh từ `Cap3D_data/rendered_images/` (ảnh 2D render từ object 3D), KHÔNG dùng ảnh phong cảnh
- **Text demo**: Lấy caption từ `Cap3D_data/captions.json` cho object tương ứng
- **Nếu máy nóng**: Config đã có `cooldown_seconds: 30` giữa các epoch, có thể tăng lên 60 trong `configs/default.yaml`
