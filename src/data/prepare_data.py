"""
Bước 1: Chuẩn bị dữ liệu
- Giải nén point clouds và rendered images
- Parse captions CSV
- Kiểm tra tính toàn vẹn dữ liệu
- Chia train/val/test splits
"""

import argparse
import csv
import json
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path


def extract_pointclouds(data_dir: str) -> set:
    """Giải nén point clouds từ zip và trả về set các object IDs."""
    zip_path = os.path.join(data_dir, "PointCloud_zips_ABO", "compressed_pcs_00.zip")
    out_dir = os.path.join(data_dir, "pointclouds")
    os.makedirs(out_dir, exist_ok=True)

    # Kiểm tra đã giải nén chưa (kiểm tra TRƯỚC khi tìm zip)
    existing = set()
    for f in os.listdir(out_dir):
        if f.endswith(".ply"):
            existing.add(f.replace(".ply", ""))

    if len(existing) > 0:
        print(f"[PointClouds] Đã có {len(existing)} files, bỏ qua giải nén.")
        return existing

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Không tìm thấy: {zip_path}")

    print(f"[PointClouds] Đang giải nén {zip_path}...")
    extracted_ids = set()
    with zipfile.ZipFile(zip_path, "r") as z:
        members = [m for m in z.namelist() if m.endswith(".ply")]
        total = len(members)
        for i, member in enumerate(members, 1):
            # Tên file: ABO_pcs/B0825D873D.ply → B0825D873D.ply
            filename = os.path.basename(member)
            obj_id = filename.replace(".ply", "")
            target = os.path.join(out_dir, filename)

            with z.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())

            extracted_ids.add(obj_id)
            if i % 1000 == 0 or i == total:
                print(f"  [{i}/{total}] extracted")

    print(f"[PointClouds] Hoàn thành: {len(extracted_ids)} files")
    return extracted_ids


def extract_rendered_images(data_dir: str) -> set:
    """Giải nén rendered images (zip lồng zip) và trả về set các object IDs."""
    zip_path = os.path.join(
        data_dir, "RenderedImage_perobj_zips_ABO", "compressed_imgs_perobj_00.zip"
    )
    out_dir = os.path.join(data_dir, "rendered_images")
    os.makedirs(out_dir, exist_ok=True)

    # Kiểm tra đã giải nén chưa (kiểm tra TRƯỚC khi tìm zip)
    existing = set()
    for d in os.listdir(out_dir):
        if os.path.isdir(os.path.join(out_dir, d)):
            existing.add(d)

    if len(existing) > 0:
        print(f"[RenderedImages] Đã có {len(existing)} folders, bỏ qua giải nén.")
        return existing

    if not os.path.exists(zip_path):
        print(f"[RenderedImages] Không tìm thấy zip và chưa có dữ liệu giải nén.")
        return existing

    print(f"[RenderedImages] Đang giải nén {zip_path}...")
    print("  (File ~37GB, có thể mất 10-30 phút...)")
    extracted_ids = set()

    with zipfile.ZipFile(zip_path, "r") as outer_z:
        inner_zips = [m for m in outer_z.namelist() if m.endswith(".zip")]
        total = len(inner_zips)

        for i, inner_name in enumerate(inner_zips, 1):
            # Tên: Cap3D_ABO_renderimgs/B0825D873D.zip → B0825D873D
            obj_id = os.path.basename(inner_name).replace(".zip", "")
            obj_dir = os.path.join(out_dir, obj_id)
            os.makedirs(obj_dir, exist_ok=True)

            # Đọc inner zip từ memory và giải nén
            inner_bytes = outer_z.read(inner_name)
            import io

            with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as inner_z:
                for member in inner_z.namelist():
                    if member.endswith("/"):
                        continue
                    filename = os.path.basename(member)
                    if not filename:
                        continue
                    target = os.path.join(obj_dir, filename)
                    with inner_z.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())

            extracted_ids.add(obj_id)
            if i % 500 == 0 or i == total:
                print(f"  [{i}/{total}] extracted")

    print(f"[RenderedImages] Hoàn thành: {len(extracted_ids)} folders")
    return extracted_ids


def parse_captions(data_dir: str) -> dict:
    """Parse ABO CSV thành mapping {object_id → {general: str}}.
    
    ABO CSV có 2 cột: obj_id, caption_text (không có header).
    """
    csv_path = os.path.join(data_dir, "Cap3D_automated_ABO.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy: {csv_path}")

    captions = defaultdict(dict)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                obj_id = row[0].strip()
                # ABO có thể có 2 cột (obj_id, caption) hoặc 3 cột (obj_id, type, caption)
                if len(row) >= 3:
                    caption_type, text = row[1].strip(), row[2].strip()
                    captions[obj_id][caption_type] = text
                else:
                    captions[obj_id]["general"] = row[1].strip()

    print(f"[ABO Captions] Đọc {len(captions)} objects từ CSV")
    return dict(captions)


def parse_shapenet_captions(data_dir: str) -> dict:
    """Parse ShapeNet CSV (2 cột: obj_id, caption) thành mapping {obj_id → {general: str}}."""
    csv_path = os.path.join(data_dir, "Cap3D_automated_ShapeNet.csv")
    if not os.path.exists(csv_path):
        print(f"[ShapeNet] Không tìm thấy: {csv_path} → bỏ qua")
        return {}

    captions = defaultdict(dict)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                obj_id = row[0].strip()
                text = row[1].strip()
                captions[obj_id]["general"] = text

    print(f"[ShapeNet Captions] Đọc {len(captions)} objects từ CSV")
    return dict(captions)


def verify_data(captions: dict, pc_ids: set, img_ids: set, require_images: bool = True) -> list:
    """Kiểm tra objects có đủ modality và trả về danh sách hợp lệ.
    
    Args:
        require_images: Nếu True (ABO), yêu cầu có rendered images.
                        Nếu False (ShapeNet), chỉ cần caption + point cloud.
    """
    caption_ids = set(captions.keys())

    if require_images:
        valid_ids = caption_ids & pc_ids & img_ids
    else:
        valid_ids = caption_ids & pc_ids  # ShapeNet: không có rendered images

    # Chỉ giữ objects có ít nhất 1 caption
    valid_ids = {oid for oid in valid_ids if len(captions.get(oid, {})) > 0}

    print(f"[Verify] Objects có caption: {len(caption_ids)}")
    print(f"[Verify] Objects có point cloud: {len(pc_ids)}")
    if require_images:
        print(f"[Verify] Objects có rendered images: {len(img_ids)}")
        print(f"[Verify] Objects hợp lệ (đủ 3 modality): {len(valid_ids)}")
    else:
        print(f"[Verify] Objects hợp lệ (caption + PC, không cần images): {len(valid_ids)}")

    missing_pc = caption_ids - pc_ids
    if missing_pc:
        print(f"  [!] {len(missing_pc)} objects thiếu point cloud")
    if require_images:
        missing_img = caption_ids - img_ids
        if missing_img:
            print(f"  [!] {len(missing_img)} objects thiếu rendered images")

    return sorted(valid_ids)


def split_data(valid_ids: list, train_ratio=0.8, val_ratio=0.1, seed=42) -> dict:
    """Chia train/val/test theo tỉ lệ."""
    random.seed(seed)
    ids = valid_ids.copy()
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }

    print(f"[Split] Train: {len(splits['train'])}")
    print(f"[Split] Val:   {len(splits['val'])}")
    print(f"[Split] Test:  {len(splits['test'])}")
    print(f"[Split] Total: {sum(len(v) for v in splits.values())}")

    return splits


def save_splits(data_dir: str, splits: dict):
    """Lưu splits vào text files."""
    split_dir = os.path.join(data_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    for split_name, ids in splits.items():
        path = os.path.join(split_dir, f"{split_name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(ids))
        print(f"[Save] {path} ({len(ids)} objects)")


def save_captions_json(data_dir: str, captions: dict, valid_ids: list):
    """Lưu captions đã lọc thành JSON cho dễ load sau này."""
    filtered = {oid: captions[oid] for oid in valid_ids if oid in captions}
    path = os.path.join(data_dir, "captions.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"[Save] {path} ({len(filtered)} objects)")


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu Cap3D ABO + ShapeNet")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent / "Cap3D_data"
        ),
        help="Đường dẫn thư mục Cap3D_data",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Bỏ qua bước giải nén (nếu đã giải nén trước đó)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Bỏ qua giải nén rendered images (tốn thời gian nhất)",
    )
    parser.add_argument(
        "--include-shapenet",
        action="store_true",
        help="[Deprecated] ShapeNet được auto-detect từ shapenet/ShapeNet_pcs/ - không cần truyền flag này",
    )
    parser.add_argument(
        "--no-shapenet",
        action="store_true",
        help="Bỏ qua ShapeNet, chỉ dùng ABO",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    # 1. Parse captions ABO
    print("\n--- BƯỚC 1: Parse captions ABO ---")
    captions = parse_captions(data_dir)

    # 2. Giải nén point clouds ABO
    print("\n--- BƯỚC 2: Giải nén point clouds ABO ---")
    if args.skip_extract:
        pc_dir = os.path.join(data_dir, "pointclouds")
        if os.path.exists(pc_dir):
            pc_ids = {f.replace(".ply", "") for f in os.listdir(pc_dir) if f.endswith(".ply")}
            print(f"[PointClouds] Bỏ qua, đã có {len(pc_ids)} files")
        else:
            pc_ids = set()
            print("[PointClouds] Chưa giải nén, thư mục không tồn tại")
    else:
        pc_ids = extract_pointclouds(data_dir)

    # 3. Giải nén rendered images ABO
    print("\n--- BƯỚC 3: Giải nén rendered images ABO ---")
    if args.skip_extract or args.skip_images:
        img_dir = os.path.join(data_dir, "rendered_images")
        if os.path.exists(img_dir):
            img_ids = {d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))}
            print(f"[RenderedImages] Bỏ qua, đã có {len(img_ids)} folders")
        else:
            img_ids = set()
            print("[RenderedImages] Chưa giải nén, thư mục không tồn tại")
    else:
        img_ids = extract_rendered_images(data_dir)

    # 4. Kiểm tra tính toàn vẹn ABO
    print("\n--- BƯỚC 4: Kiểm tra dữ liệu ABO ---")
    require_imgs = len(img_ids) > 0
    valid_ids = verify_data(captions, pc_ids, img_ids, require_images=require_imgs)

    # 4b. Gộp ShapeNet (auto-detect - không cần --include-shapenet)
    if not args.no_shapenet:
        print("\n--- BƯỚC 4b: Gộp ShapeNet ---")
        sn_captions = parse_shapenet_captions(data_dir)

        # Tìm ShapeNet point clouds (hỗ trợ cả .ply và .pt)
        sn_pc_dir = os.path.join(data_dir, "shapenet", "ShapeNet_pcs")
        sn_pc_ids = set()
        if os.path.exists(sn_pc_dir):
            sn_pc_ids = {f.replace(".ply", "") for f in os.listdir(sn_pc_dir) if f.endswith(".ply")}
            # Cũng tính .pt
            sn_pt_dir = os.path.join(data_dir, "shapenet", "pointclouds_pt")
            if os.path.exists(sn_pt_dir):
                sn_pc_ids |= {f.replace(".pt", "") for f in os.listdir(sn_pt_dir) if f.endswith(".pt")}

        if len(sn_pc_ids) > 0:
            sn_valid = verify_data(sn_captions, sn_pc_ids, set(), require_images=False)
            print(f"[ShapeNet] {len(sn_valid)} objects hợp lệ (caption + PC)")
            # Gộp vào danh sách chung
            all_valid_ids = sorted(set(valid_ids) | set(sn_valid))
            captions.update(sn_captions)
            print(f"[Gộp] ABO: {len(valid_ids)} + ShapeNet: {len(sn_valid)} = Tổng: {len(all_valid_ids)}")
            valid_ids = all_valid_ids
        else:
            print(f"[ShapeNet] Chưa có point clouds tại {sn_pc_dir} → bỏ qua")
            print("  Tải ShapeNet trước: PointCloud_zips_ShapeNet/compressed_pcs_00.zip")

    # 5. Chia splits
    print("\n--- BƯỚC 5: Chia train/val/test ---")
    splits = split_data(valid_ids)

    # 6. Lưu kết quả
    print("\n--- BƯỚC 6: Lưu files ---")
    save_splits(data_dir, splits)
    save_captions_json(data_dir, captions, valid_ids)

    # 7. Tóm tắt
    print("\n" + "=" * 60)
    print("HOÀN THÀNH CHUẨN BỊ DỮ LIỆU!")
    print(f"  - ABO point clouds: {os.path.join(data_dir, 'pointclouds')}")
    print(f"  - ShapeNet point clouds: {os.path.join(data_dir, 'shapenet', 'ShapeNet_pcs')}")
    print(f"  - Rendered images: {os.path.join(data_dir, 'rendered_images')}")
    print(f"  - Captions JSON: {os.path.join(data_dir, 'captions.json')}")
    print(f"  - Splits: {os.path.join(data_dir, 'splits')}")
    print(f"  - Tổng objects hợp lệ: {len(valid_ids)}")


if __name__ == "__main__":
    main()
