"""
Evaluation script: tính metrics trên test set.
- Chamfer Distance (CD)
- F-Score@0.01
"""

import sys
from pathlib import Path

# Thêm project root vào sys.path để import 'src' package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.data.dataset import build_dataloader
from src.losses.losses import chamfer_distance
from src.models.multimodal3d import build_model


def f_score(pred_xyz, gt_xyz, threshold=0.01):
    """
    F-Score tại threshold τ.

    Args:
        pred_xyz: [batch, N, 3]
        gt_xyz: [batch, M, 3]
        threshold: bán kính τ
    Returns:
        f_score: scalar
    """
    # Pairwise distances
    diff = pred_xyz.unsqueeze(2) - gt_xyz.unsqueeze(1)
    dist_matrix = (diff ** 2).sum(-1)  # [B, N, M]

    # Precision: % pred points có ít nhất 1 gt point trong bán kính
    min_dist_pred = dist_matrix.min(dim=2)[0]  # [B, N]
    precision = (min_dist_pred < threshold).float().mean(dim=1)  # [B]

    # Recall: % gt points có ít nhất 1 pred point trong bán kính
    min_dist_gt = dist_matrix.min(dim=1)[0]  # [B, M]
    recall = (min_dist_gt < threshold).float().mean(dim=1)  # [B]

    # F-score
    f = 2 * precision * recall / (precision + recall + 1e-8)
    return f.mean()


def clip_similarity(images, captions, clip_model, tokenizer, device):
    """
    Tính CLIP similarity giữa ảnh render và text.

    Args:
        images: list of PIL images (rendered từ predicted point clouds)
        captions: list of str
        clip_model: CLIP model
        tokenizer: CLIP tokenizer
        device: torch device
    Returns:
        mean similarity: scalar
    """
    import open_clip
    from PIL import Image
    from torchvision import transforms

    _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")

    similarities = []
    for img, cap in zip(images, captions):
        if img is None:
            continue
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        tokens = tokenizer([cap]).to(device)

        with torch.no_grad():
            img_feat = clip_model.encode_image(img_tensor)
            txt_feat = clip_model.encode_text(tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat * txt_feat).sum().item()
        similarities.append(sim)

    return np.mean(similarities) if similarities else 0.0


def render_point_cloud_to_image(point_cloud, image_size=224):
    """
    Render point cloud thành ảnh 2D đơn giản (orthographic projection).

    Args:
        point_cloud: numpy [N, 6] (XYZ + RGB)
        image_size: kích thước ảnh đầu ra
    Returns:
        PIL Image
    """
    from PIL import Image

    xyz = point_cloud[:, :3]
    rgb = (point_cloud[:, 3:] * 255).astype(np.uint8)

    # Normalize về [0, image_size]
    xyz_norm = xyz.copy()
    xyz_min = xyz_norm.min(axis=0)
    xyz_max = xyz_norm.max(axis=0)
    extent = xyz_max - xyz_min
    extent[extent < 1e-8] = 1.0
    xyz_norm = (xyz_norm - xyz_min) / extent
    xyz_norm = (xyz_norm * (image_size - 10) + 5).astype(int)

    # Tạo ảnh
    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    # Sắp xếp theo depth (z) để vẽ từ xa đến gần
    order = np.argsort(-xyz[:, 1])  # Y-axis as depth
    for i in order:
        x, y = xyz_norm[i, 0], xyz_norm[i, 2]
        if 0 <= x < image_size and 0 <= y < image_size:
            img[y, x] = rgb[i]

    return Image.fromarray(img)


@torch.no_grad()
def evaluate(model, dataloader, device, threshold=0.01):
    """Chạy evaluation trên dataloader."""
    model.eval()

    all_cd = []
    all_fscore = []

    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        images = batch["image"].to(device)
        captions = batch["caption"]
        gt_pc = batch["point_cloud"].to(device)

        text_tokens = model.tokenize(captions).to(device)
        outputs = model(images, text_tokens)

        pred_xyz = outputs["fine_xyz"]
        gt_xyz = gt_pc[:, :, :3]

        # Chamfer Distance
        cd = chamfer_distance(pred_xyz, gt_xyz)
        all_cd.append(cd.item())

        # F-Score
        fs = f_score(pred_xyz, gt_xyz, threshold=threshold)
        all_fscore.append(fs.item())

        pbar.set_postfix(cd=f"{np.mean(all_cd):.4f}", f_score=f"{np.mean(all_fscore):.4f}")

    mean_cd = np.mean(all_cd)
    mean_fscore = np.mean(all_fscore)

    return {
        "chamfer_distance": mean_cd,
        "f_score_001": mean_fscore,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal 3D Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, default="results/evaluation.json")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(config).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"WARNING: Checkpoint not found at {args.checkpoint}, using random weights")

    # Data
    project_root = Path(__file__).resolve().parent.parent
    data_dir = os.path.join(project_root, config["data"]["data_dir"])

    dataloader = build_dataloader(
        data_dir=data_dir, split=args.split,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        num_points=config["data"]["num_points"],
        image_size=config["data"]["image_size"],
    )

    # Evaluate
    results = evaluate(model, dataloader, device)

    print("\n" + "=" * 40)
    print(f"Evaluation Results ({args.split} set)")
    print(f"  Chamfer Distance: {results['chamfer_distance']:.4f}")
    print(f"  F-Score@0.01:     {results['f_score_001']:.4f}")
    print("=" * 40)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
