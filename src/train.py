"""
Training script cho Multimodal 3D model.
"""

import sys
from pathlib import Path

# Thêm project root vào sys.path để import 'src' package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import time

# Giới hạn CPU threads để giảm nhiệt
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import build_dataloader
from src.losses.losses import MultiModal3DLoss
from src.models.multimodal3d import build_model


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class WarmupCosineScheduler:
    """Cosine annealing scheduler với warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1.0 + cos_pi(progress))
            scale = max(scale, self.min_lr / max(self.base_lrs))

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale


def cos_pi(x):
    import math
    return math.cos(math.pi * x)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, log_every=10):
    model.train()
    total_loss = 0.0
    loss_details = {"chamfer_distance": 0.0, "coarse_cd": 0.0, "emd": 0.0, "color_loss": 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        captions = batch["caption"]
        gt_pc = batch["point_cloud"].to(device)

        # Tokenize text
        text_tokens = model.tokenize(captions).to(device)

        # Forward
        outputs = model(images, text_tokens)

        # Loss
        loss_dict = criterion(outputs, gt_pc)
        loss = loss_dict["total"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        for key in loss_details:
            if key in loss_dict:
                loss_details[key] += loss_dict[key].item()
        num_batches += 1

        # Logging
        if (batch_idx + 1) % log_every == 0:
            avg = total_loss / num_batches
            pbar.set_postfix(loss=f"{avg:.4f}", cd=f"{loss_details['chamfer_distance']/num_batches:.4f}")

    metrics = {k: v / num_batches for k, v in loss_details.items()}
    metrics["total"] = total_loss / num_batches
    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    loss_details = {"chamfer_distance": 0.0, "coarse_cd": 0.0, "emd": 0.0, "color_loss": 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    for batch in pbar:
        images = batch["image"].to(device)
        captions = batch["caption"]
        gt_pc = batch["point_cloud"].to(device)

        text_tokens = model.tokenize(captions).to(device)

        outputs = model(images, text_tokens)
        loss_dict = criterion(outputs, gt_pc)

        total_loss += loss_dict["total"].item()
        for key in loss_details:
            if key in loss_dict:
                loss_details[key] += loss_dict[key].item()
        num_batches += 1

        avg = total_loss / num_batches
        pbar.set_postfix(loss=f"{avg:.4f}", cd=f"{loss_details['chamfer_distance']/num_batches:.4f}")

    metrics = {k: v / num_batches for k, v in loss_details.items()}
    metrics["total"] = total_loss / num_batches
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal 3D Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data dir in config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override từ command line
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    log_cfg = config["logging"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Giới hạn tài nguyên để giảm nhiệt máy (chỉ trên local PC)
    torch.set_num_threads(4)
    if device.type == "cuda" and train_cfg.get("cooldown_seconds", 0) > 0:
        torch.cuda.set_per_process_memory_fraction(0.7, 0)
        print(f"GPU memory limited to 70% (local PC mode)")
    elif device.type == "cuda":
        print(f"GPU memory: full (Colab/server mode)")

    # Data (hỗ trợ cả path tuyệt đối trên Colab và relative trên PC)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = data_cfg["data_dir"]
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir)

    train_loader = build_dataloader(
        data_dir=data_dir, split="train",
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        num_points=data_cfg["num_points"],
        image_size=data_cfg["image_size"],
    )
    val_loader = build_dataloader(
        data_dir=data_dir, split="val",
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        num_points=data_cfg["num_points"],
        image_size=data_cfg["image_size"],
    )

    # Model
    model = build_model(config).to(device)
    params_info = model.get_num_params()
    print(f"Model params: total={params_info['total']:,}, trainable={params_info['trainable']:,}")

    # Loss
    criterion = MultiModal3DLoss(
        lambda_cd=loss_cfg.get("lambda_cd", 1.0),
        lambda_emd=loss_cfg.get("lambda_emd", 0.5),
        lambda_color=loss_cfg.get("lambda_color", 0.2),
        use_emd=loss_cfg.get("use_emd", True),
    )

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=train_cfg["warmup_epochs"],
        total_epochs=train_cfg["epochs"],
    )

    # TensorBoard
    log_dir = os.path.join(project_root, log_cfg["log_dir"])
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Checkpoint dir
    ckpt_dir = os.path.join(project_root, log_cfg["checkpoint_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resume
    start_epoch = 0
    best_val_cd = float("inf")
    patience_counter = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_cd = ckpt.get("best_val_cd", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"Resumed from epoch {start_epoch}, best_val_cd={best_val_cd:.4f}, patience={patience_counter}")

    # Training loop
    epochs = train_cfg["epochs"]
    patience = train_cfg["patience"]

    print(f"\nStarting training: {epochs} epochs, batch_size={data_cfg['batch_size']}")
    print("=" * 60)

    for epoch in range(start_epoch, epochs):
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}/{epochs} | LR: {current_lr:.2e}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1,
            log_every=log_cfg.get("log_every", 10),
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch + 1)

        # Log
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        print(f"  Train - Loss: {train_metrics['total']:.4f}, CD: {train_metrics['chamfer_distance']:.4f}")
        print(f"  Val   - Loss: {val_metrics['total']:.4f}, CD: {val_metrics['chamfer_distance']:.4f}")

        # Save best model
        val_cd = val_metrics["chamfer_distance"]
        if val_cd < best_val_cd:
            best_val_cd = val_cd
            patience_counter = 0
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_cd": best_val_cd,
                "patience_counter": patience_counter,
                "config": config,
            }, best_path)
            print(f"  ** Best model saved! CD={best_val_cd:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        # Save last model
        last_path = os.path.join(ckpt_dir, "last_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_cd": best_val_cd,
            "patience_counter": patience_counter,
            "config": config,
        }, last_path)

        # Cooldown giữa epoch để giảm nhiệt máy
        cooldown = train_cfg.get("cooldown_seconds", 30)
        if cooldown > 0:
            print(f"  Cooling down {cooldown}s...")
            time.sleep(cooldown)

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}! Best CD={best_val_cd:.4f}")
            break

    writer.close()
    print(f"\nTraining complete! Best val CD: {best_val_cd:.4f}")


if __name__ == "__main__":
    main()
