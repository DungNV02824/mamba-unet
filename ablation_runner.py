"""
Ablation runner for Mamba-UNet research.

This script trains four model variants using the same hyperparameters
from config_optimization.py and saves weights under checkpoints/ablation/.
It then computes Dice Score on the hold-out validation split and prints
a comparison table.
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config_optimization import CONFIG_BALANCED, CONFIG_FAST, CONFIG_QUALITY, CONFIG_LOW_MEMORY
from datasets.tooth_dataset import ToothDataset
from utils.losses import get_loss
from utils.metrics import dice_coefficient, compute_all_metrics
from models.ablation_models import (
    UNet_CNN_CNN,
    UNet_Mamba_CNN,
    UNet_CNN_Mamba,
    UNet_Full_Mamba,
)


CONFIG_MAP = {
    "balanced": CONFIG_BALANCED,
    "fast": CONFIG_FAST,
    "quality": CONFIG_QUALITY,
    "low_memory": CONFIG_LOW_MEMORY,
}

MODEL_REGISTRY = {
    "UNet_CNN_CNN": UNet_CNN_CNN,
    "UNet_Mamba_CNN": UNet_Mamba_CNN,
    "UNet_CNN_Mamba": UNet_CNN_Mamba,
    "UNet_Full_Mamba": UNet_Full_Mamba,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make_data_loaders(data_path, img_size, batch_size):
    train_ds = ToothDataset(data_path, split="train", img_size=img_size, augment=True)
    val_ds = ToothDataset(data_path, split="val", img_size=img_size, augment=False)

    if hasattr(train_ds, "sample_weights") and train_ds.sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=train_ds.sample_weights,
            num_samples=len(train_ds),
            replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, warmup_epochs, base_lr):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} - Training", leave=False)

    for batch_idx, (images, masks) in enumerate(pbar, start=1):
        images = images.to(device)
        masks = masks.to(device)

        if epoch <= warmup_epochs:
            warmup_factor = (epoch - 1 + batch_idx / len(loader)) / max(1, warmup_epochs)
            lr = base_lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            iou = compute_all_metrics(outputs, masks)["iou"]

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        current_lr = optimizer.param_groups[0]["lr"]

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice:.4f}",
            "lr": f"{current_lr:.6f}"
        })

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics_sum = None

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        total_loss += loss.item()
        metrics = compute_all_metrics(outputs, masks)

        if metrics_sum is None:
            metrics_sum = {k: metrics[k] for k in metrics}
        else:
            for k in metrics:
                metrics_sum[k] += metrics[k]

    n = len(loader)
    avg_metrics = {k: v / n for k, v in metrics_sum.items()} if metrics_sum is not None else {}
    return total_loss / max(1, n), avg_metrics


@torch.no_grad()
def evaluate_dice(model, loader, device):
    model.eval()
    total = 0.0
    for images, masks in tqdm(loader, desc="Test Evaluation", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        total += dice_coefficient(outputs, masks)
    return total / max(1, len(loader))


def train_model_variant(model_name, model_cls, config, data_path, save_dir, device):
    print(f"\n=== TRAINING {model_name} ===")

    model = model_cls(
        img_size=config.get("img_size", 512),
        in_chans=1,
        num_classes=2,
        embed_dim=config["embed_dim"],
        depths=config["depths"],
        drop_path_rate=config["drop_path_rate"],
        patch_size=config.get("patch_size", 4)
    ).to(device)

    train_loader, val_loader = make_data_loaders(data_path, config.get("img_size", 512), config["batch_size"])
    criterion = get_loss(version="improved")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config["epochs"] - config["warmup_epochs"]),
        eta_min=config["lr"] * 0.01
    )
    scaler = GradScaler()

    best_dice = 0.0
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        train_loss, train_dice, train_iou = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            config["warmup_epochs"],
            config["lr"]
        )

        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_dice = val_metrics.get("dice", 0.0)
        val_iou = val_metrics.get("iou", 0.0)

        if epoch > config["warmup_epochs"]:
            scheduler.step()

        print(f"Validation  Loss: {val_loss:.4f}  Dice: {val_dice:.4f}  IoU: {val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
            print(f"✅ Saved best checkpoint for {model_name}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stop_patience"]:
                print("🛑 Early stopping")
                break

    checkpoint_path = os.path.join(save_dir, f"{model_name}.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded best checkpoint for {model_name}")

    test_dice = evaluate_dice(model, val_loader, device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6

    return {
        "name": model_name,
        "dice": test_dice,
        "params": num_params,
        "best_val_dice": best_dice,
        "checkpoint": checkpoint_path,
    }


def print_summary(results):
    print("\n" + "=" * 72)
    print("Ablation Results")
    print("=" * 72)
    print(f"{'Model':30} | {'Params (M)':>10} | {'Dice':>8}")
    print("" + "-" * 72)
    for item in results:
        print(f"{item['name']:30} | {item['params']:10.2f} | {item['dice']:8.4f}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Ablation study runner for Mamba-UNet variants")
    parser.add_argument("--data_path", type=str, default="./data/d2")
    parser.add_argument("--config", type=str, default="balanced",
                        choices=list(CONFIG_MAP.keys()))
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the number of training epochs from the selected config")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/ablation")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = dict(CONFIG_MAP[args.config])
    if args.epochs is not None:
        config["epochs"] = args.epochs

    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        cudnn.benchmark = True
        cudnn.deterministic = False

    print("\n=== ABLATION STUDY ===")
    print(f"Data path  : {args.data_path}")
    print(f"Config     : {args.config}")
    if args.epochs is not None:
        print(f"Overwrite epochs: {args.epochs}")
    print(f"Save dir   : {args.save_dir}")
    print(f"Device     : {device}")
    print(f"Embed dim  : {config['embed_dim']}")
    print(f"Depths     : {config['depths']}")
    print(f"Batch size : {config['batch_size']}")
    print(f"Epochs     : {config['epochs']}")
    print("=" * 72)

    results = []
    for model_name, model_cls in MODEL_REGISTRY.items():
        result = train_model_variant(
            model_name=model_name,
            model_cls=model_cls,
            config=config,
            data_path=args.data_path,
            save_dir=args.save_dir,
            device=device,
        )
        results.append(result)

    print_summary(results)

    summary_path = os.path.join(args.save_dir, "ablation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("Model,Params(M),Dice\n")
        for item in results:
            summary_file.write(f"{item['name']},{item['params']:.2f},{item['dice']:.4f}\n")

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
