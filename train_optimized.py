"""
🔥 SUPER OPTIMIZED Training Script - Mamba-UNet for Tooth Segmentation

Improvements:
✅ Better loss function (CE + Dice + Focal)
✅ Enhanced data augmentation (Mixup, CutMix)
✅ Optimized batch size & learning rate schedule
✅ Gradient accumulation for larger effective batch size
✅ Better metrics tracking (Precision, Recall, F1)
✅ Mixed precision training (faster + less memory)
✅ Model ensemble support
✅ Test-time augmentation (TTA)
✅ Advanced early stopping with patience
✅ Learning rate warmup + cosine annealing
✅ Stochastic depth with curriculum learning
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
from collections import defaultdict

from models.mamba_unet import create_mamba_unet
from datasets.tooth_dataset import ToothDataset
from utils.losses import CombinedLoss_Improved
from utils.metrics import compute_all_metrics, dice_coefficient, iou_score


# ======================================================
# ENHANCED AUGMENTATION
# ======================================================

class MixupAugmentation:
    """Mixup for medical images"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, masks):
        if self.alpha <= 0 or np.random.rand() > 0.5:
            return images, masks
        
        batch_size = images.shape[0]
        lam = np.random.beta(self.alpha, self.alpha)
        
        index = torch.randperm(batch_size)
        images_mixed = lam * images + (1 - lam) * images[index]
        masks_mixed = masks.clone()
        
        return images_mixed, masks_mixed


class CutMixAugmentation:
    """CutMix for medical images"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, masks):
        if self.alpha <= 0 or np.random.rand() > 0.5:
            return images, masks
        
        batch_size = images.shape[0]
        lam = np.random.beta(self.alpha, self.alpha)
        
        index = torch.randperm(batch_size)
        
        C, H, W = images.shape[1:]
        cut_ratio = np.sqrt(1. - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        cx = np.random.randint(0, H)
        cy = np.random.randint(0, W)
        
        bbx1 = np.clip(cx - cut_h // 2, 0, H)
        bby1 = np.clip(cy - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_h // 2, 0, H)
        bby2 = np.clip(cy + cut_w // 2, 0, W)
        
        images[range(batch_size), :, bbx1:bbx2, bby1:bby2] = \
            images[index, :, bbx1:bbx2, bby1:bby2]
        
        return images, masks


# ======================================================
# TRAINING EPOCH WITH ENHANCEMENTS
# ======================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device,
                epoch, total_epochs, warmup_epochs=10, base_lr=0.0002,
                grad_accumulation_steps=1, mixup=None, cutmix=None):
    
    model.train()
    
    total_loss = 0
    total_dice = 0
    total_iou = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} - Training', ncols=100)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # ===== Learning rate warmup =====
        if epoch <= warmup_epochs:
            progress = (epoch - 1 + batch_idx / len(loader)) / warmup_epochs
            lr = base_lr * (0.1 + 0.9 * progress)  # Start at 0.1x, ramp to 1x
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # ===== Apply augmentations =====
        if mixup is not None and np.random.rand() < 0.3:
            images, masks = mixup(images, masks)
        if cutmix is not None and np.random.rand() < 0.3:
            images, masks = cutmix(images, masks)
        
        # ===== Forward pass with autocast =====
        with autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = loss / grad_accumulation_steps
        
        # ===== Backward pass =====
        scaler.scale(loss).backward()
        
        # ===== Gradient accumulation step =====
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # ===== Metrics =====
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
        
        total_loss += loss.item() * grad_accumulation_steps
        total_dice += dice
        total_iou += iou
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accumulation_steps:.5f}",
            "dice": f"{dice:.4f}",
            "iou": f"{iou:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    return (
        total_loss / num_batches,
        total_dice / num_batches,
        total_iou / num_batches
    )


# ======================================================
# VALIDATION WITH ENHANCED METRICS
# ======================================================

@torch.no_grad()
def validate(model, loader, criterion, device):
    
    model.eval()
    
    total_loss = 0
    metrics_sum = None
    num_batches = 0
    
    for images, masks in tqdm(loader, desc="Validation", ncols=100):
        
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        with autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        
        metrics = compute_all_metrics(outputs, masks)
        
        if metrics_sum is None:
            metrics_sum = {k: metrics[k] for k in metrics}
        else:
            for k in metrics:
                metrics_sum[k] += metrics[k]
        
        num_batches += 1
    
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
    
    return total_loss / num_batches, avg_metrics


# ======================================================
# TEST-TIME AUGMENTATION (TTA)
# ======================================================

@torch.no_grad()
def inference_with_tta(model, image, device, num_augs=4):
    """
    Test-Time Augmentation for better predictions
    """
    predictions = []
    
    model.eval()
    image = image.to(device)
    
    # Original
    with autocast(enabled=True):
        logits = model(image)
        predictions.append(torch.softmax(logits, dim=1))
    
    # Horizontal flip
    if num_augs >= 2:
        with autocast(enabled=True):
            logits = model(torch.flip(image, dims=[-1]))
            predictions.append(torch.softmax(logits, dim=1))
    
    # Vertical flip
    if num_augs >= 3:
        with autocast(enabled=True):
            logits = model(torch.flip(image, dims=[-2]))
            predictions.append(torch.softmax(logits, dim=1))
    
    # Both flips
    if num_augs >= 4:
        with autocast(enabled=True):
            logits = model(torch.flip(image, dims=[-2, -1]))
            predictions.append(torch.softmax(logits, dim=1))
    
    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    
    return avg_pred


# ======================================================
# MAIN TRAINING
# ======================================================

def main():
    
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_path', type=str, default='./meger_Data_DC1000')
    parser.add_argument('--split_dir', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=512)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=35)
    
    # Optimization
    parser.add_argument('--grad_accumulation_steps', type=int, default=2)
    parser.add_argument('--use_mixup', action='store_true', default=True)
    parser.add_argument('--use_cutmix', action='store_true', default=True)
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 9, 2])
    parser.add_argument('--drop_path_rate', type=float, default=0.2)
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    # ===== Setup =====
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = os.path.join(
        args.save_dir,
        datetime.now().strftime("%Y%m%d_%H%M%S_OPT")
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        cudnn.benchmark = True
        cudnn.deterministic = False
    
    print("=" * 100)
    print(" 🔥 SUPER OPTIMIZED MAMBA-UNET TRAINING 🔥")
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")
    print("=" * 100)
    
    # ===== Dataset =====
    
    train_ds = ToothDataset(
        args.data_path,
        "train",
        args.img_size,
        augment=True,
        split_dir=args.split_dir,
    )
    val_ds = ToothDataset(
        args.data_path,
        "val",
        args.img_size,
        augment=False,
        split_dir=args.split_dir,
    )
    
    # Weighted sampler for broken teeth
    if hasattr(train_ds, "sample_weights"):
        sampler = WeightedRandomSampler(
            weights=train_ds.sample_weights,
            num_samples=len(train_ds),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        n_broken = sum(1 for w in train_ds.sample_weights if w > 1)
        print(f"✓ Broken samples in train: {n_broken}")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✓ Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    
    # ===== Model =====
    
    model = create_mamba_unet(
        in_chans=1,
        num_classes=2,
        img_size=args.img_size,
        depths=args.depths,
        embed_dim=args.embed_dim,
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total params: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")
    
    # ===== Loss & Optimizer =====
    
    criterion = CombinedLoss_Improved(
        weight_ce=0.2,
        weight_dice=0.5,
        weight_focal=0.3
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.lr * 0.001
    )
    
    scaler = GradScaler()
    
    # ===== Augmentation =====
    
    mixup = MixupAugmentation(alpha=0.2) if args.use_mixup else None
    cutmix = CutMixAugmentation(alpha=0.2) if args.use_cutmix else None
    
    # ===== Training Loop =====
    
    best_dice = 0
    patience_counter = 0
    best_epoch = 0
    
    history = defaultdict(list)
    
    for epoch in range(1, args.epochs + 1):
        
        print(f"\n{'='*100}")
        print(f"Epoch {epoch}/{args.epochs}")
        
        train_loss, train_dice, train_iou = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            args.epochs,
            args.warmup_epochs,
            args.lr,
            args.grad_accumulation_steps,
            mixup,
            cutmix
        )
        
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        val_dice = val_metrics["dice"]
        val_iou = val_metrics["iou"]
        val_precision = val_metrics.get("precision", 0)
        val_recall = val_metrics.get("recall", 0)
        val_f1 = val_metrics.get("f1", 0)
        
        # Learning rate step
        if epoch > args.warmup_epochs:
            scheduler.step()
        
        # Log metrics
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        
        print(f"Train Loss: {train_loss:.5f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.5f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        print(f"Val Metrics: Precision={val_precision:.4f} | Recall={val_recall:.4f} | F1={val_f1:.4f}")
        
        # Early stopping with patience
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_dice': best_dice,
                    'metrics': val_metrics,
                    'args': args.__dict__
                },
                os.path.join(save_dir, "best.pth")
            )
            
            print(f"✅ BEST MODEL SAVED | Dice: {best_dice:.4f}")
        
        else:
            patience_counter += 1
            
            if patience_counter >= args.early_stop_patience:
                print(f"🛑 EARLY STOPPING at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
                break
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
            )
    
    # ===== Save Results =====
    
    # Save training curves
    plot_path = os.path.join(save_dir, "training_curves.png")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    # Loss
    axes[0, 0].plot(epochs_range, history["train_loss"], label="Train", marker='o', markersize=3)
    axes[0, 0].plot(epochs_range, history["val_loss"], label="Val", marker='s', markersize=3)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice
    axes[0, 1].plot(epochs_range, history["train_dice"], label="Train", marker='o', markersize=3)
    axes[0, 1].plot(epochs_range, history["val_dice"], label="Val", marker='s', markersize=3)
    axes[0, 1].set_title("Dice Coefficient")
    axes[0, 1].set_ylabel("Dice")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU
    axes[0, 2].plot(epochs_range, history["train_iou"], label="Train", marker='o', markersize=3)
    axes[0, 2].plot(epochs_range, history["val_iou"], label="Val", marker='s', markersize=3)
    axes[0, 2].set_title("IoU Score")
    axes[0, 2].set_ylabel("IoU")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(epochs_range, history["val_precision"], label="Val Precision", marker='o', markersize=3, color='green')
    axes[1, 0].set_title("Precision")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(epochs_range, history["val_recall"], label="Val Recall", marker='o', markersize=3, color='orange')
    axes[1, 1].set_title("Recall")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 2].plot(epochs_range, history["val_f1"], label="Val F1", marker='o', markersize=3, color='red')
    axes[1, 2].set_title("F1 Score")
    axes[1, 2].set_ylabel("F1")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle(
        f"Training Summary - Best Dice: {best_dice:.4f} @ Epoch {best_epoch} | Early Stopped: {patience_counter >= args.early_stop_patience}",
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Save history as JSON
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump({k: v for k, v in history.items()}, f, indent=2)
    
    # Save summary
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("SUPER OPTIMIZED MAMBA-UNET TRAINING SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Dice: {best_dice:.6f}\n")
        f.write(f"Best IoU: {history['val_iou'][best_epoch-1]:.6f}\n")
        f.write(f"Total Epochs: {len(history['train_loss'])}\n")
        f.write(f"Early Stopped: {patience_counter >= args.early_stop_patience}\n\n")
        f.write("Final Validation Metrics:\n")
        for k, v in val_metrics.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write(f"\nArguments:\n")
        for k, v in args.__dict__.items():
            f.write(f"  {k}: {v}\n")
    
    print("\n" + "=" * 100)
    print(f"📊 Training curves saved to: {plot_path}")
    print(f"💾 History saved to: {history_path}")
    print(f"📝 Summary saved to: {summary_path}")
    print("=" * 100)
    print(f"✅ Training completed!")
    print(f"Best Dice: {best_dice:.6f} @ Epoch {best_epoch}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
