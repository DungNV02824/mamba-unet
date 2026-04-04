

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from datetime import datetime

from utils.metrics import compute_all_metrics, dice_coefficient, iou_score
from models.mamba_unet import create_mamba_unet
from datasets.tooth_dataset import ToothDataset
from utils.losses import get_loss


# ==========================
# TRAIN ONE EPOCH
# ==========================
def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, warmup_epochs, base_lr):
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} - Training')

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Warmup LR
        if epoch <= warmup_epochs:
            warmup_factor = (epoch + batch_idx / len(loader)) / warmup_epochs
            warmup_factor = min(warmup_factor, 1.0)
            lr = base_lr * warmup_factor
            for g in optimizer.param_groups:
                g['lr'] = lr

        # Forward
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        # Metrics (IMPORTANT: dùng outputs, KHÔNG dùng preds)
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ==========================
# VALIDATION
# ==========================
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_sum = None

    for images, masks in tqdm(loader, desc='Validation'):
        images = images.to(device)
        masks = masks.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        total_loss += loss.item()

        # IMPORTANT: truyền outputs
        metrics = compute_all_metrics(outputs, masks)

        if metrics_sum is None:
            metrics_sum = metrics
        else:
            for k in metrics:
                metrics_sum[k] += metrics[k]

    n = len(loader)
    avg_metrics = {k: v / n for k, v in metrics_sum.items()}
    return total_loss / n, avg_metrics


# ==========================
# MAIN
# ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/d2')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--embed_dim', type=int, default=64)

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = os.path.join(args.save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)

    # Seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        cudnn.benchmark = True

    print("=" * 60)
    print("MAMBA-UNET TRAINING (FINAL)")
    print(f"Device: {device} | Torch: {torch.__version__}")
    print("=" * 60)

    # ==========================
    # DATA
    # ==========================
    train_ds = ToothDataset(args.data_path, 'train', args.img_size, augment=True)
    val_ds = ToothDataset(args.data_path, 'val', args.img_size, augment=False)

    if train_ds.sample_weights is not None:
        sampler = WeightedRandomSampler(
            train_ds.sample_weights,
            num_samples=len(train_ds.sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ==========================
    # MODEL
    # ==========================
    model = create_mamba_unet(
        in_chans=1,
        num_classes=2,
        img_size=args.img_size,
        embed_dim=args.embed_dim
    ).to(device)

    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ==========================
    # LOSS + OPTIM
    # ==========================
    criterion = get_loss('improved')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01
    )

    scaler = GradScaler()

    # ==========================
    # TRAIN LOOP
    # ==========================
    best_dice = 0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args.warmup_epochs, args.lr
        )

        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        if epoch >= args.warmup_epochs:
            scheduler.step()

        val_dice = val_metrics['dice']

        print(f"Val Dice: {val_dice:.4f} | IoU: {val_metrics['iou']:.4f}")

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            patience = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            print("✅ Saved best model")
        else:
            patience += 1
            print(f"⏳ No improve ({patience})")

        # Early stopping
        if patience >= args.early_stop_patience:
            print("🛑 Early stopping")
            break

    print(f"\n🔥 BEST DICE: {best_dice:.4f}")
    print(f"📁 Saved at: {save_dir}")


if __name__ == '__main__':
    main()