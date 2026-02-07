"""
OPTIMIZED Training script - Mamba-UNet
- Learning rate warmup
- Gradient clipping
- Better monitoring
- Early stopping
- Model checkpointing
"""
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from utils.metrics import compute_all_metrics


from models.mamba_unet import create_mamba_unet
from datasets.tooth_dataset import ToothDataset
from utils.losses import get_loss
from utils.metrics import dice_coefficient, iou_score

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, warmup_epochs=5, base_lr=0.0001):
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} - Training')
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        # WARMUP Learning Rate
        if epoch <= warmup_epochs:
            warmup_factor = (epoch - 1 + batch_idx / len(loader)) / warmup_epochs
            lr = base_lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # GRADIENT CLIPPING - Tránh exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
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
    return total_loss/n, total_dice/n, total_iou/n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_sum = None

    for images, masks in tqdm(loader, desc='Validation'):
        images, masks = images.to(device), masks.to(device)

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
    avg_metrics = {k: v / n for k, v in metrics_sum.items()}
    return total_loss / n, avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/d2')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    # ADVANCED OPTIONS
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=30)
    parser.add_argument('--embed_dim', type=int, default=96, help='96, 128, or 192')
    
    args = parser.parse_args()
    
    # Setup
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
        cudnn.deterministic = False  # Faster training
    
    print("="*80)
    print(" MAMBA-UNET OPTIMIZED TRAINING")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*80)
    
    # Data
    train_ds = ToothDataset(args.data_path, 'train', args.img_size, augment=True)
    val_ds = ToothDataset(args.data_path, 'val', args.img_size, augment=False)
    
    train_loader = DataLoader(
        train_ds, 
        args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True  # Faster data loading
    )
    val_loader = DataLoader(
        val_ds, 
        1, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    
    # Model
    model = create_mamba_unet(
        in_chans=1, 
        num_classes=2, 
        img_size=args.img_size,
       
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.2f}M")
    print(f"Trainable params: {trainable_params/1e6:.2f}M")
    
    # Training setup
    criterion = get_loss(version='improved')

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Cosine Annealing with Warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01
    )
    
    scaler = GradScaler()
    
    # Training loop
    best_dice = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [],
        'val_loss': [], 'val_dice': [], 'val_iou': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, args.warmup_epochs, args.lr
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        val_dice = val_metrics['dice']
        val_iou = val_metrics['iou']


        
        # Scheduler step (after warmup)
        if epoch > args.warmup_epochs:
            scheduler.step()
        
        # Log
        print(
            f"Val: Loss={val_loss:.4f} | "
            f"Dice={val_dice:.4f} | IoU={val_iou:.4f} | "
            f"Prec={val_metrics['precision']:.4f} | "
            f"Rec={val_metrics['recall']:.4f} | "
            f"Spec={val_metrics['specificity']:.4f} | "
            f"F1={val_metrics['f1']:.4f}"
        )

        
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'dice': best_dice,
                'iou': val_iou,
                'args': args
            }, os.path.join(save_dir, 'best.pth'))
            
            print(f"✅ BEST MODEL SAVED! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n🛑 EARLY STOPPING at epoch {epoch}")
            break
        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'dice': val_dice
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth'))
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_dice'], label='Train')
    axes[0, 1].plot(history['val_dice'], label='Val')
    axes[0, 1].set_title('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['train_iou'], label='Train')
    axes[1, 0].plot(history['val_iou'], label='Val')
    axes[1, 0].set_title('IoU Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].text(0.5, 0.5, f'Best Dice: {best_dice:.4f}', 
                    ha='center', va='center', fontsize=20, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETED!")
    print(f"Best Dice Score: {best_dice:.4f}")
    print(f"Models saved in: {save_dir}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()