"""
Training script - Tương thích với mamba-ssm 1.2.0
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

from models.mamba_unet import create_mamba_unet
from datasets.tooth_dataset import ToothDataset
from utils.losses import CombinedLoss
from utils.metrics import dice_coefficient, iou_score

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
        
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0
    
    for images, masks in tqdm(loader, desc='Validation'):
        images, masks = images.to(device), masks.to(device)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, masks)
        total_iou += iou_score(outputs, masks)
    
    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/tooth_dataset/d2')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
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
    
    print("="*80)
    print("🚀 MAMBA-UNET TRAINING (Pure Implementation)")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print("="*80)
    
    # Data
    train_ds = ToothDataset(args.data_path, 'train', args.img_size, augment=True)
    val_ds = ToothDataset(args.data_path, 'val', args.img_size, augment=False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = create_mamba_unet(in_chans=1, num_classes=2, img_size=args.img_size).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    
    # Training
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler()
    
    best_dice = 0
    history = {'train_loss': [], 'val_dice': []}
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_dice, _ = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_dice, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, Dice={val_dice:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_dice)
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dice': best_dice
            }, os.path.join(save_dir, 'best.pth'))
            print(f"✓ Best model saved! Dice: {best_dice:.4f}")
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history['train_loss'])
    plt.title('Train Loss')
    plt.subplot(122)
    plt.plot(history['val_dice'])
    plt.title('Val Dice')
    plt.savefig(os.path.join(save_dir, 'curves.png'))
    print(f"\n✅ Done! Best Dice: {best_dice:.4f}")

if __name__ == '__main__':
    main()