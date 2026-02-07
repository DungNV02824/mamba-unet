"""
OPTIMIZED Dataset loader cho ảnh răng panoramic
- Augmentation mạnh hơn
- CLAHE preprocessing
- Better normalization
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class ToothDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=512, augment=True):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'img')
        self.mask_dir = os.path.join(root_dir, 'masks_machine')
        self.img_size = img_size
        self.split = split
        
        # Lấy danh sách file
        all_images = sorted([f for f in os.listdir(self.img_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # SHUFFLE trước khi chia để tránh bias
        import random
        random.seed(42)
        random.shuffle(all_images)
        
        # Chia train/val (80/20)
        n_total = len(all_images)
        n_train = int(0.8 * n_total)
        
        if split == 'train':
            self.image_list = all_images[:n_train]
        else:
            self.image_list = all_images[n_train:]
        
        # AUGMENTATION MẠNH HƠN
        if augment and split == 'train':
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                
                # Geometric
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.15, 
                    rotate_limit=20, 
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.6
                ),
                A.ElasticTransform(
                    alpha=1, 
                    sigma=50, 
                    alpha_affine=50, 
                    p=0.3
                ),
                
                # Intensity
                A.RandomBrightnessContrast(
                    brightness_limit=0.25, 
                    contrast_limit=0.25, 
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                
                # CLAHE - Cải thiện contrast cho X-ray
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # Luôn dùng cho val
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])
        
        print(f"✓ Loaded {len(self.image_list)} {split} samples")
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        mask_name = os.path.splitext(img_name)[0] + '.png'
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load
        image = np.array(Image.open(img_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Binarize mask
        mask = (mask > 127).astype(np.uint8)  # Threshold = 127 cho chắc chắn
        
        # Transform
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        
        return image.float(), mask.long()