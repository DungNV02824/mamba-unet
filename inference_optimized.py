"""
🔥 OPTIMIZED INFERENCE SCRIPT - With TTA & Post-processing

Features:
✅ Test-Time Augmentation (TTA) - 4 flips for better predictions
✅ Ensemble predictions
✅ Post-processing (morphological operations)
✅ Confidence thresholding
✅ GPU memory efficient
✅ Batch inference
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import argparse
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.mamba_unet import create_mamba_unet


class OptimizedInference:
    """High-performance inference with TTA & post-processing"""
    
    def __init__(self, model_path, device='cuda', img_size=512):
        self.device = device
        self.img_size = img_size
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
    
    def _load_model(self, model_path):
        """Load model from checkpoint"""
        model = create_mamba_unet(
            in_chans=1,
            num_classes=2,
            img_size=self.img_size,
            depths=[2, 2, 9, 2],
            embed_dim=96,
            drop_path_rate=0.2,
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _get_transform(self):
        """Get preprocessing transform"""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    
    @torch.no_grad()
    def predict_tta(self, image, num_augs=4):
        """
        Predict with Test-Time Augmentation
        
        num_augs: 1=original, 2=+hflip, 3=+vflip, 4=+both flips
        """
        image = image.to(self.device)
        predictions = []
        
        # Original
        with torch.autocast('cuda'):
            logits = self.model(image)
            predictions.append(torch.softmax(logits, dim=1))
        
        # Horizontal flip
        if num_augs >= 2:
            with torch.autocast('cuda'):
                logits = self.model(torch.flip(image, dims=[-1]))
                preds = torch.softmax(logits, dim=1)
                preds = torch.flip(preds, dims=[-1])
                predictions.append(preds)
        
        # Vertical flip
        if num_augs >= 3:
            with torch.autocast('cuda'):
                logits = self.model(torch.flip(image, dims=[-2]))
                preds = torch.softmax(logits, dim=1)
                preds = torch.flip(preds, dims=[-2])
                predictions.append(preds)
        
        # Both flips
        if num_augs >= 4:
            with torch.autocast('cuda'):
                logits = self.model(torch.flip(image, dims=[-2, -1]))
                preds = torch.softmax(logits, dim=1)
                preds = torch.flip(preds, dims=[-2, -1])
                predictions.append(preds)
        
        # Average predictions
        avg_pred = torch.stack(predictions).mean(dim=0)
        
        return avg_pred
    
    @torch.no_grad()
    def predict_single(self, image_path, use_tta=True, tta_augs=4):
        """
        Predict on single image
        
        Returns:
            pred_mask: binary mask (H, W)
            confidence: confidence map (H, W)
        """
        # Read image
        img = np.array(Image.open(image_path).convert('L'), dtype=np.uint8)
        
        # Transform
        transformed = self.transform(image=img)
        img_tensor = transformed['image'].unsqueeze(0)  # (1, 1, H, W)
        
        # Predict
        if use_tta:
            pred = self.predict_tta(img_tensor, num_augs=tta_augs)
        else:
            with torch.autocast('cuda'):
                logits = self.model(img_tensor)
                pred = torch.softmax(logits, dim=1)
        
        # Extract foreground probability
        prob = pred[0, 1, :, :].cpu().numpy()  # (H, W)
        
        # Binary prediction with threshold
        pred_mask = (prob > 0.5).astype(np.uint8)
        
        # Post-processing: morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
        
        return pred_mask, prob
    
    @torch.no_grad()
    def predict_batch(self, image_paths, use_tta=True, tta_augs=4, batch_size=4):
        """Predict on batch of images"""
        results = []
        
        for img_path in tqdm(image_paths, desc="Inference"):
            pred_mask, confidence = self.predict_single(img_path, use_tta, tta_augs)
            results.append({
                'path': img_path,
                'mask': pred_mask,
                'confidence': confidence
            })
        
        return results


# ======================================================
# EVALUATION METRICS
# ======================================================

def compute_metrics(pred_mask, true_mask):
    """Compute metrics for binary segmentation"""
    
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    
    tp = np.sum(pred & true)
    tn = np.sum(~pred & ~true)
    fp = np.sum(pred & ~true)
    fn = np.sum(~pred & true)
    
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_on_dataset(model_path, img_dir, mask_dir, device='cuda', use_tta=True):
    """Evaluate model on test dataset"""
    
    inference = OptimizedInference(model_path, device=device)
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    all_metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    print(f"Evaluating on {len(img_files)} images...")
    
    for img_file in tqdm(img_files):
        img_path = os.path.join(img_dir, img_file)
        mask_name = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            continue
        
        # Predict
        pred_mask, _ = inference.predict_single(img_path, use_tta=use_tta)
        
        # Load true mask
        true_mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
        true_mask = (true_mask > 127).astype(np.uint8)
        
        # Resize to same size if needed
        if pred_mask.shape != true_mask.shape:
            pred_mask = cv2.resize(pred_mask, (true_mask.shape[1], true_mask.shape[0]))
        
        # Compute metrics
        metrics = compute_metrics(pred_mask, true_mask)
        
        for k, v in metrics.items():
            all_metrics[k].append(v)
    
    # Compute averages
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    for metric_name in all_metrics:
        values = all_metrics[metric_name]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name.upper():12} | Mean: {mean_val:.6f} | Std: {std_val:.6f}")
    
    print("=" * 80 + "\n")
    
    return all_metrics


# ======================================================
# DEMO
# ======================================================

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--img_dir', type=str, help='Directory with images for inference')
    parser.add_argument('--mask_dir', type=str, help='Directory with ground truth masks for evaluation')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Output directory')
    parser.add_argument('--use_tta', action='store_true', default=True, help='Use TTA')
    parser.add_argument('--tta_augs', type=int, default=4, help='Number of TTA augmentations')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference
    inference = OptimizedInference(args.model_path, device=args.device)
    
    # Inference mode
    if args.img_dir:
        print("\n" + "=" * 80)
        print("RUNNING INFERENCE")
        print("=" * 80 + "\n")
        
        img_files = sorted([f for f in os.listdir(args.img_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_file in tqdm(img_files, desc="Processing"):
            img_path = os.path.join(args.img_dir, img_file)
            
            # Predict
            pred_mask, confidence = inference.predict_single(
                img_path,
                use_tta=args.use_tta,
                tta_augs=args.tta_augs
            )
            
            # Save prediction
            out_mask = os.path.join(args.output_dir, f"{os.path.splitext(img_file)[0]}_pred.png")
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(out_mask)
            
            # Save confidence map
            out_conf = os.path.join(args.output_dir, f"{os.path.splitext(img_file)[0]}_conf.png")
            Image.fromarray((confidence * 255).astype(np.uint8)).save(out_conf)
        
        print(f"\n✅ Predictions saved to: {args.output_dir}\n")
    
    # Evaluation mode
    if args.img_dir and args.mask_dir:
        evaluate_on_dataset(args.model_path, args.img_dir, args.mask_dir, 
                          device=args.device, use_tta=args.use_tta)


if __name__ == "__main__":
    main()
