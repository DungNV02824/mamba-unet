"""
📊 PERFORMANCE OPTIMIZATION TUNING GUIDE

This file contains optimized configurations for different scenarios.
Copy and modify based on your needs.
"""

# ======================================================
# CONFIGURATION 1: FAST TRAINING (For quick experiments)
# ======================================================
CONFIG_FAST = {
    'batch_size': 8,
    'epochs': 80,
    'lr': 2e-4,
    'warmup_epochs': 5,
    'early_stop_patience': 30,
    'embed_dim': 24,
    'depths': [1, 1, 1, 1],
    'drop_path_rate': 0.1,
    'grad_accumulation_steps': 1,
    'use_mixup': True,
    'use_cutmix': False,
}


# ======================================================
# CONFIGURATION 2: BALANCED (Recommended - Good quality/speed tradeoff)
# ======================================================
CONFIG_BALANCED = {
    'batch_size': 12,
    'epochs': 150,
    'lr': 2e-4,
    'warmup_epochs': 10,
    'early_stop_patience': 35,
    'embed_dim': 32,
    'depths': [2, 2, 2, 1],
    'drop_path_rate': 0.2,
    'grad_accumulation_steps': 2,
    'use_mixup': True,
    'use_cutmix': True,
}


# ======================================================
# CONFIGURATION 3: HIGH QUALITY (For best results)
# ======================================================
CONFIG_QUALITY = {
    'batch_size': 16,
    'epochs': 200,
    'lr': 1.5e-4,
    'warmup_epochs': 15,
    'early_stop_patience': 40,
    'embed_dim': 48,
    'depths': [2, 2, 3, 2],
    'drop_path_rate': 0.3,
    'grad_accumulation_steps': 4,
    'use_mixup': True,
    'use_cutmix': True,
}


# ======================================================
# CONFIGURATION 4: MEMORY CONSTRAINED (For low-memory GPU)
# ======================================================
CONFIG_LOW_MEMORY = {
    'batch_size': 6,
    'epochs': 120,
    'lr': 2e-4,
    'warmup_epochs': 8,
    'early_stop_patience': 25,
    'embed_dim': 24,
    'depths': [2, 1, 1, 1],
    'drop_path_rate': 0.15,
    'grad_accumulation_steps': 4,
    'use_mixup': True,
    'use_cutmix': False,
}


# ======================================================
# INFERENCE CONFIGURATIONS
# ======================================================

INFERENCE_FAST = {
    'use_tta': False,  # No augmentation
    'batch_size': 8,
}

INFERENCE_BALANCED = {
    'use_tta': True,   # 2x augmentation (hflip, vflip)
    'tta_augs': 2,
    'batch_size': 4,
}

INFERENCE_HIGH_QUALITY = {
    'use_tta': True,   # 4x augmentation (original, hflip, vflip, both)
    'tta_augs': 4,
    'batch_size': 2,
}


# ======================================================
# COMMAND EXAMPLES
# ======================================================

"""
1. FAST TRAINING:
   python train_optimized.py \\
     --batch_size 8 --epochs 80 --early_stop_patience 30 \\
     --embed_dim 24 --depths 1 1 1 1 --drop_path_rate 0.1

2. BALANCED TRAINING (RECOMMENDED):
   python train_optimized.py \\
     --batch_size 12 --epochs 150 --early_stop_patience 35 \\
     --grad_accumulation_steps 2 --use_mixup --use_cutmix

3. HIGH QUALITY TRAINING:
   python train_optimized.py \\
     --batch_size 16 --epochs 200 --early_stop_patience 40 \\
     --embed_dim 48 --depths 2 2 3 2 --drop_path_rate 0.3 \\
     --grad_accumulation_steps 4

4. INFERENCE WITH TTA:
   python inference_optimized.py \\
     --model_path checkpoints/best.pth \\
     --img_dir meger_Data_DC1000/img \\
     --mask_dir meger_Data_DC1000/masks_machine \\
     --use_tta --tta_augs 4

5. FAST INFERENCE:
   python inference_optimized.py \\
     --model_path checkpoints/best.pth \\
     --img_dir meger_Data_DC1000/img \\
     --output_dir predictions
"""


# ======================================================
# OPTIMIZATION TIPS
# ======================================================

OPTIMIZATION_TIPS = """
🔧 OPTIMIZATION TIPS FOR SUPER RESULTS:

1. DATA QUALITY:
   ✓ Ensure masks are correctly binary (0 or 255)
   ✓ Check image contrast - use histogram equalization if needed
   ✓ Remove mislabeled samples
   ✓ Consider data cleaning pipeline

2. HYPERPARAMETER TUNING:
   ✓ Start with BALANCED config
   ✓ If overfitting: increase drop_path_rate (0.3-0.4)
   ✓ If underfitting: increase model capacity (embed_dim, depths)
   ✓ If memory issues: reduce batch_size, increase grad_accumulation_steps
   ✓ Learning rate: Usually 1e-4 to 2e-4 works best

3. AUGMENTATION:
   ✓ Mixup helps with small datasets
   ✓ CutMix helps with boundary learning
   ✓ Elastic transform and rotation help with broken teeth

4. LOSS FUNCTION:
   ✓ CombinedLoss_Improved (CE + Dice + Focal) recommended
   ✓ Dice weight > CE weight for segmentation tasks
   ✓ Focal weight helps with hard examples

5. TRAINING STRATEGY:
   ✓ Use warmup first 10-15 epochs
   ✓ Cosine annealing for learning rate decay
   ✓ Early stopping with patience 30-40
   ✓ Save checkpoint every 10 epochs

6. INFERENCE:
   ✓ Use TTA (Test-Time Augmentation) for better results
   ✓ Apply morphological operations (open/close) for cleanup
   ✓ Post-process with confidence thresholding

7. MONITORING:
   ✓ Track both Dice and IoU (segmentation metrics)
   ✓ Monitor Precision, Recall, F1 for imbalanced data
   ✓ Check validation loss doesn't diverge
   ✓ Plot training curves to detect overfitting

8. ENSEMBLE (For even better results):
   ✓ Train multiple models with different seeds
   ✓ Average predictions
   ✓ Typically 2-3 models give best results
"""

if __name__ == "__main__":
    print(OPTIMIZATION_TIPS)
