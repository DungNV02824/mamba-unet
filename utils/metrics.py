"""
OPTIMIZED Evaluation Metrics for Tooth Segmentation

Key improvements:
- Per-sample computation (accurate for batch_size > 1)
- Additional metrics: Precision, Recall, F1, Specificity
- Confusion matrix stats
- Boundary metrics (optional)
"""
import torch
import torch.nn.functional as F



# BASIC METRICS 


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-5):
    """
    Dice Coefficient (F1 Score for binary segmentation)
    
    FIXED: Compute per-sample then average (more accurate)
    
    Args:
        pred: (B, num_classes, H, W) - logits
        target: (B, H, W) - ground truth labels
        threshold: float - probability threshold
        smooth: float - smoothing constant
    
    Returns:
        float - mean dice score
    """
    # Get probability for foreground class (teeth)
    probs = torch.softmax(pred, dim=1)[:, 1]  # (B, H, W)
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()
    
    # Compute per-sample (IMPORTANT!)
    batch_size = pred.shape[0]
    dice_scores = []
    
    for i in range(batch_size):
        pred_i = pred_binary[i]
        target_i = target_binary[i]
        
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()
        
        dice_i = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice_i.item())
    
    return sum(dice_scores) / len(dice_scores)


def iou_score(pred, target, threshold=0.5, smooth=1e-5):
    """
    IoU (Jaccard Index)
    
    FIXED: Compute per-sample then average
    
    Args:
        pred: (B, num_classes, H, W)
        target: (B, H, W)
        threshold: float
        smooth: float
    
    Returns:
        float - mean IoU score
    """
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()
    
    batch_size = pred.shape[0]
    iou_scores = []
    
    for i in range(batch_size):
        pred_i = pred_binary[i]
        target_i = target_binary[i]
        
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum() - intersection
        
        iou_i = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou_i.item())
    
    return sum(iou_scores) / len(iou_scores)



# ADDITIONAL METRICS


def precision_score(pred, target, threshold=0.5, smooth=1e-5):
    """
    Precision = TP / (TP + FP)
    Measure: Trong những pixel dự đoán là răng, bao nhiêu % đúng?
    """
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()
    
    batch_size = pred.shape[0]
    precision_scores = []
    
    for i in range(batch_size):
        TP = (pred_binary[i] * target_binary[i]).sum()
        FP = (pred_binary[i] * (1 - target_binary[i])).sum()
        
        precision_i = (TP + smooth) / (TP + FP + smooth)
        precision_scores.append(precision_i.item())
    
    return sum(precision_scores) / len(precision_scores)


def recall_score(pred, target, threshold=0.5, smooth=1e-5):
    """
    Recall (Sensitivity) = TP / (TP + FN)
    Measure: Trong tất cả răng thật, bao nhiêu % được phát hiện?
    """
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()
    
    batch_size = pred.shape[0]
    recall_scores = []
    
    for i in range(batch_size):
        TP = (pred_binary[i] * target_binary[i]).sum()
        FN = ((1 - pred_binary[i]) * target_binary[i]).sum()
        
        recall_i = (TP + smooth) / (TP + FN + smooth)
        recall_scores.append(recall_i.item())
    
    return sum(recall_scores) / len(recall_scores)


def specificity_score(pred, target, threshold=0.5, smooth=1e-5):
    """
    Specificity = TN / (TN + FP)
    Measure: Trong tất cả background, bao nhiêu % được phát hiện đúng?
    """
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()
    
    batch_size = pred.shape[0]
    specificity_scores = []
    
    for i in range(batch_size):
        TN = ((1 - pred_binary[i]) * (1 - target_binary[i])).sum()
        FP = (pred_binary[i] * (1 - target_binary[i])).sum()
        
        specificity_i = (TN + smooth) / (TN + FP + smooth)
        specificity_scores.append(specificity_i.item())
    
    return sum(specificity_scores) / len(specificity_scores)


def f1_score(pred, target, threshold=0.5, smooth=1e-5):
    """
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    Same as Dice coefficient
    """
    precision = precision_score(pred, target, threshold, smooth)
    recall = recall_score(pred, target, threshold, smooth)
    
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    return f1


def accuracy(pred, target, threshold=0.5):
    """
    Pixel Accuracy = (TP + TN) / (TP + TN + FP + FN)
    NOTE: Không phù hợp cho imbalanced data
    """
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()
    
    correct = (pred_binary == target_binary).float().sum()
    total = target_binary.numel()
    
    return (correct / total).item()



# CONFUSION MATRIX


def confusion_matrix_stats(pred, target, threshold=0.5):
    """
    Compute TP, TN, FP, FN
    
    Returns:
        dict with keys: TP, TN, FP, FN (all as counts)
    """
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()
    
    TP = (pred_binary * target_binary).sum().item()
    TN = ((1 - pred_binary) * (1 - target_binary)).sum().item()
    FP = (pred_binary * (1 - target_binary)).sum().item()
    FN = ((1 - pred_binary) * target_binary).sum().item()
    
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Total': TP + TN + FP + FN
    }



# ALL METRICS AT ONCE


def compute_all_metrics(pred, target, threshold=0.5, smooth=1e-5):
    """
    Compute all metrics at once (efficient)
    
    Returns:
        dict with all metric scores
    """
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()
    
    batch_size = pred.shape[0]
    
    metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'specificity': [],
        'f1': []
    }
    
    for i in range(batch_size):
        pred_i = pred_binary[i]
        target_i = target_binary[i]
        
        # Confusion matrix elements
        TP = (pred_i * target_i).sum()
        TN = ((1 - pred_i) * (1 - target_i)).sum()
        FP = (pred_i * (1 - target_i)).sum()
        FN = ((1 - pred_i) * target_i).sum()
        
        # Dice & IoU
        intersection = TP
        union = pred_i.sum() + target_i.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        iou = (intersection + smooth) / (union - intersection + smooth)
        
        # Precision, Recall, Specificity
        precision = (TP + smooth) / (TP + FP + smooth)
        recall = (TP + smooth) / (TP + FN + smooth)
        specificity = (TN + smooth) / (TN + FP + smooth)
        
        # F1
        f1 = (2 * precision * recall) / (precision + recall + smooth)
        
        metrics['dice'].append(dice.item())
        metrics['iou'].append(iou.item())
        metrics['precision'].append(precision.item())
        metrics['recall'].append(recall.item())
        metrics['specificity'].append(specificity.item())
        metrics['f1'].append(f1.item())
    
    # Average across batch
    return {k: sum(v) / len(v) for k, v in metrics.items()}



# BOUNDARY METRICS (Advanced - Optional)

def boundary_iou(pred, target, threshold=0.5, dilation=2):
    """
    Boundary IoU - Focus on tooth edges
    Useful for medical segmentation
    
    Args:
        dilation: int - boundary thickness in pixels
    """
    import torch.nn.functional as F
    
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float().unsqueeze(1)  # (B, 1, H, W)
    target_binary = (target == 1).float().unsqueeze(1)
    
    # Compute boundaries using dilation-erosion
    kernel = torch.ones(1, 1, dilation*2+1, dilation*2+1).to(pred.device)
    
    pred_dilated = F.conv2d(pred_binary, kernel, padding=dilation) > 0
    pred_eroded = F.conv2d(pred_binary, kernel, padding=dilation) >= kernel.sum()
    pred_boundary = (pred_dilated.float() - pred_eroded.float()).squeeze(1)
    
    target_dilated = F.conv2d(target_binary, kernel, padding=dilation) > 0
    target_eroded = F.conv2d(target_binary, kernel, padding=dilation) >= kernel.sum()
    target_boundary = (target_dilated.float() - target_eroded.float()).squeeze(1)
    
    # IoU on boundaries
    intersection = (pred_boundary * target_boundary).sum()
    union = pred_boundary.sum() + target_boundary.sum() - intersection
    
    boundary_iou = (intersection + 1e-5) / (union + 1e-5)
    
    return boundary_iou.item()


# UTILITY FUNCTIONS


def find_optimal_threshold(pred, target, metric='dice', search_range=(0.3, 0.7), steps=20):
    """
    Find optimal threshold for a given metric
    
    Args:
        metric: 'dice', 'iou', 'f1', 'precision', 'recall'
    
    Returns:
        best_threshold, best_score
    """
    thresholds = torch.linspace(search_range[0], search_range[1], steps)
    
    metric_funcs = {
        'dice': dice_coefficient,
        'iou': iou_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }
    
    metric_func = metric_funcs[metric]
    
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        score = metric_func(pred, target, threshold=thresh.item())
        if score > best_score:
            best_score = score
            best_threshold = thresh.item()
    
    return best_threshold, best_score

