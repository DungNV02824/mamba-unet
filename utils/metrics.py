"""
Evaluation metrics
"""
import torch

def dice_coefficient(pred, target, smooth=1e-5):
    pred = torch.softmax(pred, dim=1)[:, 1, :, :]
    pred = (pred > 0.5).float()
    target = (target == 1).float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def iou_score(pred, target, smooth=1e-5):
    pred = torch.softmax(pred, dim=1)[:, 1, :, :]
    pred = (pred > 0.5).float()
    target = (target == 1).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()