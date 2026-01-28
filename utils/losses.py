import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        probs_fg = probs[:, 1, :, :]
        targets_fg = (targets == 1).float()

        dims = (1, 2)

        intersection = (probs_fg * targets_fg).sum(dims)
        union = probs_fg.sum(dims) + targets_fg.sum(dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice = dice.mean()

        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=0.3, weight_dice=0.7):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
    
    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.weight_ce * ce + self.weight_dice * dice
