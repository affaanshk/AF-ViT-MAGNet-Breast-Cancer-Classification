# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedClassificationLoss(nn.Module):
    """
    Combines Cross-Entropy classification loss with optional segmentation-guided regularization:
    L = L_cls + beta * L_seg[cite: 3]
    """
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets, seg_loss=None):
        cls_loss = self.ce(logits, targets)
        if seg_loss is not None:
            return cls_loss + self.beta * seg_loss
        return cls_loss