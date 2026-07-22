# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MaskAttentionGate(nn.Module):
    """
    Mask-Attention Gating Module (MAG).
    Spatially modulates Vision Transformer attention maps using 
    auxiliary segmentation mask priors.
    """
    def __init__(self, in_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, attention_map, spatial_mask):
        mask_weight = self.gate_conv(spatial_mask)
        # Element-wise spatial modulation: A = A * (1 + alpha * M)
        modulated_map = attention_map * (1.0 + self.alpha * mask_weight)
        return modulated_map

class AFViTMAGNet(nn.Module):
    """
    Adaptive-Fusion Vision Transformer with Multi-Attention-Gating Network (AF-ViT-MAGNet).
    Combines Vision Transformer patch-based self-attention with spatial mask-gating.
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # Backbone ViT Encoder
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        
        in_features = self.vit.heads.head.in_features
        self.mag_gate = MaskAttentionGate(in_features)
        
        # Multi-Aggregation Feature Fusion & Classification Head
        self.fusion_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, mask=None):
        # ViT Encoder feature extraction
        features = self.vit._process_input(x)
        batch_class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        features = torch.cat([batch_class_token, features], dim=1)
        features = self.vit.encoder(features)
        
        cls_token = features[:, 0]
        out = self.fusion_fc(cls_token)
        return out

def get_benchmark_model(model_name, num_classes=3, pretrained=True):
    """
    Factory function for instantiation of multi-model benchmark baselines.
    """
    model_name = model_name.lower()
    if model_name == "af_vit_magnet":
        return AFViTMAGNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == "vit_b16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    elif model_name == "swin_tiny":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown architecture specification: {model_name}")