"""
classifier.py - TomatoCare CNN Architecture

MobileNetV2 transfer learning for tomato disease classification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from configs.config import NUM_CLASSES, PRETRAINED, DEVICE, get_device_info


class TomatoClassifier(nn.Module):
    """
    MobileNetV2-based classifier for tomato disease detection.
    
    Architecture:
    - Backbone: MobileNetV2 (pretrained on ImageNet)
    - Head: Dropout → FC(1280 → num_classes)
    """
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=PRETRAINED, freeze_backbone=True):
        super(TomatoClassifier, self).__init__()
        
        # Load pretrained MobileNetV2
        if pretrained:
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Get the input features of the classifier
        in_features = self.backbone.classifier[1].in_features  # 1280
        
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        
        # Only freeze backbone if using pretrained weights
        if freeze_backbone and pretrained:
            self.freeze_backbone()
        elif not pretrained:
            print("🔓 Training from scratch - all layers trainable")
    
    def freeze_backbone(self):
        """Freeze all backbone layers (features), only train classifier."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen - only training classifier head")
    
    def unfreeze_backbone(self, unfreeze_from=14):
        """
        Unfreeze backbone layers for fine-tuning.
        MobileNetV2 has 19 inverted residual blocks (0-18).
        
        Args:
            unfreeze_from: Unfreeze from this index onwards (default 14 = last 5)
        """
        for i, layer in enumerate(self.backbone.features):
            if i >= unfreeze_from:
                for param in layer.parameters():
                    param.requires_grad = True
        
        unfrozen = len(self.backbone.features) - unfreeze_from
        print(f"🔓 Unfroze last {unfrozen} backbone blocks for fine-tuning")
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_trainable_params(self):
        """Count trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable
        return {"trainable": trainable, "frozen": frozen, "total": total}


def get_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED, freeze_backbone=True):
    """Factory function to create the model."""
    return TomatoClassifier(num_classes, pretrained, freeze_backbone)


def model_summary(model, input_size=(1, 3, 224, 224)):
    """Print model summary with parameter counts."""
    print("\n" + "=" * 60)
    print("📊 TOMATOCARE MODEL SUMMARY")
    print("=" * 60)
    print(f"Architecture: MobileNetV2 + Custom Classifier")
    print(f"Input size:   {input_size[2]}x{input_size[3]} RGB")
    print(f"Output:       {NUM_CLASSES} classes")
    print("-" * 60)
    
    params = model.get_trainable_params()
    print(f"Total parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters:    {params['frozen']:,}")
    print("-" * 60)
    
    # Estimate model size
    size_mb = params['total'] * 4 / (1024 * 1024)
    print(f"Estimated size: {size_mb:.2f} MB (FP32)")
    print(f"Quantized:      ~{size_mb/4:.2f} MB (INT8)")
    print("=" * 60)
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Forward pass: {list(input_size)} → {list(output.shape)}")
    print("=" * 60 + "\n")


# === MAIN (for testing) ===
if __name__ == "__main__":
    device = get_device_info()
    
    # Create model
    model = get_model(freeze_backbone=True)
    model = model.to(device)
    model_summary(model)
    
    # Test unfreezing
    print("🔄 Testing backbone unfreezing...")
    model.unfreeze_backbone(unfreeze_from=14)
    model_summary(model)
