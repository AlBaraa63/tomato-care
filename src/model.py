import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Channel Attention)
    
    Helps the network focus on important channels (disease features) 
    and ignore irrelevant ones (background noise).
    Low computational cost, high accuracy gain.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    """
    Residual Convolutional Block:
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+shortcut) -> ReLU -> MaxPool -> Dropout

    The 1x1 shortcut projection lets gradients flow directly through the
    network, improving training from scratch with near-zero param overhead.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.25):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + identity)
        out = self.pool(out)
        out = self.dropout(out)
        return out

class TomatoCareNet(nn.Module):
    """
    Custom Architecture built from scratch.
    optimized for Mobile (Global Average Pooling, SE Blocks).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 1. Feature Extractor (The "Eye")
        # Increases channels (complexity) while reducing spatial size (resolution)
        self.block1 = ConvBlock(3,   32,  dropout_rate=0.2)   # 224 -> 112
        self.block2 = ConvBlock(32,  64,  dropout_rate=0.25)  # 112 -> 56
        self.block3 = ConvBlock(64,  128, dropout_rate=0.3)   # 56 -> 28
        self.block4 = ConvBlock(128, 256, dropout_rate=0.3)   # 28 -> 14
        
        # 2. Attention Mechanism (The "Brain" focus)
        self.se_block = SEBlock(256, reduction=16)
        
        # 3. Global Pooling (The "Compressor" for Mobile)
        # Instead of flattening 14x14x256 (50k params), we average to 1x1x256
        # This massively reduces parameters, saving file size for mobile apps.
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 4. Classifier (The "Decision Maker")
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.se_block(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        Kaiming/He Initialization - Crucial for training from scratch
        Helps the model converge faster by setting smart random starting weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    # Estimate size in MB (float32 = 4 bytes)
    size_mb = params * 4 / (1024 ** 2)
    print(f"Estimated Model Size: {size_mb:.2f} MB")
    return params
