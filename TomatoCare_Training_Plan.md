w# 🍅 TomatoCare — Full Training Plan (PyTorch + RTX 4070 Ti Super)

---

## Quick Answer: Do We Need Augmentation & Class Weights?

**YES to both.** Here's why:

| Technique | Do We Need It? | Why? |
|-----------|---------------|------|
| **Augmentation** | ✅ YES | Our dataset has ~32K images but many are lab photos (PlantVillage). Augmentation simulates real-world variation: different angles, lighting, zoom levels. Without it, the model memorizes lab conditions and fails on a real garden photo. |
| **Class Weights** | ✅ YES | Spider Mites has 1,773 training images vs Healthy has 2,992. That's a 1.7x gap. Without weights, the model gets lazy — it learns to predict the bigger classes more often because it "sees" them more. Class weights tell the model: "Pay MORE attention to the smaller classes." |

---

## Part 1: Environment Setup (Your Local Machine)

### 1.1 Install PyTorch with CUDA (for your RTX 4070 Ti Super)

Open your terminal and run:

```bash
# Create a conda environment (recommended)
conda create -n tomatocare python=3.11
conda activate tomatocare

# Install PyTorch with CUDA 12.4 (best for 4070 Ti Super)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install albumentations matplotlib seaborn scikit-learn pandas tqdm tensorboard pillow

# Verify GPU is detected
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

You should see:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
```

### 1.2 Install Claude Code

Claude Code is a command-line tool that lets Claude help you write code directly in your terminal/IDE.

```bash
# Install Claude Code (requires Node.js 18+)
npm install -g @anthropic-ai/claude-code

# Navigate to your project
cd path/to/TomatoCare

# Start Claude Code
claude
```

For the most current installation instructions, see: https://docs.claude.com/en/docs/claude-code/overview

### 1.3 Using Claude Code for This Project

Once inside your TomatoCare folder, you can give Claude Code instructions like:

```
> Create the TomatoCareNet model in src/model.py following the architecture 
  in docs/TomatoCare_Research.md. Use PyTorch, 4 conv blocks with 
  BatchNorm, SE attention, and GlobalAveragePooling. 10 output classes.

> Write the training loop in src/train.py with: focal loss, class weights 
  calculated from the training set, cosine annealing LR scheduler, 
  mixed precision training, and TensorBoard logging.

> Create a data loader in src/data_loader.py that loads from 
  data/processed/{train,val,test}/ with Albumentations augmentation 
  for training only.
```

**Tips for effective Claude Code prompts:**
- Reference specific files: "Edit `src/model.py`"
- Be specific about what you want: architecture details, loss function, etc.
- Ask it to explain what it writes: "Add comments explaining each block"
- Build incrementally: one file at a time, test, then move on

---

## Part 2: The Training Pipeline (Step by Step)

Here's every file we need to create and what it does:

```
src/
├── config.py          ← All settings in one place (easy to change)
├── data_loader.py     ← Load images + augmentation
├── model.py           ← TomatoCareNet architecture
├── train.py           ← Training loop
├── evaluate.py        ← Testing + metrics
└── gradcam.py         ← Explainability (later)
```

---

### Step 1: `src/config.py` — Central Configuration

**What this does:** Keeps ALL settings in one place. When you want to change 
the learning rate or batch size, you change it here — not scattered across 
10 files.

```python
import torch
from pathlib import Path

class Config:
    # ── Paths ──
    PROJECT_ROOT = Path(__file__).parent.parent        # TomatoCare/
    DATA_DIR     = PROJECT_ROOT / "data" / "processed"
    TRAIN_DIR    = DATA_DIR / "train"
    VAL_DIR      = DATA_DIR / "val"
    TEST_DIR     = DATA_DIR / "test"
    CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
    RESULTS_DIR    = PROJECT_ROOT / "results"
    
    # ── Classes ──
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "Bacterial_Spot", "Early_Blight", "Healthy", "Late_Blight",
        "Leaf_Mold", "Septoria_Leaf_Spot", "Spider_Mites",
        "Target_Spot", "Yellow_Leaf_Curl_Virus", "Mosaic_Virus"
    ]
    
    # ── Image Settings ──
    IMG_SIZE = 224           # All images resized to 224x224
    IN_CHANNELS = 3          # RGB
    
    # ── Training Hyperparameters ──
    BATCH_SIZE = 32          # 32 fits well on 4070 Ti Super (16GB VRAM)
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3     # Starting LR (will decay with scheduler)
    WEIGHT_DECAY = 1e-4      # L2 regularization (prevents overfitting)
    
    # ── Scheduler ──
    WARMUP_EPOCHS = 5        # Gradually increase LR for first 5 epochs
    MIN_LR = 1e-6            # Lowest LR the scheduler will go to
    
    # ── Early Stopping ──
    PATIENCE = 15            # Stop if val loss doesn't improve for 15 epochs
    
    # ── Focal Loss ──
    FOCAL_GAMMA = 2.0        # Focus more on hard-to-classify samples
    
    # ── Device ──
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ── Mixed Precision ──
    USE_AMP = True           # Automatic Mixed Precision (faster on RTX cards)
    
    # ── DataLoader ──
    NUM_WORKERS = 4          # Parallel data loading threads
    PIN_MEMORY = True        # Faster GPU transfer
    
    # ── Reproducibility ──
    SEED = 42
```

**Why each setting matters:**

| Setting | Value | Explanation |
|---------|-------|-------------|
| `BATCH_SIZE = 32` | With 16GB VRAM and 224×224 images, 32 is the sweet spot. Larger = faster but uses more memory. |
| `LEARNING_RATE = 1e-3` | Starting learning rate for Adam optimizer. Too high = model overshoots. Too low = trains forever. |
| `WEIGHT_DECAY = 1e-4` | Adds a small penalty for large weights → prevents overfitting. |
| `WARMUP_EPOCHS = 5` | First 5 epochs use a gradually increasing LR so training starts gently. |
| `PATIENCE = 15` | If the model doesn't improve for 15 epochs, stop early (it's probably done learning). |
| `FOCAL_GAMMA = 2.0` | Makes the loss function focus harder on samples the model gets wrong. Great for imbalanced classes. |
| `USE_AMP = True` | Mixed precision: uses float16 where possible → 2x faster training on your RTX 4070 Ti Super. |

---

### Step 2: `src/data_loader.py` — Loading Images + Augmentation

**What this does:** 
1. Reads images from the `train/`, `val/`, `test/` folders
2. Resizes them all to 224×224
3. Applies augmentation (ONLY on training images)
4. Calculates class weights (for imbalance handling)
5. Returns PyTorch DataLoaders (batches of images ready for the GPU)

```python
import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import Counter
from config import Config


# ── Augmentation Pipelines ──

def get_train_transforms():
    """
    Training augmentation: makes each image look slightly different 
    every time the model sees it. This forces the model to learn 
    GENERAL patterns (disease features) not SPECIFIC images.
    """
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        
        # Geometric: simulate different camera angles
        A.HorizontalFlip(p=0.5),              # 50% chance flip left-right
        A.VerticalFlip(p=0.3),                # 30% chance flip up-down
        A.RandomRotate90(p=0.3),              # 30% chance rotate 90°
        A.ShiftScaleRotate(                   # Random shift + zoom + rotate
            shift_limit=0.1,                  # Move image up to 10%
            scale_limit=0.15,                 # Zoom in/out up to 15%
            rotate_limit=30,                  # Rotate up to 30°
            p=0.5
        ),
        
        # Color: simulate different lighting conditions
        A.RandomBrightnessContrast(           # Brighter/darker + more/less contrast
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(                 # Slight color shifts
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        
        # Noise: simulate camera sensor noise
        A.GaussNoise(p=0.2),                  # 20% chance add random noise
        
        # Occlusion: simulate partially hidden leaves
        A.CoarseDropout(                      # Random rectangular holes
            num_holes_range=(1, 3),           # 1-3 holes
            hole_height_range=(20, 40),       # 20-40 px height
            hole_width_range=(20, 40),        # 20-40 px width
            fill="random",
            p=0.3
        ),
        
        # Normalize + convert to PyTorch tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],       # ImageNet standard values
            std=[0.229, 0.224, 0.225]          # (works well even without pretraining)
        ),
        ToTensorV2()
    ])


def get_val_transforms():
    """
    Validation/Test: NO augmentation. Just resize + normalize.
    We want to test on clean, unmodified images.
    """
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


# ── Custom Dataset Class ──

class TomatoDataset(Dataset):
    """
    A PyTorch Dataset that:
    1. Scans a folder for images organized in class subfolders
    2. Loads images on demand (not all at once — saves RAM)
    3. Applies the appropriate transforms
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []      # List of (image_path, class_index) tuples
        self.class_to_idx = {} # Maps class name → number
        
        # Scan folders and build sample list
        classes = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(classes):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((
                        os.path.join(class_path, img_name),
                        idx
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image as numpy array (RGB)
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        return image, label


# ── Class Weights Calculation ──

def compute_class_weights(dataset):
    """
    Calculate weights inversely proportional to class frequency.
    
    Example:
      - Healthy has 2,992 images (common) → gets LOWER weight
      - Spider Mites has 1,773 images (rare) → gets HIGHER weight
    
    This makes the model pay equal attention to all classes.
    """
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    total = len(labels)
    num_classes = len(class_counts)
    
    weights = []
    for i in range(num_classes):
        # Formula: total / (num_classes × count_of_this_class)
        w = total / (num_classes * class_counts[i])
        weights.append(w)
    
    return torch.FloatTensor(weights).to(Config.DEVICE)


# ── DataLoader Factory ──

def get_dataloaders():
    """
    Creates train, val, and test DataLoaders.
    Returns them along with class weights.
    """
    train_dataset = TomatoDataset(Config.TRAIN_DIR, transform=get_train_transforms())
    val_dataset   = TomatoDataset(Config.VAL_DIR,   transform=get_val_transforms())
    test_dataset  = TomatoDataset(Config.TEST_DIR,  transform=get_val_transforms())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,             # Shuffle training data every epoch
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True            # Drop incomplete last batch (stabilizes BatchNorm)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,            # Never shuffle validation
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    class_weights = compute_class_weights(train_dataset)
    
    print(f"Training samples:   {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Test samples:       {len(test_dataset):,}")
    print(f"Class weights: {class_weights}")
    
    return train_loader, val_loader, test_loader, class_weights
```

---

### Step 3: `src/model.py` — TomatoCareNet Architecture

**What this does:** Defines our custom CNN. Four convolutional blocks that 
progressively extract more complex features, an SE attention module, and 
a classifier head.

```python
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Channel Attention)
    
    Think of it like this:
    - Your CNN produces 256 feature maps (channels)
    - Some channels detect useful stuff (disease spots)
    - Some channels detect useless stuff (background texture)
    - SE Block learns which channels are important and amplifies them
    
    How it works:
    1. SQUEEZE: Average each channel into a single number
    2. EXCITE: Pass through 2 small Dense layers to learn importance
    3. SCALE: Multiply original channels by their importance scores
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)                    # Global average
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), # Compress
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False), # Expand back
            nn.Sigmoid()                                            # 0-1 importance scores
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)        # [B, C, H, W] → [B, C]
        y = self.excitation(y).view(b, c, 1, 1)  # [B, C] → [B, C, 1, 1]
        return x * y                            # Scale channels by importance


class ConvBlock(nn.Module):
    """
    A building block: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool → Dropout
    
    We use TWO conv layers per block because:
    - Two 3×3 convs = same receptive field as one 5×5 conv
    - But with fewer parameters and more non-linearity (better learning)
    """
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves spatial dimensions
            nn.Dropout2d(dropout_rate)               # Randomly zero out channels
        )
    
    def forward(self, x):
        return self.block(x)


class TomatoCareNet(nn.Module):
    """
    Custom CNN for Tomato Leaf Disease Classification
    
    Architecture:
        Input (224×224×3)
            ↓
        Block 1: 32 filters  → 112×112×32   (edges, textures)
        Block 2: 64 filters  → 56×56×64     (spots, color patterns)
        Block 3: 128 filters → 28×28×128    (disease shapes)
        Block 4: 256 filters → 14×14×256    (disease signatures)
            ↓
        SE Attention (which channels matter most?)
            ↓
        Global Average Pooling → 256
            ↓
        Dense(512) → Dense(256) → Dense(10)
            ↓
        Output: 10 disease probabilities
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extraction blocks
        self.block1 = ConvBlock(3,   32,  dropout_rate=0.2)   # 224 → 112
        self.block2 = ConvBlock(32,  64,  dropout_rate=0.25)  # 112 → 56
        self.block3 = ConvBlock(64,  128, dropout_rate=0.3)   # 56  → 28
        self.block4 = ConvBlock(128, 256, dropout_rate=0.3)   # 28  → 14
        
        # Attention
        self.se_block = SEBlock(256, reduction=16)
        
        # Global Average Pooling (replaces Flatten — much fewer parameters)
        self.gap = nn.AdaptiveAvgPool2d(1)  # 14×14×256 → 1×1×256
        
        # Classifier head
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
            
            nn.Linear(256, num_classes)  # Raw scores (logits), NOT softmax
            # Note: We don't add Softmax here because PyTorch's
            # CrossEntropyLoss already includes it internally
        )
        
        # Initialize weights (He initialization — best for ReLU networks)
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
        """He initialization for Conv layers, constant for BatchNorm"""
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
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size (approx):  {total * 4 / 1024 / 1024:.1f} MB (float32)")
    return total, trainable
```

---

### Step 4: `src/train.py` — Training Loop

**What this does:** The main training engine. For each epoch:
1. Feed batches of training images through the model
2. Calculate loss (how wrong the predictions are)
3. Backpropagate to update weights
4. Check performance on validation set
5. Save the best model
6. Stop early if no improvement

```python
import torch
import torch.nn as nn
import time
import copy
from pathlib import Path
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss — a smarter version of CrossEntropyLoss
    
    Problem: With imbalanced classes, the model gets lots of "easy" samples 
    right (common diseases) and ignores "hard" samples (rare diseases).
    
    Solution: Focal Loss DOWN-WEIGHTS easy samples and UP-WEIGHTS hard ones.
    
    Formula: FL = -alpha * (1 - p)^gamma * log(p)
    
    - gamma=0: Same as regular CrossEntropy
    - gamma=2: Hard samples get 100x more attention than easy ones
    """
    
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    
    Like a student studying for an exam: if your practice test scores 
    haven't improved in 15 attempts, you're probably not going to get 
    better — time to stop and use the best version.
    """
    
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch, return average loss and accuracy"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass (faster on RTX cards)
        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate on validation set, return average loss and accuracy"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, class_weights):
    """
    Full training pipeline.
    
    Returns the trained model and training history.
    """
    device = Config.DEVICE
    model = model.to(device)
    
    # ── Loss Function (Focal Loss with class weights) ──
    criterion = FocalLoss(weight=class_weights, gamma=Config.FOCAL_GAMMA)
    
    # ── Optimizer (AdamW = Adam + proper weight decay) ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # ── Learning Rate Scheduler (Cosine Annealing with Warmup) ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,           # Restart every 10 epochs
        T_mult=2,         # Double the period after each restart
        eta_min=Config.MIN_LR
    )
    
    # ── Mixed Precision Scaler ──
    scaler = GradScaler(enabled=Config.USE_AMP)
    
    # ── Early Stopping ──
    early_stopping = EarlyStopping(patience=Config.PATIENCE)
    
    # ── TensorBoard Logger ──
    writer = SummaryWriter(Config.RESULTS_DIR / "tensorboard")
    
    # ── Training History ──
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\n{'='*60}")
    print(f"  Training TomatoCareNet on {device}")
    print(f"  Epochs: {Config.NUM_EPOCHS} | Batch: {Config.BATCH_SIZE} | LR: {Config.LEARNING_RATE}")
    print(f"  AMP: {Config.USE_AMP} | Focal γ: {Config.FOCAL_GAMMA}")
    print(f"{'='*60}\n")
    
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # TensorBoard logging
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)
        
        elapsed = time.time() - start_time
        
        # Save best model
        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, Config.CHECKPOINT_DIR / "best_model.pth")
            improved = " ★ BEST"
        
        # Print progress
        print(
            f"Epoch [{epoch+1:3d}/{Config.NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s{improved}"
        )
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break
    
    writer.close()
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\n✓ Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history
```

---

### Step 5: `src/evaluate.py` — Testing & Metrics

**What this does:** After training, we test the model on the held-out test 
set and generate comprehensive metrics.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from torch.amp import autocast
from config import Config


@torch.no_grad()
def evaluate_model(model, test_loader, device=None):
    """
    Run model on test set and collect all predictions.
    Returns true labels, predicted labels, and raw probabilities.
    """
    if device is None:
        device = Config.DEVICE
    
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(images)
        
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_classification_report(y_true, y_pred, class_names=None):
    """Print detailed per-class metrics"""
    if class_names is None:
        class_names = Config.CLASS_NAMES
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "="*70)
    print("  CLASSIFICATION REPORT")
    print("="*70)
    print(report)
    return report


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot and optionally save confusion matrix"""
    if class_names is None:
        class_names = Config.CLASS_NAMES
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix — TomatoCareNet', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot training curves (loss and accuracy)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1].set_title('Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(history['lr'], color='green', linewidth=2)
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('TomatoCareNet Training History', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

---

### Step 6: `main.py` — Put It All Together

```python
"""
TomatoCare — Main Training Script
Run: python main.py
"""
import torch
import random
import numpy as np
from src.config import Config
from src.data_loader import get_dataloaders
from src.model import TomatoCareNet, count_parameters
from src.train import train_model
from src.evaluate import (
    evaluate_model, print_classification_report,
    plot_confusion_matrix, plot_training_history
)


def set_seed(seed=42):
    """Ensure reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 1. Set seed for reproducibility
    set_seed(Config.SEED)
    
    # 2. Create output directories
    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (Config.RESULTS_DIR / "plots").mkdir(exist_ok=True)
    (Config.RESULTS_DIR / "metrics").mkdir(exist_ok=True)
    
    # 3. Load data
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()
    
    # 4. Create model
    print("\nBuilding TomatoCareNet...")
    model = TomatoCareNet(num_classes=Config.NUM_CLASSES)
    count_parameters(model)
    
    # 5. Train
    model, history = train_model(model, train_loader, val_loader, class_weights)
    
    # 6. Plot training history
    plot_training_history(
        history,
        save_path=Config.RESULTS_DIR / "plots" / "training_history.png"
    )
    
    # 7. Evaluate on test set
    print("\nEvaluating on test set...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader)
    
    # 8. Print metrics
    print_classification_report(y_true, y_pred)
    
    # 9. Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=Config.RESULTS_DIR / "plots" / "confusion_matrix.png"
    )
    
    # 10. Save final model
    torch.save(model.state_dict(), Config.PROJECT_ROOT / "models" / "final" / "tomatocare_final.pth")
    print("\n✓ Model saved to models/final/tomatocare_final.pth")


if __name__ == "__main__":
    main()
```

---

## Part 3: Claude Code Workflow

Here's the exact step-by-step workflow to build this in your IDE:

### Session 1: Setup & Data Loading

```
You:   cd TomatoCare && claude

Claude Code Prompts:
> Create src/config.py with the configuration class from the training plan.
  Use Path objects for all directories. Set IMG_SIZE=224, BATCH_SIZE=32,
  NUM_EPOCHS=100, LEARNING_RATE=1e-3. Device should auto-detect CUDA.

> Create src/data_loader.py with: TomatoDataset class using Albumentations,
  get_train_transforms() with geometric + color + noise + cutout augmentation,
  get_val_transforms() with only resize + normalize,
  compute_class_weights() function, and get_dataloaders() factory.

> Test: run a quick script that loads 1 batch from train and prints
  the shape and labels to verify everything works.
```

### Session 2: Model Architecture

```
> Create src/model.py with TomatoCareNet: 4 ConvBlocks (32→64→128→256),
  SE attention block after block 4, GlobalAveragePooling, classifier head
  (512→256→10). Use He initialization. Add count_parameters() function.

> Test: create a dummy input tensor of shape (1, 3, 224, 224) and pass
  it through the model. Print the output shape and parameter count.
```

### Session 3: Training

```
> Create src/train.py with: FocalLoss class, EarlyStopping class,
  train_one_epoch() with AMP, validate() function, and train_model()
  that ties everything together with AdamW + CosineAnnealingWarmRestarts
  + TensorBoard logging.

> Create main.py that imports everything, sets seeds, loads data,
  creates model, trains, evaluates, and saves results.

> Run: python main.py
```

### Session 4: Evaluation

```
> Create src/evaluate.py with: evaluate_model(), classification_report,
  confusion matrix plot, and training history plots.

> Run evaluation on the test set and save all plots to results/plots/
```

---

## Part 4: Expected Results & What to Watch For

### What Good Training Looks Like:
- Train accuracy climbs steadily: 50% → 70% → 85% → 90%+
- Val accuracy follows but is ~2-5% lower (this gap is normal)
- Loss decreases smoothly
- No sudden spikes or crashes

### Warning Signs:
| Sign | What It Means | Fix |
|------|--------------|-----|
| Train acc: 99%, Val acc: 75% | **Overfitting** — model memorized training data | More dropout, more augmentation, reduce model size |
| Both accuracies stuck at ~60% | **Underfitting** — model too simple | Add more layers, increase channels, train longer |
| Val loss goes UP while train loss goes DOWN | **Overfitting starting** | Early stopping should catch this |
| Loss is NaN or Inf | **Numerical instability** | Lower learning rate, check data for corrupted images |

### Realistic Accuracy Targets:
- **Custom CNN from scratch on mixed data: 88–93%** would be excellent
- **Don't expect 99%** — that only happens on clean lab-only datasets
- **Per-class F1 > 0.85** for all classes would be a strong result

---

## Summary: The Order of Operations

```
 1. Setup environment (conda + PyTorch + CUDA)          ← Do first
 2. Install Claude Code                                  ← Your AI coding assistant
 3. Create src/config.py                                 ← Settings
 4. Create src/data_loader.py                            ← Data pipeline
 5. Create src/model.py                                  ← CNN architecture
 6. Test data loading + model forward pass               ← Verify before training
 7. Create src/train.py                                  ← Training engine
 8. Create main.py                                       ← Orchestrator
 9. Run training (python main.py)                        ← ~1-2 hours on 4070 Ti
10. Create src/evaluate.py                               ← Analyze results
11. Generate plots + metrics                             ← For your capstone report
12. Iterate: adjust hyperparameters, try variations      ← The fun part
```
