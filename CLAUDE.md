# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TomatoCare is a deep learning project for classifying tomato leaf diseases from images. It uses a custom CNN architecture (TomatoCareNet) built from scratch with PyTorch, trained on merged data from four sources: PlantVillage, PlantDoc, TomatoVillage, and Mendeley. The model classifies into 10 classes (9 diseases + healthy) and is optimized for mobile deployment.

## Commands

### Setup
```bash
# Install PyTorch with CUDA first (adjust cu124 to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

### Training
```bash
# Full training pipeline (loads data, trains, evaluates, generates Grad-CAM, saves model)
python main.py
```

### TensorBoard
```bash
tensorboard --logdir results/tensorboard
```

## Architecture

### Training Pipeline (`main.py`)
The entry point runs the full pipeline sequentially: seed setup → data loading → model creation → training → evaluation → TTA evaluation → Grad-CAM generation → model save. All configuration is centralized in `src/config.py` (hyperparameters, paths, device settings).

### Model (`src/model.py`)
**TomatoCareNet** is a custom 4-block CNN with:
- `ConvBlock`: Residual conv blocks (Conv3x3 → BN → ReLU → Conv3x3 → BN + shortcut → ReLU → MaxPool → Dropout). Each block halves spatial dimensions: 224→112→56→28→14.
- `SEBlock`: Squeeze-and-Excitation channel attention after the last conv block.
- Global Average Pooling → 2-layer classifier (256→512→256→10).
- Kaiming initialization throughout.

### Data Pipeline (`src/data_loader.py`)
- `TomatoDataset`: Custom PyTorch Dataset loading from class-subfolder structure (`data/processed/{train,val,test}/{class_name}/`).
- Augmentation via Albumentations (geometric, color, noise, occlusion for train; resize+normalize only for val/test).
- MixUp augmentation applied at training time in the training loop.
- Class weights computed inversely proportional to frequency for the focal loss.

### Training Loop (`src/train.py`)
- **Loss**: `FocalLoss` with class weights, focal gamma, and label smoothing.
- **Optimizer**: AdamW with weight decay.
- **Scheduler**: Linear warmup (5 epochs) → CosineAnnealingWarmRestarts.
- **Mixed precision** (AMP) enabled by default.
- **Early stopping** on validation loss (patience=15).
- Best checkpoint saved to `models/checkpoints/best_model.pth`.
- TensorBoard logging of loss, accuracy, and LR per epoch.

### Evaluation (`src/evaluate.py`)
- Standard evaluation and Test-Time Augmentation (TTA) with averaged predictions over multiple augmented views.
- Outputs classification reports (sklearn) and confusion matrix heatmaps (seaborn).

### Explainability (`src/gradcam.py`)
Grad-CAM hooks into `model.block4.conv2[-1]` (last BN in the deepest conv block) to produce heatmaps.

### Data Preparation (`src/data_prep/`)
One-off scripts for dataset preparation: extracting archives, consolidating sources, splitting into train/val/test (70/15/15), checking and removing duplicates. These are run once before training, not part of the main pipeline.

### Visualization (`src/visualization/`)
Standalone scripts for visualizing dataset distributions, augmentation effects, and comparing datasets. Not used during training.

## Data Layout

Images are organized as `data/processed/{train,val,test}/{class_name}/*.JPG`. The 10 class folder names must match `Config.CLASS_NAMES` in `src/config.py` exactly. Raw datasets in `data/raw/` should never be modified.

## Key Conventions

- All paths are derived from `Config.PROJECT_ROOT` using `pathlib.Path` — avoid hardcoded paths.
- ImageNet normalization constants (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) are used throughout for all transforms.
- The `src/` package is imported as `from src.module import ...` (project root must be in Python path).
- Model outputs raw logits; softmax is applied only during evaluation/inference.
- Augmentation is applied only to training data, never to validation or test sets.
