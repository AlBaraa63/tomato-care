"""
config.py - Centralized Configuration for TomatoCare

All hyperparameters, paths, and settings in one place.
"""

import os
import torch

# ============================================================
# PATHS
# ============================================================

# Project root (2 levels up from configs/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "tomato")
DISEASE_INFO_PATH = os.path.join(PROJECT_ROOT, "data", "disease_info.json")

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
EXPLORATION_DIR = os.path.join(OUTPUT_DIR, "exploration")
TRAINING_DIR = os.path.join(OUTPUT_DIR, "training")
CHECKPOINT_DIR = os.path.join(TRAINING_DIR, "checkpoints")
EVALUATION_DIR = os.path.join(OUTPUT_DIR, "evaluation")
MOBILE_DIR = os.path.join(OUTPUT_DIR, "mobile")

# Best model path
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_CLASSES = 10
IMAGE_SIZE = 224
PRETRAINED = False  # Train from scratch (no ImageNet weights)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# General
BATCH_SIZE = 32
NUM_WORKERS = 4  # Parallel data loading

# Training from scratch (single phase, all layers trainable)
PHASE1_EPOCHS = 50  # More epochs needed for from-scratch training
PHASE1_LR = 1e-3

# Phase 2: Additional fine-tuning
PHASE2_EPOCHS = 30
PHASE2_LR = 1e-4
UNFREEZE_FROM = 0  # All layers trainable from start

# Optimization
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
EARLY_STOP_PATIENCE = 7

# Mixed precision (faster on modern GPUs)
USE_AMP = True


# ============================================================
# DATA CONFIGURATION
# ============================================================

# ImageNet normalization (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CLASS NAMES (matched to folder names)
# ============================================================

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [EXPLORATION_DIR, TRAINING_DIR, CHECKPOINT_DIR, EVALUATION_DIR, MOBILE_DIR]:
        os.makedirs(d, exist_ok=True)


def get_device_info():
    """Print device information."""
    print(f"🖥️  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
    return DEVICE


# Auto-create directories on import
ensure_dirs()
