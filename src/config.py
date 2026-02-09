
import torch
from pathlib import Path


class Config:
    # ── Paths ──
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"
    CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # ── Classes (must match actual folder names on disk) ──
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "Bacterial_spot", "Early_blight", "Healthy", "Late_blight",
        "Leaf_Mold", "Septoria_leaf_spot", "Spider_mites",
        "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus"
    ]

    # ── Image Settings ──
    IMG_SIZE = 224
    IN_CHANNELS = 3

    # ── Training Hyperparameters ──
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0

    # ── Scheduler ──
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6

    # ── Early Stopping ──
    PATIENCE = 15

    # ── Focal Loss ──
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1

    # ── MixUp ──
    MIXUP_ALPHA = 0.2

    # ── Device ──
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Mixed Precision ──
    USE_AMP = True

    # ── DataLoader ──
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # ── Reproducibility ──
    SEED = 42