import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import Counter
from src.config import Config


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
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            p=0.5
        ),

        # Color: simulate different lighting conditions
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),

        # Noise: simulate camera sensor noise
        A.GaussNoise(p=0.2),

        # Occlusion: simulate partially hidden leaves
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(20, 40),
            hole_width_range=(20, 40),
            fill="random",
            p=0.3
        ),

        # Normalize + convert to PyTorch tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
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
        self.samples = []
        self.class_to_idx = {}

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
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


# ── MixUp Augmentation ──

def mixup_data(x, y, alpha=0.2):
    """
    Blends two random images and their labels together.
    Forces the model to learn more robust, interpolated features.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ── Class Weights Calculation ──

def compute_class_weights(dataset):
    """
    Calculate weights inversely proportional to class frequency.
    Makes the model pay equal attention to all classes.
    """
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    total = len(labels)
    num_classes = len(class_counts)

    weights = []
    for i in range(num_classes):
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
    val_dataset = TomatoDataset(Config.VAL_DIR, transform=get_val_transforms())
    test_dataset = TomatoDataset(Config.TEST_DIR, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
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
