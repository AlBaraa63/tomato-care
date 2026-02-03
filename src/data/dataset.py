"""
dataset.py - Data Loading for TomatoCare

Handles loading tomato leaf images and creating PyTorch DataLoaders.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from configs.config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, 
    IMAGENET_MEAN, IMAGENET_STD, EXPLORATION_DIR
)

# Handle both direct execution and module import
try:
    from .transforms import get_train_transforms, get_val_transforms
except ImportError:
    from transforms import get_train_transforms, get_val_transforms


def create_dataloaders(data_dir=None, batch_size=None):
    """
    Create DataLoaders for train, val, and test splits.
    
    Args:
        data_dir: Path to data directory (default: from config)
        batch_size: Batch size (default: from config)
    
    Returns:
        loaders: Dict with 'train', 'val', 'test' DataLoaders
        class_names: List of class names
    """
    data_dir = data_dir or DATA_DIR
    batch_size = batch_size or BATCH_SIZE
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    # Check directories exist
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms())
    val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transforms())
    test_dataset = datasets.ImageFolder(test_dir, transform=get_val_transforms())
    
    class_names = train_dataset.classes
    
    # Create loaders
    use_cuda = torch.cuda.is_available()
    loaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=NUM_WORKERS, 
            pin_memory=use_cuda
        ),
        'val': DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=NUM_WORKERS, 
            pin_memory=use_cuda
        ),
        'test': DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=NUM_WORKERS, 
            pin_memory=use_cuda
        )
    }
    
    return loaders, class_names


def denormalize(tensor):
    """
    Reverse normalization for visualization.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    t = tensor.clone().detach().cpu()
    t = t * std + mean
    return t.clamp(0, 1).permute(1, 2, 0).numpy()


def visualize_augmentations(data_dir=None, num_versions=5, save_dir=None):
    """Visualize how training augmentation transforms an image."""
    data_dir = data_dir or DATA_DIR
    save_dir = save_dir or EXPLORATION_DIR
    
    train_dir = os.path.join(data_dir, "train")
    classes = os.listdir(train_dir)
    class_name = classes[0]
    class_path = os.path.join(train_dir, class_name)
    
    image_name = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))][0]
    img_path = os.path.join(class_path, image_name)
    
    original_img = Image.open(img_path).convert("RGB")
    transform = get_train_transforms()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, num_versions + 1, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis("off")
    
    for i in range(num_versions):
        augmented = transform(original_img)
        plt.subplot(1, num_versions + 1, i + 2)
        plt.imshow(denormalize(augmented))
        plt.title(f"Augmented {i+1}")
        plt.axis("off")
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "augmentation_samples.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved augmentation samples to {save_path}")


# === MAIN (for testing) ===
if __name__ == "__main__":
    print("🚀 Initializing TomatoCare DataLoaders...")
    
    try:
        loaders, classes = create_dataloaders()
        
        print(f"✓ Training samples:   {len(loaders['train'].dataset):,}")
        print(f"✓ Validation samples: {len(loaders['val'].dataset):,}")
        print(f"✓ Test samples:       {len(loaders['test'].dataset):,}")
        print(f"✓ Classes ({len(classes)}): {classes}")
        
        # Get a sample batch
        images, labels = next(iter(loaders['train']))
        print(f"✓ Batch shape: {images.shape}")
        
        # Visualize augmentations
        visualize_augmentations()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
