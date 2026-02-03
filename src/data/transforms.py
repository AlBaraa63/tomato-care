"""
transforms.py - Data Augmentation & Preprocessing

Training transforms include augmentation to prevent overfitting.
Validation/test transforms are standard normalization only.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torchvision import transforms
from configs.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms():
    """
    Transforms for TRAINING data including augmentation.
    Helps the model generalize by seeing variations of the same image.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms():
    """
    Transforms for VALIDATION and TEST data.
    Standard resize and normalization without randomness.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_inference_transforms():
    """
    Transforms for single image inference.
    Same as validation transforms.
    """
    return get_val_transforms()
