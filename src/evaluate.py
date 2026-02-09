import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.amp import autocast
from PIL import Image
from src.config import Config


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


@torch.no_grad()
def evaluate_model_tta(model, test_dataset, device=None, num_augments=5):
    """
    Test Time Augmentation — averages predictions over multiple augmented
    views of each image for a free accuracy boost on the test report.
    Not used on-device (too slow for mobile), only for final evaluation.
    """
    if device is None:
        device = Config.DEVICE

    model.eval()

    clean_transform = A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    tta_transform = A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    all_labels = []
    all_preds = []
    all_probs = []

    for img_path, label in test_dataset.samples:
        image = np.array(Image.open(img_path).convert("RGB"))

        # Clean prediction + N augmented predictions
        views = [clean_transform(image=image)["image"].unsqueeze(0)]
        for _ in range(num_augments):
            views.append(tta_transform(image=image)["image"].unsqueeze(0))

        batch = torch.cat(views, dim=0).to(device)
        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(batch)
        avg_probs = torch.softmax(outputs, dim=1).mean(dim=0)

        all_labels.append(label)
        all_preds.append(avg_probs.argmax().cpu().item())
        all_probs.append(avg_probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_classification_report(y_true, y_pred, class_names=None):
    """Print detailed per-class metrics"""
    if class_names is None:
        class_names = Config.CLASS_NAMES

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "=" * 70)
    print("  CLASSIFICATION REPORT")
    print("=" * 70)
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
    plt.title('Confusion Matrix - TomatoCareNet', fontsize=14)
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
    """Plot training curves (loss, accuracy, learning rate)"""
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
