"""
evaluate.py - TomatoCare Model Evaluation

Comprehensive metrics and visualizations.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from tqdm import tqdm

from configs.config import (
    DEVICE, EVALUATION_DIR, BEST_MODEL_PATH, NUM_CLASSES,
    get_device_info, ensure_dirs
)
from src.data.dataset import create_dataloaders, denormalize
from src.models.classifier import get_model


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    model = get_model(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']} | Val Acc: {checkpoint['val_acc']:.2f}%")
    return model


def get_predictions(model, loader, device):
    """Get predictions from model."""
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            outputs = model(images.to(device))
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred, class_names):
    """Compute all metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision_macro': precision_score(y_true, y_pred, average='macro') * 100,
        'recall_macro': recall_score(y_true, y_pred, average='macro') * 100,
        'f1_macro': f1_score(y_true, y_pred, average='macro') * 100,
        'per_class': {}
    }
    
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    for i, name in enumerate(class_names):
        metrics['per_class'][name] = {
            'precision': precision[i] * 100,
            'recall': recall[i] * 100,
            'f1': f1[i] * 100
        }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    display_names = [n.replace('Tomato___', '').replace('_', ' ')[:15] for n in class_names]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=display_names, yticklabels=display_names)
    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Saved: {save_path}")


def plot_per_class_accuracy(metrics, class_names, save_path):
    """Plot per-class recall."""
    display_names = [n.replace('Tomato___', '').replace('_', ' ') for n in class_names]
    recalls = [metrics['per_class'][n]['recall'] for n in class_names]
    colors = ['#2ecc71' if r >= 90 else '#f39c12' if r >= 80 else '#e74c3c' for r in recalls]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(display_names, recalls, color=colors)
    for bar, val in zip(bars, recalls):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center')
    
    plt.xlabel('Recall (%)')
    plt.title('Per-Class Recall', fontweight='bold')
    plt.xlim(0, 110)
    plt.axvline(x=90, color='green', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Saved: {save_path}")


def save_misclassified(model, loader, class_names, device, save_path, max_samples=25):
    """Save misclassified samples grid."""
    misclassified = []
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            _, preds = outputs.max(1)
            
            for i in range(len(labels)):
                if preds[i].item() != labels[i].item():
                    misclassified.append({
                        'image': images[i],
                        'true': labels[i].item(),
                        'pred': preds[i].item()
                    })
            if len(misclassified) >= max_samples:
                break
    
    if not misclassified:
        print("✅ No misclassified samples!")
        return
    
    n = min(len(misclassified), max_samples)
    cols, rows = 5, (n + 4) // 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < n:
            s = misclassified[i]
            ax.imshow(denormalize(s['image']))
            true_name = class_names[s['true']].replace('Tomato___', '')[:12]
            pred_name = class_names[s['pred']].replace('Tomato___', '')[:12]
            ax.set_title(f"T:{true_name}\nP:{pred_name}", fontsize=8, color='red')
        ax.axis('off')
    
    plt.suptitle('Misclassified Samples', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Saved: {save_path}")


def print_report(metrics, class_names):
    """Print evaluation report."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION REPORT")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision_macro']:.2f}%")
    print(f"Recall:    {metrics['recall_macro']:.2f}%")
    print(f"F1 Score:  {metrics['f1_macro']:.2f}%")
    print("-" * 60)
    print(f"{'Class':<30} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 60)
    for name in class_names:
        display = name.replace('Tomato___', '')[:28]
        pc = metrics['per_class'][name]
        print(f"{display:<30} {pc['precision']:>7.1f}% {pc['recall']:>7.1f}% {pc['f1']:>7.1f}%")
    print("=" * 60)
    print("✅ TARGET MET!" if metrics['accuracy'] >= 90 else f"⚠️ Below target: {metrics['accuracy']:.2f}%")


def evaluate(checkpoint_path=None):
    """Run full evaluation."""
    ensure_dirs()
    device = get_device_info()
    checkpoint_path = checkpoint_path or BEST_MODEL_PATH
    
    print(f"\n📂 Loading data...")
    loaders, class_names = create_dataloaders()
    print(f"   Test samples: {len(loaders['test'].dataset):,}")
    
    print(f"\n🧠 Loading model...")
    model = load_model(checkpoint_path, device)
    
    print(f"\n🔍 Running inference...")
    y_pred, y_true = get_predictions(model, loaders['test'], device)
    
    print(f"\n📊 Computing metrics...")
    metrics = compute_metrics(y_true, y_pred, class_names)
    print_report(metrics, class_names)
    
    # Save metrics
    with open(os.path.join(EVALUATION_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plots
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(EVALUATION_DIR, 'confusion_matrix.png'))
    plot_per_class_accuracy(metrics, class_names, os.path.join(EVALUATION_DIR, 'per_class_accuracy.png'))
    save_misclassified(model, loaders['test'], class_names, device, os.path.join(EVALUATION_DIR, 'misclassified.png'))
    
    print(f"\n✅ Evaluation complete! Results in: {EVALUATION_DIR}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint)
