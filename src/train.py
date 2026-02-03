"""
train.py - TomatoCare Training Pipeline

Two-phase training:
1. Phase 1: Train classifier head (backbone frozen)
2. Phase 2: Fine-tune backbone + classifier
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.config import (
    DEVICE, BATCH_SIZE, TRAINING_DIR, CHECKPOINT_DIR,
    PHASE1_EPOCHS, PHASE1_LR, PHASE2_EPOCHS, PHASE2_LR,
    UNFREEZE_FROM, WEIGHT_DECAY, SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR, EARLY_STOP_PATIENCE, USE_AMP,
    get_device_info, ensure_dirs
)
from src.data.dataset import create_dataloaders
from src.models.classifier import get_model, model_summary


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return running_loss / total, 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, 100. * correct / total


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)


def plot_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    if 'phase2_start' in history:
        axes[0].axvline(x=history['phase2_start'], color='gray', linestyle='--', label='Phase 2')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val', linewidth=2)
    if 'phase2_start' in history:
        axes[1].axvline(x=history['phase2_start'], color='gray', linestyle='--', label='Phase 2')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📈 Training history saved to {save_path}")


def train(args):
    """Main training function."""
    ensure_dirs()
    device = get_device_info()
    
    # Quick test mode
    phase1_epochs = 1 if args.quick_test else PHASE1_EPOCHS
    phase2_epochs = 1 if args.quick_test else PHASE2_EPOCHS
    
    if args.quick_test:
        print("\n⚡ QUICK TEST MODE")
    
    # Load data
    print(f"\n📂 Loading data...")
    loaders, class_names = create_dataloaders(batch_size=args.batch_size)
    print(f"   Classes: {len(class_names)}")
    print(f"   Train: {len(loaders['train'].dataset):,} | Val: {len(loaders['val'].dataset):,}")
    
    # Create model
    print("\n🧠 Creating model...")
    model = get_model(num_classes=len(class_names), freeze_backbone=True)
    model = model.to(device)
    model_summary(model)
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if (USE_AMP and torch.cuda.is_available()) else None
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0
    epochs_no_improve = 0
    total_epochs = 0
    
    # === PHASE 1 ===
    print("\n" + "=" * 60)
    print("🚀 PHASE 1: Training Classifier Head")
    print("=" * 60)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)
    
    for epoch in range(phase1_epochs):
        total_epochs += 1
        print(f"\n📅 Epoch {total_epochs} (Phase 1: {epoch+1}/{phase1_epochs})")
        
        train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, loaders['val'], criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, total_epochs, val_acc, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"   ✅ New best! (Val Acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
    
    # === PHASE 2 ===
    print("\n" + "=" * 60)
    print("🔓 PHASE 2: Fine-tuning Backbone")
    print("=" * 60)
    
    history['phase2_start'] = total_epochs
    model.unfreeze_backbone(unfreeze_from=UNFREEZE_FROM)
    model_summary(model)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE2_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)
    
    for epoch in range(phase2_epochs):
        total_epochs += 1
        print(f"\n📅 Epoch {total_epochs} (Phase 2: {epoch+1}/{phase2_epochs})")
        
        train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, loaders['val'], criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, total_epochs, val_acc, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"   ✅ New best! (Val Acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n⚠️ Early stopping! No improvement for {EARLY_STOP_PATIENCE} epochs.")
                break
    
    # === COMPLETE ===
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"   Total Epochs: {total_epochs}")
    
    # Save history
    with open(os.path.join(TRAINING_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    with open(os.path.join(TRAINING_DIR, 'class_names.json'), 'w') as f:
        json.dump(class_names, f, indent=2)
    
    plot_history(history, os.path.join(TRAINING_DIR, 'training_history.png'))
    
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TomatoCare')
    parser.add_argument('--quick-test', action='store_true', help='Quick test (1 epoch each)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    args = parser.parse_args()
    
    start = time.time()
    best_acc = train(args)
    elapsed = time.time() - start
    
    print(f"\n⏱️ Training time: {elapsed/60:.1f} min")
    print("✅ Target (90%+) achieved!" if best_acc >= 90 else f"⚠️ Target not met: {best_acc:.2f}%")
