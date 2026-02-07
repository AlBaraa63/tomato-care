import torch
import torch.nn as nn
import time
import copy
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from src.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss â€” a smarter version of CrossEntropyLoss.

    DOWN-WEIGHTS easy samples and UP-WEIGHTS hard ones.
    Formula: FL = -alpha * (1 - p)^gamma * log(p)

    gamma=0: Same as regular CrossEntropy
    gamma=2: Hard samples get ~100x more attention than easy ones
    """

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    If no improvement for `patience` epochs, training stops.
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

        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

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

    # Loss Function (Focal Loss with class weights)
    criterion = FocalLoss(weight=class_weights, gamma=Config.FOCAL_GAMMA)

    # Optimizer (AdamW = Adam + proper weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Learning Rate Scheduler (Cosine Annealing with Warm Restarts)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=Config.MIN_LR
    )

    # Mixed Precision Scaler
    scaler = GradScaler(enabled=Config.USE_AMP)

    # Early Stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE)

    # TensorBoard Logger
    writer = SummaryWriter(Config.RESULTS_DIR / "tensorboard")

    # Training History
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': []
    }

    best_val_acc = 0.0
    best_model_state = None

    print(f"\n{'='*60}")
    print(f"  Training TomatoCareNet on {device}")
    print(f"  Epochs: {Config.NUM_EPOCHS} | Batch: {Config.BATCH_SIZE} | LR: {Config.LEARNING_RATE}")
    print(f"  AMP: {Config.USE_AMP} | Focal gamma: {Config.FOCAL_GAMMA}")
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
            Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, Config.CHECKPOINT_DIR / "best_model.pth")
            improved = " * BEST"

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
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    writer.close()

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")

    return model, history