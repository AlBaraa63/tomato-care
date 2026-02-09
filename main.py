
"""
TomatoCare - Main Training Script
Run: python main.py
"""
import torch
import random
import numpy as np
from src.config import Config
from src.data_loader import get_dataloaders, TomatoDataset, get_val_transforms
from src.model import TomatoCareNet, count_parameters
from src.train import train_model
from src.evaluate import (
    evaluate_model, evaluate_model_tta,
    print_classification_report,
    plot_confusion_matrix, plot_training_history
)
from src.gradcam import visualize_gradcam


def set_seed(seed=42):
    """Ensure reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 1. Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Create output directories
    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (Config.RESULTS_DIR / "plots").mkdir(exist_ok=True)
    (Config.RESULTS_DIR / "metrics").mkdir(exist_ok=True)
    (Config.RESULTS_DIR / "gradcam").mkdir(exist_ok=True)

    # 3. Load data
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()

    # 4. Create model
    print("\nBuilding TomatoCareNet...")
    model = TomatoCareNet(num_classes=Config.NUM_CLASSES)
    count_parameters(model)

    # 5. Train
    model, history = train_model(model, train_loader, val_loader, class_weights)

    # 6. Plot training history
    plot_training_history(
        history,
        save_path=Config.RESULTS_DIR / "plots" / "training_history.png"
    )

    # 7. Evaluate on test set (standard)
    print("\nEvaluating on test set...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader)
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=Config.RESULTS_DIR / "plots" / "confusion_matrix.png"
    )

    # 8. Evaluate with TTA (for best possible test metrics)
    print("\nEvaluating with Test Time Augmentation (5 augments)...")
    test_dataset = TomatoDataset(Config.TEST_DIR, transform=get_val_transforms())
    y_true_tta, y_pred_tta, _ = evaluate_model_tta(model, test_dataset, num_augments=5)
    print_classification_report(y_true_tta, y_pred_tta)
    plot_confusion_matrix(
        y_true_tta, y_pred_tta,
        save_path=Config.RESULTS_DIR / "plots" / "confusion_matrix_tta.png"
    )

    # 9. Grad-CAM on a sample from each class
    print("\nGenerating Grad-CAM visualizations...")
    model.to(Config.DEVICE)
    for class_name in Config.CLASS_NAMES:
        class_dir = Config.TEST_DIR / class_name
        if not class_dir.exists():
            continue
        sample = next(class_dir.iterdir(), None)
        if sample:
            visualize_gradcam(
                model, str(sample),
                save_path=Config.RESULTS_DIR / "gradcam" / f"gradcam_{class_name}.png"
            )

    # 10. Save final model
    final_dir = Config.PROJECT_ROOT / "models" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_dir / "tomatocare_final.pth")
    print(f"\nModel saved to models/final/tomatocare_final.pth")


if __name__ == "__main__":
    main()
