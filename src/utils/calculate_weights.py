import os
import json
import numpy as np
import logging
from pathlib import Path
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_class_weights(data_dir, output_file):
    train_dir = Path(data_dir) / 'train'
    if not train_dir.exists():
        logging.error(f"Train directory {train_dir} does not exist.")
        return

    # Count samples per class
    class_counts = {}
    total_samples = 0
    
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    
    for cls in classes:
        cls_dir = train_dir / cls
        count = len([f for f in cls_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        class_counts[cls] = count
        total_samples += count

    num_classes = len(class_counts)
    logging.info(f"Total samples: {total_samples}, Num classes: {num_classes}")

    # Calculate weights: Total / (Num_Classes * Class_Count)
    class_weights = {}
    for cls, count in class_counts.items():
        if count > 0:
            weight = total_samples / (num_classes * count)
        else:
            weight = 0
        class_weights[cls] = weight
        logging.info(f"Class: {cls}, Count: {count}, Weight: {weight:.4f}")

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(class_weights, f, indent=4)
    
    logging.info(f"Class weights saved to {output_file}")
    
    # Save as list for PyTorch CrossEntropyLoss order
    weights_list = [class_weights[cls] for cls in classes]
    weights_list_file = Path(output_file).parent / 'class_weights_list.json'
    with open(weights_list_file, 'w') as f:
         json.dump(weights_list, f)
    logging.info(f"Class weights list saved to {weights_list_file}")

if __name__ == "__main__":
    DATA_DIR = r"c:\Users\POTATO\Desktop\Code\tomato-care\data\processed"
    OUTPUT_FILE = r"c:\Users\POTATO\Desktop\Code\tomato-care\reports\class_weights.json"
    
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "data/processed"
        OUTPUT_FILE = "reports/class_weights.json"

    calculate_class_weights(DATA_DIR, OUTPUT_FILE)
