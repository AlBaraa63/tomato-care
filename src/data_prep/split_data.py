import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping for the 10 specific categories from ashishmotwani dataset
# We exclude 'powdery_mildew' to keep to the standard 10 categories.
CATEGORY_MAPPING = {
    # Bacterial Spot
    "Bacterial_spot": "Bacterial_spot", 
    "Bacterial_Spot": "Bacterial_spot",
    "Tomato___Bacterial_spot": "Bacterial_spot",

    # Early Blight
    "Early_blight": "Early_blight",
    "Early_Blight": "Early_blight",
    "Tomato___Early_blight": "Early_blight",

    # Late Blight
    "Late_blight": "Late_blight",
    "Late_Blight": "Late_blight",
    "Tomato___Late_blight": "Late_blight",

    # Leaf Mold
    "Leaf_Mold": "Leaf_Mold",
    "Tomato___Leaf_Mold": "Leaf_Mold",

    # Septoria Leaf Spot
    "Septoria_leaf_spot": "Septoria_leaf_spot",
    "Septoria_Leaf_Spot": "Septoria_leaf_spot",
    "Tomato___Septoria_leaf_spot": "Septoria_leaf_spot",

    # Spider Mites
    "Spider_mites Two-spotted_spider_mite": "Spider_mites",
    "Spider_Mites": "Spider_mites",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider_mites",

    # Target Spot
    "Target_Spot": "Target_Spot",
    "Tomato___Target_Spot": "Target_Spot",

    # Yellow Leaf Curl Virus
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato_Yellow_Leaf_Curl_Virus",
    "Yellow_Leaf_Curl_Virus": "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato_Yellow_Leaf_Curl_Virus",

    # Mosaic Virus
    "Tomato_mosaic_virus": "Tomato_mosaic_virus",
    "Mosaic_Virus": "Tomato_mosaic_virus",
    "Tomato___Tomato_mosaic_virus": "Tomato_mosaic_virus",

    # Healthy
    "healthy": "Healthy",
    "Healthy": "Healthy",
    "Tomato___healthy": "Healthy"
}

def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the dataset into train, val, and test sets.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        logging.error(f"Source directory {source_path} does not exist.")
        return

    # Check for split ratios
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        logging.error("Split ratios must sum to 1.0")
        return

    # Clean target directory to avoid contamination with old folders
    if target_path.exists():
        logging.info(f"Cleaning target directory {target_path}...")
        shutil.rmtree(target_path)
    
    target_path.mkdir(parents=True, exist_ok=True)

    # Create target directories
    for split in ['train', 'val', 'test']:
        (target_path / split).mkdir(parents=True, exist_ok=True)

    # Gather all images and group by normalized category
    category_files = defaultdict(list)
    
    logging.info(f"Scanning for images in {source_path}...")
    
    # Walk through source directory
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
                # Get the immediate parent folder name as the raw category
                raw_category = os.path.basename(root)
                
                # Check mapping
                if raw_category in CATEGORY_MAPPING:
                    target_category = CATEGORY_MAPPING[raw_category]
                    
                    if not file.startswith('.'):
                         category_files[target_category].append(os.path.join(root, file))
    
    logging.info(f"Found {len(category_files)} valid target categories.")

    # Process each category
    for category, files in category_files.items():
        logging.info(f"Processing category: {category} ({len(files)} images)")
        
        # Shuffle files
        random.shuffle(files)
        
        # Calculate split indices
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Remaining goes to test
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Helper to copy files
        def copy_files(file_list, split_name):
            dest_dir = target_path / split_name / category
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in file_list:
                try:
                    shutil.copy2(f, dest_dir)
                except Exception as e:
                    logging.error(f"Failed to copy {f} to {dest_dir}: {e}")

        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')
        
        logging.info(f"  Split {category}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    logging.info("Data splitting complete.")

if __name__ == "__main__":
    # Process Tomato Leaf Disease Dataset
    SOURCE_DIR_TOMATO_LEAF = r"c:\Users\POTATO\Desktop\Code\tomato-care\data\raw\Tomato Leaf Disease Dataset\Extracted\TomatoDataset"
    TARGET_DIR_TOMATO_LEAF = r"c:\Users\POTATO\Desktop\Code\tomato-care\data\processed_tomato_leaf"
    
    logging.info("--- Processing Tomato Leaf Disease Dataset ---")
    split_dataset(SOURCE_DIR_TOMATO_LEAF, TARGET_DIR_TOMATO_LEAF)
