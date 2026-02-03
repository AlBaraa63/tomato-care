"""
data_split.py - Create 70/15/15 Train/Val/Test Split

Current state:
- train/: 1,000 images per class
- val/: 100 images per class
Total: 1,100 images per class

Target state:
- train/: 770 images per class (70%)
- val/: 165 images per class (15%)
- test/: 165 images per class (15%)
"""

import os
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = os.path.join("data", "tomato")
SEED = 42

# Ensure reproducibility
random.seed(SEED)

# ============================================================
# MAIN FUNCTION
# ============================================================

def perform_split():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    # Create test directory if it doesn't exist
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"✅ Created test folder: {test_dir}")

    # Get class names
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"\n📂 Found {len(classes)} classes")
    print("-" * 60)

    for cls in classes:
        cls_train_path = os.path.join(train_dir, cls)
        cls_val_path = os.path.join(val_dir, cls)
        cls_test_path = os.path.join(test_dir, cls)

        os.makedirs(cls_test_path, exist_ok=True)

        # 1. Collect all images and their current paths
        # Using list of tuples: (filename, current_full_path)
        all_images = []
        
        train_imgs = [f for f in os.listdir(cls_train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for f in train_imgs:
            all_images.append((f, os.path.join(cls_train_path, f), 'train'))
            
        val_imgs = [f for f in os.listdir(cls_val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for f in val_imgs:
            all_images.append((f, os.path.join(cls_val_path, f), 'val'))

        # 2. Shuffle
        random.shuffle(all_images)

        # 3. Calculate split sizes
        total = len(all_images)
        train_count = int(total * 0.70)
        val_count = int(total * 0.15)
        # test_count will be the rest

        train_set = all_images[:train_count]
        val_set = all_images[train_count:train_count + val_count]
        test_set = all_images[train_count + val_count:]

        # 4. Handle moves to temporary locations to avoid conflicts 
        # (Actually, since we know unique filenames, we can move directly to temporary names 
        # or just handle the 'move-out' first)
        
        # We'll use a simple strategy: 
        # Move everything that needs to go to 'test' or 'val' (that isn't already there)
        # And move from 'val' to 'train' if needed.
        
        # Move to VAL
        for f, src, current_split in val_set:
            if current_split != 'val':
                dst = os.path.join(cls_val_path, f)
                shutil.move(src, dst)

        # Move to TEST
        for f, src, current_split in test_set:
            if current_split != 'test':
                dst = os.path.join(cls_test_path, f)
                shutil.move(src, dst)
        
        # Move to TRAIN (from val)
        # Specifically, any image in the train_set that is currently in 'val'
        for f, src, current_split in train_set:
            if current_split == 'val':
                dst = os.path.join(cls_train_path, f)
                shutil.move(src, dst)

        print(f"  {cls:35s} | Train: {len(train_set):4d} | Val: {len(val_set):4d} | Test: {len(test_set):4d}")

    print("-" * 60)
    print("\n✅ 70/15/15 Split Complete!")

def verify_split():
    print("\n" + "=" * 60)
    print("📊 FINAL DATASET VERIFICATION")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_path):
            print(f"❌ {split}/ folder missing!")
            continue
            
        print(f"\n📂 {split}/")
        classes = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        
        total_in_split = 0
        for cls in classes:
            count = len([f for f in os.listdir(os.path.join(split_path, cls)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"   {cls:35s}: {count:4d}")
            total_in_split += count
        print(f"   {'---':35s}: ----")
        print(f"   {'TOTAL':35s}: {total_in_split:,}")

if __name__ == "__main__":
    perform_split()
    verify_split()