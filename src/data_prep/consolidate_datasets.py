import os
import shutil
from pathlib import Path

# The target directory where everything will be combined
TARGET_BASE = Path(r"c:\Users\POTATO\Desktop\Code\tomato-care\data\processed")

# Source directories to merge INTO the target
SOURCES = {
    "mendeley": Path(r"c:\Users\POTATO\Desktop\Code\tomato-care\data\processed_mendeley"),
    "plantdoc": Path(r"c:\Users\POTATO\Desktop\Code\tomato-care\data\processed_plantdoc"),
    "tomato_leaf": Path(r"c:\Users\POTATO\Desktop\Code\tomato-care\data\processed_tomato_leaf")
}

def consolidate():
    print(f"Consolidating datasets into {TARGET_BASE}...")
    
    for dataset_name, source_path in SOURCES.items():
        if not source_path.exists():
            print(f"Skipping {dataset_name}: Source path {source_path} does not exist.")
            continue
            
        print(f"Processing source: {dataset_name}...")
        
        # Iterate through splits: train, val, test
        for split in ['train', 'val', 'test']:
            split_src = source_path / split
            if not split_src.exists():
                continue
                
            # Iterate through categories
            for category_path in split_src.iterdir():
                if category_path.is_dir():
                    category = category_path.name
                    target_dir = TARGET_BASE / split / category
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy files with prefix to avoid collision
                    for image_file in category_path.iterdir():
                        if image_file.is_file():
                            new_filename = f"{dataset_name}_{image_file.name}"
                            target_file = target_dir / new_filename
                            
                            # Use copy2 to preserve metadata if possible
                            try:
                                shutil.copy2(image_file, target_file)
                            except Exception as e:
                                print(f"Error copying {image_file} to {target_file}: {e}")
                                
        print(f"Finished merging {dataset_name}.")

    print("Consolidation complete.")

if __name__ == "__main__":
    consolidate()
