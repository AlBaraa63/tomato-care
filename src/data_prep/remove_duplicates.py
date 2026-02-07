import os
import hashlib
from pathlib import Path
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_file_hash(file_path):
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logging.error(f"Error hashing file {file_path}: {e}")
        return None

def remove_duplicates(data_dir):
    data_path = Path(data_dir)
    hash_map = defaultdict(list)
    
    logging.info(f"Scanning for duplicates in {data_path}...")
    
    file_count = 0
    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
                file_path = Path(root) / file
                file_hash = get_file_hash(file_path)
                if file_hash:
                    # Store as tuple (split_name, full_path)
                    # Path is data/processed/SPLIT/CATEGORY/FILE
                    parts = file_path.parts
                    # find 'processed' in parts
                    try:
                        idx = parts.index('processed')
                        split = parts[idx+1]
                    except (ValueError, IndexError):
                        split = 'unknown'
                        
                    hash_map[file_hash].append({
                        'split': split,
                        'path': file_path
                    })
                file_count += 1
                
                if file_count % 5000 == 0:
                    logging.info(f"Processed {file_count} files...")

    logging.info("Scanning complete. Applying deduplication logic...")
    
    removal_count = 0
    for h, entries in hash_map.items():
        if len(entries) > 1:
            # Sort entries by priority: train > val > test
            # We want to pick ONE to keep and delete the rest.
            priority = {'train': 0, 'val': 1, 'test': 2, 'unknown': 3}
            sorted_entries = sorted(entries, key=lambda x: priority.get(x['split'], 99))
            
            # The first one is the one we keep
            to_keep = sorted_entries[0]
            to_delete = sorted_entries[1:]
            
            for entry in to_delete:
                try:
                    os.remove(entry['path'])
                    removal_count += 1
                except Exception as e:
                    logging.error(f"Failed to delete {entry['path']}: {e}")

    return file_count, removal_count

def main():
    DATA_DIR = r"c:\Users\POTATO\Desktop\Code\tomato-care\data\processed"
    
    if not os.path.exists(DATA_DIR):
        logging.error(f"Directory not found: {DATA_DIR}")
        return

    total_scanned, deleted_count = remove_duplicates(DATA_DIR)
    
    logging.info(f"Deduplication complete!")
    logging.info(f"Total files scanned: {total_scanned}")
    logging.info(f"Duplicate files removed: {deleted_count}")
    logging.info(f"Final unique image count: {total_scanned - deleted_count}")

if __name__ == "__main__":
    main()
