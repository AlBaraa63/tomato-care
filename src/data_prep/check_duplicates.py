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

def find_duplicates(data_dir):
    data_path = Path(data_dir)
    hash_map = defaultdict(list)
    
    logging.info(f"Scanning for duplicates in {data_path}...")
    
    file_count = 0
    # Walk through all subdirectories (train/val/test and categories)
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
                file_path = os.path.join(root, file)
                file_hash = get_file_hash(file_path)
                if file_hash:
                    hash_map[file_hash].append(file_path)
                file_count += 1
                
                if file_count % 5000 == 0:
                    logging.info(f"Processed {file_count} files...")

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    
    return duplicates, file_count

def main():
    DATA_DIR = r"c:\Users\POTATO\Desktop\Code\tomato-care\data\processed"
    
    if not os.path.exists(DATA_DIR):
        logging.error(f"Directory not found: {DATA_DIR}")
        return

    duplicates, total_files = find_duplicates(DATA_DIR)
    
    logging.info(f"Scan complete. Total files processed: {total_files}")
    
    if not duplicates:
        logging.info("No exact duplicates (identical hashes) found!")
        return

    num_duplicates = sum(len(paths) - 1 for paths in duplicates.values())
    logging.info(f"Found {len(duplicates)} groups of duplicates, totaling {num_duplicates} duplicate files.")
    
    # Save a report
    report_path = Path(r"c:\Users\POTATO\Desktop\Code\tomato-care\results\duplicate_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Duplicate Report for {DATA_DIR}\n")
        f.write(f"Total files scanned: {total_files}\n")
        f.write(f"Duplicate files found: {num_duplicates}\n\n")
        
        for i, (h, paths) in enumerate(duplicates.items()):
            f.write(f"Group {i+1} (Hash: {h}):\n")
            for path in paths:
                f.write(f"  - {path}\n")
            f.write("\n")
            
    logging.info(f"Detailed report saved to {report_path}")

if __name__ == "__main__":
    main()
