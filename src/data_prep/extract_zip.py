import zipfile
import os

zip_path = r"c:\Users\POTATO\Desktop\Code\tomato-care\data\raw\Tomato Leaf Disease Dataset\Tomato Leaf Disease Dataset\TomatoDataset.zip"
extract_path = r"c:\Users\POTATO\Desktop\Code\tomato-care\data\raw\Tomato Leaf Disease Dataset\Extracted"

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

print(f"Extracting {zip_path} to {extract_path}...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Extraction complete.")
