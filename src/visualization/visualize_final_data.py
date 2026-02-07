import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
import random
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import Config

# Configuration
DATA_DIR = Config.DATA_DIR
OUTPUT_DIR = Config.RESULTS_DIR / "final_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_dataset_stats():
    stats = []
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = DATA_DIR / split
        if not split_path.exists():
            continue
        for category_path in split_path.iterdir():
            if category_path.is_dir():
                count = len(list(category_path.glob('*')))
                stats.append({
                    'Split': split,
                    'Category': category_path.name,
                    'Count': count
                })
    return pd.DataFrame(stats)

def plot_distribution(df):
    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid")
    
    # Pivot for stacked bar
    pivot_df = df.pivot_table(index='Category', columns='Split', values='Count', aggfunc='sum', fill_value=0)
    # Reorder columns
    available_splits = [s for s in ['train', 'val', 'test'] if s in pivot_df.columns]
    pivot_df = pivot_df[available_splits]
    
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(15, 8), colormap='viridis', edgecolor='black')
    
    plt.title('Final Combined Dataset Distribution', fontsize=18, pad=20)
    plt.xlabel('Disease Category', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(title='Split', fontsize=12)
    
    # Add value labels
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fontsize=10, color='white', weight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_distribution.png", dpi=300)
    plt.close()

def plot_sample_grid():
    categories = sorted([d.name for d in (DATA_DIR / 'train').iterdir() if d.is_dir()])
    n_categories = len(categories)
    
    rows = 2
    cols = 5
    plt.figure(figsize=(20, 8))
    
    for i, category in enumerate(categories):
        cat_path = DATA_DIR / 'train' / category
        images = list(cat_path.glob('*'))
        if not images:
            continue
            
        img_path = random.choice(images)
        img = Image.open(img_path)
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(category.replace('_', ' '), fontsize=14, fontweight='bold', pad=10)
        plt.axis('off')
        
    plt.suptitle("Combined Dataset: Sample Images from Each Category", fontsize=22, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_images.png", dpi=300)
    plt.close()

def main():
    print("Generating final visualizations...")
    
    # 1. Stats and Distribution
    df = get_dataset_stats()
    if df.empty:
        print("No data found!")
        return
    
    plot_distribution(df)
    print(f"Distribution plot saved to {OUTPUT_DIR / 'final_distribution.png'}")
    
    # 2. Sample Grid
    try:
        plot_sample_grid()
        print(f"Sample grid saved to {OUTPUT_DIR / 'sample_images.png'}")
    except Exception as e:
        print(f"Error generating sample grid: {e}")
        
    # 3. Save Summary Table
    pivot_df = df.pivot_table(index='Category', columns='Split', values='Count', aggfunc='sum', fill_value=0)
    pivot_df['Total'] = pivot_df.sum(axis=1)
    summary_file = OUTPUT_DIR / "final_stats.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Final Consolidated Dataset Summary\n\n")
        f.write(pivot_df.to_markdown())
        f.write(f"\n\n**Total Images in Dataset:** {df['Count'].sum()}\n")
    
    print(f"Summary statistics saved to {summary_file}")
    print("Done.")

if __name__ == "__main__":
    main()
