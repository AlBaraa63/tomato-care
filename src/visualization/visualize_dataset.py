import os
import matplotlib.pyplot as plt
import pandas as pd
import logging
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_dataset(data_dir, output_dir):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'val', 'test']
    stats = []

    # Collect statistics
    logging.info(f"Scanning directory: {data_path}")
    for split in splits:
        split_dir = data_path / split
        if not split_dir.exists():
            logging.warning(f"Split directory not found: {split_dir}")
            continue
        
        for category_dir in split_dir.iterdir():
            if category_dir.is_dir():
                # Count image files
                count = len([f for f in category_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']])
                if count > 0:
                    stats.append({
                        'Split': split,
                        'Category': category_dir.name,
                        'Count': count
                    })

    if not stats:
        logging.error("No image data found to visualize.")
        return

    df = pd.DataFrame(stats)
    
    # Generate Pivot Table for Plotting
    try:
        pivot_df = df.pivot_table(index='Category', columns='Split', values='Count', aggfunc='sum', fill_value=0)
    except Exception as e:
        logging.error(f"Error creating pivot table: {e}")
        return
    
    # Ensure all splits are present in columns
    for split in splits:
        if split not in pivot_df.columns:
            pivot_df[split] = 0
            
    # Reorder columns
    pivot_df = pivot_df[splits]

    # Plotting using Seaborn style
    sns.set_theme(style="whitegrid")
    
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis', edgecolor='black')
    
    plt.title('Dataset Distribution by Category and Split', fontsize=16)
    plt.xlabel('Disease Category', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Split')
    
    # Add value labels
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fontsize=8, color='white', weight='bold')

    plt.tight_layout()
    
    plot_file = output_path / 'dataset_distribution.png'
    plt.savefig(plot_file, dpi=300)
    plt.close()
    logging.info(f"Plot saved to {plot_file}")

    # Generate Markdown Table matching the requested format
    md_file = output_path / 'dataset_stats.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Dataset Statistics\n\n")
        f.write(pivot_df.to_markdown())
        f.write("\n\n")
        total_images = df['Count'].sum()
        f.write(f"**Total Images:** {total_images}\n")
    
    logging.info(f"Stats saved to {md_file}")
    
    # Print to console for immediate check
    print("\nDataset Statistics Summary:")
    print(pivot_df.to_markdown())
    print(f"\nTotal Images: {total_images}")

if __name__ == "__main__":
    # Define paths based on user environment
    # Processing Tomato Leaf Disease Dataset results
    DATA_DIR = Path(r"c:\Users\POTATO\Desktop\Code\tomato-care\data\processed")
    OUTPUT_DIR = Path(r"c:\Users\POTATO\Desktop\Code\tomato-care\results\plots_combined")
    
    # Verify input directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
    else:
        visualize_dataset(DATA_DIR, OUTPUT_DIR)
