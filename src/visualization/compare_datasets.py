import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Dataset paths and names
DATASETS = {
    "Ashishmotwani": "data/processed",
    "Mendeley": "data/processed_mendeley",
    "PlantDoc": "data/processed_plantdoc",
    "Tomato Leaf": "data/processed_tomato_leaf"
}

OUTPUT_DIR = Path("results/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def count_images(base_dir):
    """
    Counts images in train, val, test subdirectories for each category.
    Returns a list of dictionaries.
    """
    data = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Warning: {base_dir} does not exist.")
        return data

    for split in ['train', 'val', 'test']:
        split_path = base_path / split
        if not split_path.exists():
            continue
            
        for category_path in split_path.iterdir():
            if category_path.is_dir():
                category = category_path.name
                count = len([f for f in category_path.iterdir() if f.is_file()])
                data.append({
                    "Split": split,
                    "Category": category,
                    "Count": count
                })
    return data

def main():
    all_data = []

    # 1. Collect Data
    for dataset_name, dataset_path in DATASETS.items():
        print(f"Processing {dataset_name}...")
        dataset_counts = count_images(dataset_path)
        for record in dataset_counts:
            record["Dataset"] = dataset_name
        all_data.extend(dataset_counts)

    if not all_data:
        print("No data found!")
        return

    df = pd.DataFrame(all_data)

    # 2. Save Statistics
    stats_file = OUTPUT_DIR / "comparison_stats.md"
    with open(stats_file, "w") as f:
        f.write("# Dataset Comparison Statistics\n\n")
        
        # Summary by Dataset
        f.write("## Total Images by Dataset\n")
        total_by_dataset = df.groupby("Dataset")["Count"].sum().reset_index()
        f.write(total_by_dataset.to_markdown(index=False))
        f.write("\n\n")

        # Summary by Dataset and Split
        f.write("## Images by Dataset and Split\n")
        split_stats = df.groupby(["Dataset", "Split"])["Count"].sum().reset_index()
        f.write(split_stats.to_markdown(index=False))
        f.write("\n\n")

        # Detailed Category Breakdown
        f.write("## Category Breakdown per Dataset\n")
        category_stats = df.groupby(["Dataset", "Category"])["Count"].sum().reset_index()
        pivot_category = category_stats.pivot(index="Category", columns="Dataset", values="Count").fillna(0).astype(int)
        f.write(pivot_category.to_markdown())
        f.write("\n\n")

    print(f"Statistics saved to {stats_file}")

    # 3. Visualizations

    # Set style
    sns.set_style("whitegrid")

    # Plot 1: Total Images per Dataset (Grouped by Category)
    plt.figure(figsize=(15, 8))
    category_dataset_stats = df.groupby(["Dataset", "Category"])["Count"].sum().reset_index()
    sns.barplot(data=category_dataset_stats, x="Category", y="Count", hue="Dataset", palette="viridis")
    plt.title("Number of Images per Category by Dataset")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_comparison.png")
    plt.close()

    # Plot 2: Total Dataset Sizes
    plt.figure(figsize=(10, 6))
    dataset_totals = df.groupby("Dataset")["Count"].sum().reset_index()
    sns.barplot(data=dataset_totals, x="Dataset", y="Count", palette="magma")
    plt.title("Total Dataset Sizes")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_sizes.png")
    plt.close()

    # Plot 3: Split Distribution per Dataset (Stacked Bar)
    plt.figure(figsize=(10, 6))
    split_counts = df.groupby(["Dataset", "Split"])["Count"].sum().reset_index()
    pivot_split = split_counts.pivot(index="Dataset", columns="Split", values="Count")
    pivot_split = pivot_split.reindex(columns=['train', 'val', 'test']) # Ensure order
    
    pivot_split.plot(kind='bar', stacked=True, color=['#3498db', '#2ecc71', '#e74c3c'], figsize=(10, 6))
    plt.title("Split Distribution by Dataset (Train/Val/Test)")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=0)
    plt.legend(title="Split")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "split_distribution.png")
    plt.close()

    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
