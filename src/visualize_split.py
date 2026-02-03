"""
visualize_split.py - Visualize the new 70/15/15 Train/Val/Test Split
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DATA_DIR = os.path.join("data", "tomato")
OUTPUT_DIR = "outputs"
SPLITS = ['train', 'val', 'test']
COLORS = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red

def get_counts():
    data = {}
    classes = []
    
    # Get all unique class names across all splits
    for split in SPLITS:
        split_path = os.path.join(DATA_DIR, split)
        if os.path.exists(split_path):
            current_classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            for c in current_classes:
                if c not in classes:
                    classes.append(c)
    
    classes.sort()
    
    # Initialize data structure
    for split in SPLITS:
        data[split] = []
        for cls in classes:
            path = os.path.join(DATA_DIR, split, cls)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                data[split].append(count)
            else:
                data[split].append(0)
                
    return classes, data

def draw_chart(classes, data):
    # Shorten class names for display
    display_names = [c.replace('Tomato___', '').replace('_', ' ') for c in classes]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(len(display_names))
    left = np.zeros(len(display_names))
    
    for i, split in enumerate(SPLITS):
        counts = np.array(data[split])
        ax.barh(y_pos, counts, left=left, label=split.capitalize(), color=COLORS[i], edgecolor='white', height=0.7)
        
        # Add text labels for the counts inside the bars if they are large enough
        for j, val in enumerate(counts):
            if val > 50:
                ax.text(left[j] + val/2, y_pos[j], str(val), va='center', ha='center', color='white', fontweight='bold', fontsize=9)
        
        left += counts

    # Aesthetics
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('TomatoCare Dataset Split (70/15/15)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add total count labels at the end of bars
    for i, total in enumerate(left):
        ax.text(total + 10, y_pos[i], f'Total: {int(total)}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "split_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")

if __name__ == "__main__":
    print("📊 Generating dataset split visualization...")
    classes, data = get_counts()
    draw_chart(classes, data)
