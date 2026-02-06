# 🍅 TomatoCare — Project Structure

## What is this?
This is the folder structure for our TomatoCare project. Every folder has a 
specific purpose, so we always know where to find things.

## Folder Map

```
TomatoCare/
│
├── data/                        ← ALL dataset-related files live here
│   ├── raw/                     ← Original downloaded datasets (NEVER modify these)
│   │   ├── PlantVillage/        ← ~14,500 lab images (10 classes)
│   │   ├── PlantDoc/            ← ~400 real-world images 
│   │   ├── TomatoVillage/       ← ~1,000 field images
│   │   └── Mendeley/            ← ~5,000 Taiwan field images
│   │
│   ├── processed/               ← Cleaned, merged, and split dataset
│   │   ├── train/               ← 70% of data (model learns from this)
│   │   │   ├── Bacterial_Spot/
│   │   │   ├── Early_Blight/
│   │   │   ├── Late_Blight/
│   │   │   ├── Leaf_Mold/
│   │   │   ├── Septoria_Leaf_Spot/
│   │   │   ├── Spider_Mites/
│   │   │   ├── Target_Spot/
│   │   │   ├── Yellow_Leaf_Curl_Virus/
│   │   │   ├── Mosaic_Virus/
│   │   │   └── Healthy/
│   │   ├── val/                 ← 15% of data (model checks itself during training)
│   │   │   └── (same 10 class folders)
│   │   └── test/                ← 15% of data (final exam — model never sees this until the end)
│   │       └── (same 10 class folders)
│   │
│   └── augmented/               ← Extra images created by augmentation (rotated, flipped, etc.)
│
├── notebooks/                   ← Jupyter notebooks (our step-by-step experiments)
│   ├── 01_EDA.ipynb             ← Step 1: Explore & understand the data
│   ├── 02_preprocessing.ipynb   ← Step 2: Clean, merge, split the datasets
│   ├── 03_model_v1.ipynb        ← Step 3: Build & train first model version
│   ├── 04_model_v2.ipynb        ← Step 4: Improve the model
│   ├── 05_evaluation.ipynb      ← Step 5: Test & analyze results
│   └── 06_gradcam.ipynb         ← Step 6: Explainability visualizations
│
├── src/                         ← Reusable Python code (functions we use across notebooks)
│   ├── __init__.py              ← Makes this folder a Python package
│   ├── data_loader.py           ← Functions to load and prepare images
│   ├── augmentation.py          ← Data augmentation functions
│   ├── model.py                 ← Our custom CNN architecture (TomatoCareNet)
│   ├── train.py                 ← Training loop and callbacks
│   ├── evaluate.py              ← Evaluation metrics and plots
│   └── gradcam.py               ← Grad-CAM explainability functions
│
├── models/                      ← Saved model files
│   ├── checkpoints/             ← Auto-saved during training (best weights so far)
│   └── final/                   ← The finished trained model
│       ├── tomatocare_best.h5   ← Best Keras model
│       └── tomatocare.tflite    ← Converted for mobile deployment
│
├── results/                     ← All output results
│   ├── plots/                   ← Training curves, data distribution charts
│   ├── metrics/                 ← Accuracy, F1, confusion matrices (saved as CSV/JSON)
│   └── gradcam/                 ← Grad-CAM heatmap images
│
├── app/                         ← Mobile app code (Flutter/React Native — later phase)
│
├── docs/                        ← Documentation and reports
│   └── TomatoCare_Research.md   ← Our research document
│
├── README.md                    ← This file — project overview
└── requirements.txt             ← Python packages needed to run the project
```

## Why This Structure?

### 🔑 Key Principles:

1. **`raw/` is sacred** — We NEVER modify original downloaded data. If something goes 
   wrong with processing, we can always start fresh from raw data.

2. **`processed/` is our working dataset** — After merging all sources, cleaning, and 
   splitting into train/val/test, this is what the model actually uses.

3. **`notebooks/` are numbered** — So anyone (including future-you) can follow the 
   project step by step, in order.

4. **`src/` avoids code duplication** — Instead of copying the same function into every 
   notebook, we write it once in `src/` and import it everywhere.

5. **`models/checkpoints/` saves progress** — Training can take hours. If it crashes at 
   epoch 80, we don't lose the best model from epoch 65.

6. **`results/` is for evidence** — Every plot, metric, and visualization is saved here.
   This is what goes into your capstone report.

## The Data Pipeline (How Data Flows)

```
Step 1: Download         Step 2: Merge & Clean      Step 3: Split
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│ data/raw/        │     │ Merge all sources│     │ train/ (70%) │
│  PlantVillage/   │────▶│ Resize to 224x224│────▶│ val/   (15%) │
│  PlantDoc/       │     │ Fix labels       │     │ test/  (15%) │
│  TomatoVillage/  │     │ Remove duplicates│     └──────────────┘
│  Mendeley/       │     └──────────────────┘            │
└─────────────────┘                                      ▼
                                                  Step 4: Augment
                                                  ┌──────────────┐
                                                  │ Rotate, flip, │
                                                  │ brightness,   │
                                                  │ zoom, noise   │
                                                  │ (train only!) │
                                                  └──────────────┘
                                                         │
                                                         ▼
                                                  Step 5: Train
                                                  ┌──────────────┐
                                                  │ Feed into CNN │
                                                  │ TomatoCareNet │
                                                  └──────────────┘
```

**Important:** We ONLY augment training data, never validation or test data.
Validation and test must reflect real-world conditions to give honest results.

## Getting Started

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download datasets into data/raw/ (see notebooks/01_EDA.ipynb)

# Step 3: Follow the notebooks in order (01, 02, 03...)
```

## Tech Stack
- **Python 3.10+**
- **TensorFlow / Keras** — Deep learning framework
- **OpenCV** — Image processing
- **Matplotlib / Seaborn** — Visualization
- **scikit-learn** — Metrics and data splitting
- **Albumentations** — Advanced image augmentation
- **NumPy / Pandas** — Data handling