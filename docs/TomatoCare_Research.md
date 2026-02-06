# TomatoCare — Research & Project Plan
## AI-Powered Tomato Disease Detection (Custom CNN from Scratch)

**Author:** Albaraa  
**Date:** February 2026  
**Project Type:** Capstone Project — Computer Science, Al Ain University

---

## 1. Problem Statement

Tomato is one of the most cultivated crops worldwide, with annual production exceeding 170 million tons. However, various diseases cause **25–50% of annual yield losses**, posing a major threat to food security — particularly in hot, arid regions like the UAE where home gardening is growing in popularity.

Traditional disease detection relies on expert visual inspection, which is:
- Time-consuming and labor-intensive
- Prone to human error (diseases often look similar)
- Inaccessible to home gardeners without agricultural training

**TomatoCare Goal:** Build a **custom CNN from scratch** (no pretrained models) that can accurately classify tomato leaf diseases from **real-world images** and be deployable on mobile/edge devices for UAE home gardeners.

---

## 2. Disease Classes (10 Classes)

| # | Disease | Type | Key Visual Symptoms |
|---|---------|------|---------------------|
| 1 | **Bacterial Spot** | Bacterial | Small, dark brown/black lesions with light centers and yellow halos |
| 2 | **Early Blight** | Fungal | Concentric "bull's-eye" ring pattern, starts on lower leaves |
| 3 | **Late Blight** | Oomycete | Water-soaked greenish-black blotches, white fuzzy growth underneath |
| 4 | **Leaf Mold** | Fungal | Yellow spots on upper surface, olive-green mold on underside |
| 5 | **Septoria Leaf Spot** | Fungal | Small circular spots with gray-white centers and dark edges |
| 6 | **Two-Spotted Spider Mite** | Pest | Stippling/speckling pattern, fine webbing, yellow discoloration |
| 7 | **Target Spot** | Fungal | Brown lesions with concentric rings (smaller than early blight) |
| 8 | **Tomato Yellow Leaf Curl Virus** | Viral | Upward curling, yellowing leaf margins, stunted growth |
| 9 | **Tomato Mosaic Virus** | Viral | Mottled light/dark green pattern, distorted leaves |
| 10 | **Healthy** | — | Uniform green, no lesions or discoloration |

### Challenging Confusions (Hard Pairs)
- **Bacterial Spot ↔ Early Blight** — similar dark spots, especially in early stages
- **Early Blight ↔ Target Spot** — both show concentric rings
- **Septoria Leaf Spot ↔ Bacterial Spot** — small spots with dark edges
- **Late Blight ↔ Leaf Mold** — both cause discoloration with underside symptoms

---

## 3. Datasets for Merging

### Why Merge Multiple Datasets?

The most widely used dataset, PlantVillage, was captured in **controlled lab conditions** (single leaf, uniform white/gray background). Models trained only on PlantVillage achieve 95–99% accuracy in testing but **fail significantly on real-world field images** with complex backgrounds, varying lighting, and occlusion.

**Our strategy: Merge lab + field + internet-sourced datasets** to create a robust, diverse training set.

### Dataset Sources

#### A. PlantVillage — Tomato Subset (Lab/Controlled)
- **Source:** [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Images:** ~14,531 tomato leaf images
- **Classes:** 10 (9 diseases + healthy)
- **Characteristics:** Single leaf on uniform background, consistent lighting
- **Strength:** Large volume, clean labels, well-studied benchmark
- **Weakness:** No real-world complexity; models overfit to lab conditions

#### B. PlantDoc (Real-World / Internet-Scraped)
- **Source:** [PlantDoc Dataset — GitHub](https://github.com/pratikkayal/PlantDoc-Dataset)
- **Images:** ~2,598 total (tomato subset: ~300–500 images)
- **Classes:** 13 plant species, 17 disease types (includes tomato)
- **Characteristics:** Internet-scraped real-world images, complex backgrounds, multiple leaves per image
- **Strength:** Real conditions — lighting variation, occlusion, noise
- **Weakness:** Smaller volume, noisier labels

#### C. Tomato-Village (Real-World Field Dataset)
- **Source:** [Kaggle — Tomato-Village](https://www.kaggle.com/datasets/mamtag/tomato-village)
- **Images:** Field-captured images from real agricultural environments
- **Classes:** Tomato disease classes matching PlantVillage categories
- **Characteristics:** Natural backgrounds, multiple growth stages, in-field conditions
- **Strength:** Closest to actual deployment scenario
- **Weakness:** May require label verification/cleaning

#### D. Mendeley Tomato Leaf Dataset (Lab + Taiwan Field)
- **Source:** [Mendeley Data](https://data.mendeley.com/datasets/ngdgg79rzb/1)
- **Images:** 14,531 (PlantVillage subset) + 4,976 augmented Taiwan field images
- **Classes:** 10 (PlantVillage) + 6 (Taiwan field — includes gray leaf spot, powdery mildew)
- **Characteristics:** Two subsets — lab and field; field images have complex backgrounds
- **Strength:** Includes region-specific diseases; pre-augmented field images

#### E. Bangladesh Tomato Dataset (Real-World, 2024)
- **Source:** [ScienceDirect / Data in Brief](https://www.sciencedirect.com/science/article/pii/S2352340925000599)
- **Images:** 1,028 (482 healthy + 546 diseased)
- **Characteristics:** Captured in Bangladesh fields (Feb 2024), diverse backgrounds, angles, lighting
- **Strength:** Recent, real-world, annotated disease regions
- **Weakness:** Binary labels only (healthy vs. diseased)

### Proposed Merged Dataset Strategy

| Source | Role | Estimated Images |
|--------|------|-----------------|
| PlantVillage (tomato) | Core training volume | ~14,500 |
| PlantDoc (tomato subset) | Real-world diversity | ~400 |
| Tomato-Village | Field validation | ~1,000+ |
| Mendeley (Taiwan field) | Cross-region variety | ~4,976 |
| Own collection (UAE) | Domain-specific fine-tuning | Target: 500+ |

**Total estimated: ~21,000+ images across 10 classes**

### Class Balancing Strategy

The raw merged dataset will be **heavily imbalanced** (Yellow Leaf Curl Virus: ~5,000+ vs. Mosaic Virus: ~373). We'll address this through:

1. **Aggressive augmentation on minority classes** — rotation, flip, brightness, contrast, affine, noise injection
2. **Target: 2,000–3,000 images per class** after augmentation
3. **Class-weighted loss function** as additional safeguard
4. **Focal loss** to focus training on hard-to-classify samples

---

## 4. Custom CNN Architecture (From Scratch)

### Design Philosophy
- **No transfer learning** — learn features from our specific domain data
- **Lightweight** — target <5MB model size for mobile deployment
- **Attention-enhanced** — help the model focus on disease-relevant regions
- **Progressive feature extraction** — small-to-large receptive fields

### Proposed Architecture: TomatoCareNet

```
Input: 224 × 224 × 3 (RGB image)

BLOCK 1 — Low-level features (edges, textures)
├── Conv2D(32, 3×3) → BatchNorm → ReLU
├── Conv2D(32, 3×3) → BatchNorm → ReLU
├── MaxPool(2×2)
└── Dropout(0.25)

BLOCK 2 — Mid-level features (spots, patterns)
├── Conv2D(64, 3×3) → BatchNorm → ReLU
├── Conv2D(64, 3×3) → BatchNorm → ReLU
├── MaxPool(2×2)
└── Dropout(0.25)

BLOCK 3 — High-level features (disease signatures)
├── Conv2D(128, 3×3) → BatchNorm → ReLU
├── Conv2D(128, 3×3) → BatchNorm → ReLU
├── MaxPool(2×2)
└── Dropout(0.3)

BLOCK 4 — Abstract features (disease classification cues)
├── Conv2D(256, 3×3) → BatchNorm → ReLU
├── Conv2D(256, 3×3) → BatchNorm → ReLU
├── MaxPool(2×2)
└── Dropout(0.3)

ATTENTION MODULE — Channel Attention (SE-style)
├── GlobalAveragePooling
├── Dense(256 // 16 = 16) → ReLU
├── Dense(256) → Sigmoid
└── Multiply (reweight feature maps)

CLASSIFIER
├── GlobalAveragePooling2D
├── Dense(512) → BatchNorm → ReLU → Dropout(0.5)
├── Dense(256) → BatchNorm → ReLU → Dropout(0.4)
└── Dense(10) → Softmax

Output: 10 class probabilities
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **3×3 kernels throughout** | Proven most effective; two 3×3 = one 5×5 receptive field with fewer parameters |
| **BatchNorm after every Conv** | Stabilizes training, allows higher learning rates |
| **Progressive channel increase** (32→64→128→256) | Gradually captures more complex features |
| **SE Attention block** | Helps model focus on disease-relevant channels without adding much computation |
| **GlobalAveragePooling (not Flatten)** | Reduces parameters dramatically; acts as structural regularizer |
| **Heavy dropout** (0.25–0.5) | Critical for preventing overfitting on lab images |
| **No pretrained weights** | All features learned from our merged dataset |

### Estimated Model Size
- Parameters: ~2–3 million
- Model size: ~8–12 MB (float32), **~2–3 MB** after int8 quantization

---

## 5. Training Strategy

### Data Split
- **Training:** 70%
- **Validation:** 15%
- **Testing:** 15%

**Important:** Ensure real-world images (PlantDoc, Tomato-Village) are proportionally represented in **all** splits, not concentrated in training only.

### Preprocessing Pipeline
1. **Resize:** All images to 224 × 224
2. **Normalize:** Pixel values to [0, 1]
3. **Color space:** RGB (standard for CNN input)

### Data Augmentation (Training Only)
- Random rotation: ±30°
- Random horizontal flip
- Random vertical flip
- Random brightness: ±20%
- Random contrast: ±20%
- Random zoom: ±15%
- Random shift: ±10% (width and height)
- Gaussian noise injection (σ = 0.01)
- Cutout/Random erasing (simulate occlusion)

### Training Configuration
- **Optimizer:** Adam (lr=1e-3 with cosine decay or ReduceLROnPlateau)
- **Loss:** Focal Loss (γ=2.0) or Weighted Categorical Cross-Entropy
- **Batch size:** 32
- **Epochs:** 100–150 with EarlyStopping (patience=15)
- **Learning rate schedule:** Warmup (5 epochs) → Cosine annealing
- **Callbacks:** ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

---

## 6. Evaluation Metrics

| Metric | Why |
|--------|-----|
| **Accuracy** | Overall correctness |
| **Precision** (per-class) | Among predictions of class X, how many are correct? |
| **Recall** (per-class) | Among actual class X samples, how many were detected? |
| **F1-Score** (per-class + macro) | Harmonic mean — balances precision and recall |
| **Confusion Matrix** | Reveals which diseases are confused with each other |
| **ROC-AUC** (per-class) | Performance across all classification thresholds |
| **Inference Time** | Speed on target device (mobile/edge) |
| **Model Size** | Practical for deployment |

### Real-World Evaluation
- **Cross-dataset testing:** Train on PlantVillage + augmented → Test on PlantDoc/Tomato-Village separately
- **UAE field testing:** Collect images from local gardens → Test generalization
- **Grad-CAM visualization:** Verify model is looking at disease regions, not background artifacts

---

## 7. Explainability (Grad-CAM)

A key differentiator for TomatoCare is **visual explainability**. Using Gradient-weighted Class Activation Mapping (Grad-CAM), we can generate heatmaps showing which regions of the leaf image influenced the model's prediction.

**Benefits:**
- Users can see *why* the model classified a disease
- Helps verify the model isn't relying on background cues (a known problem with PlantVillage-only models)
- Builds trust with non-technical users (home gardeners)
- Strong contribution for the capstone research component

---

## 8. Deployment Pipeline

### Phase 1: Model Training (Python/TensorFlow-Keras)
- Train on merged dataset
- Evaluate and optimize

### Phase 2: Model Conversion
- TensorFlow → TFLite (int8 quantization)
- Target: <3MB model file
- Benchmark: <200ms inference on mid-range smartphone

### Phase 3: Mobile App (Flutter)
- Camera capture or gallery upload
- On-device inference (TFLite)
- Disease info + treatment recommendations
- Grad-CAM visualization overlay
- Offline capability (no internet needed)

### Phase 4: UAE Localization
- Arabic language support
- UAE-specific treatment recommendations
- Integration with local gardening resources

---

## 9. Project Timeline

| Week | Task |
|------|------|
| 1–2 | Dataset collection, merging, cleaning, EDA |
| 3–4 | Data preprocessing, augmentation pipeline, class balancing |
| 5–7 | Custom CNN design, training, hyperparameter tuning |
| 8–9 | Evaluation, Grad-CAM explainability, cross-dataset testing |
| 10–11 | TFLite conversion, mobile app prototype |
| 12 | UAE field testing, documentation, capstone report |

---

## 10. Key References

1. **PlantVillage Dataset** — Hughes & Salathé (2015), arXiv:1511.08060
2. **PlantDoc Dataset** — Singh et al. (2020), ACM IKDD CoDS
3. **Tomato-Village** — Gehlot et al. (2023), Multimedia Systems
4. **Standalone Edge-AI for Tomato Disease** — ScienceDirect (2024) — MobileNetV2 achieves 98.25% on edge devices
5. **RTR_Lite_MobileNet** — ScienceDirect (2025) — Attention-enhanced lightweight model
6. **Hybrid DL + Fuzzy Logic** — Scientific Reports (2026) — ResNet-50 + EfficientNet-B0 + DenseNet-121 ensemble
7. **V²PlantNet** — Scientific Reports (2025) — Modified MobileNet, 389K params, 1.46MB, 98% test accuracy
8. **Frontiers Review (2024)** — Comprehensive review of DL for tomato disease using real field datasets

---

## 11. Novel Contributions of TomatoCare

1. **Custom CNN from scratch** — Unlike most studies relying on transfer learning, we design and train from zero
2. **Multi-source merged dataset** — Combining lab + field + internet images for real-world robustness
3. **UAE-focused deployment** — Tailored for hot, arid climate conditions and local gardening practices
4. **Explainable AI** — Grad-CAM integration for user trust and model validation
5. **Edge-optimized** — Designed for mobile deployment from the start, not as an afterthought
6. **Open dataset contribution** — UAE-collected images contributed back to the research community

---

*This document serves as the research foundation for the TomatoCare capstone project. Next step: Begin coding — data loading, EDA, and baseline model implementation.*