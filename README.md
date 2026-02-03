# 🍅 TomatoCare

> AI-powered tomato disease detection for UAE home gardeners

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

TomatoCare helps home gardeners identify tomato plant diseases using AI. Take a photo, get instant diagnosis with UAE-specific treatment recommendations.

## ✨ Features

- 🔍 **Disease Detection**: Identifies 9 diseases + healthy leaves
- 📱 **Offline-First**: Works without internet
- 🔒 **Privacy-Focused**: On-device processing
- 🌴 **UAE-Specific**: Tailored treatment advice

## 🦠 Supported Diseases

| # | Disease | Cause |
|---|---------|-------|
| 1 | Bacterial Spot | *Xanthomonas* bacteria |
| 2 | Early Blight | *Alternaria solani* fungus |
| 3 | Late Blight | *Phytophthora infestans* |
| 4 | Leaf Mold | *Passalora fulva* fungus |
| 5 | Septoria Leaf Spot | *Septoria lycopersici* fungus |
| 6 | Spider Mites | *Tetranychus urticae* pest |
| 7 | Target Spot | *Corynespora cassiicola* fungus |
| 8 | Yellow Leaf Curl Virus | Begomovirus |
| 9 | Mosaic Virus | Tobamovirus |
| 10 | Healthy | - |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 11.8+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Usage

```bash
# Train the model
python src/train.py

# Quick test (2 epochs)
python src/train.py --quick-test

# Evaluate on test set
python src/evaluate.py

# Predict single image
python src/predict.py path/to/leaf.jpg

# Export for mobile
python src/export.py
```

## 📁 Project Structure

```
TomatoCare/
├── configs/
│   └── config.py           # Centralized settings
├── data/
│   ├── tomato/             # Dataset (train/val/test)
│   └── disease_info.json   # UAE treatment database
├── src/
│   ├── data/
│   │   ├── dataset.py      # DataLoaders
│   │   └── transforms.py   # Augmentations
│   ├── models/
│   │   └── classifier.py   # MobileNetV2 architecture
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Metrics & plots
│   ├── predict.py          # Single image inference
│   └── export.py           # Mobile export
├── outputs/
│   ├── exploration/        # Dataset visualizations
│   ├── training/           # Checkpoints & history
│   ├── evaluation/         # Metrics & confusion matrix
│   └── mobile/             # Exported models
├── docs/
│   └── research.md         # Research & references
└── app/                    # Future mobile app
```

## 🎯 Model

**MobileNetV2** with transfer learning:
- Parameters: ~3.4M
- Input: 224×224 RGB
- Output: 10-class probabilities
- Target accuracy: >90%

## 📝 License

MIT License

---

Made with ❤️ for UAE home gardeners
