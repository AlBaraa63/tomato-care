"""
export.py - Export Model for Mobile Deployment

Exports to TorchScript, Quantized TorchScript, and ONNX.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse

import torch
import torch.nn as nn

from configs.config import (
    BEST_MODEL_PATH, MOBILE_DIR, NUM_CLASSES, IMAGE_SIZE,
    get_device_info, ensure_dirs
)
from src.models.classifier import get_model


INPUT_SIZE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def load_model(checkpoint_path):
    """Load trained model."""
    model = get_model(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded: {checkpoint_path}")
    print(f"   Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    return model, checkpoint.get('val_acc', 0)


def export_torchscript(model, save_path):
    """Export to TorchScript."""
    print(f"\n📦 Exporting TorchScript...")
    dummy = torch.randn(INPUT_SIZE)
    traced = torch.jit.trace(model, dummy)
    traced = torch.jit.optimize_for_inference(traced)
    traced.save(save_path)
    
    size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"   ✅ {save_path} ({size:.2f} MB)")
    return size


def export_quantized(model, save_path):
    """Export quantized TorchScript."""
    print(f"\n📦 Exporting Quantized TorchScript...")
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    dummy = torch.randn(INPUT_SIZE)
    traced = torch.jit.trace(quantized, dummy)
    traced.save(save_path)
    
    size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"   ✅ {save_path} ({size:.2f} MB)")
    return size


def export_onnx(model, save_path):
    """Export to ONNX."""
    print(f"\n📦 Exporting ONNX...")
    dummy = torch.randn(INPUT_SIZE)
    
    torch.onnx.export(
        model, dummy, save_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    
    size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"   ✅ {save_path} ({size:.2f} MB)")
    
    try:
        import onnx
        onnx.checker.check_model(onnx.load(save_path))
        print(f"   ✅ ONNX verification passed")
    except ImportError:
        print(f"   ⚠️ Install 'onnx' to verify")
    except Exception as e:
        print(f"   ⚠️ ONNX warning: {e}")
    
    return size


def benchmark(model, device='cpu', runs=100):
    """Benchmark inference speed."""
    print(f"\n⏱️ Benchmarking ({runs} runs, {device})...")
    model = model.to(device)
    dummy = torch.randn(INPUT_SIZE).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            model(dummy)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    avg_ms = (time.time() - start) / runs * 1000
    print(f"   Avg: {avg_ms:.2f} ms | Throughput: {1000/avg_ms:.1f} img/s")
    return avg_ms


def create_model_card(save_dir, sizes, inference_ms, val_acc):
    """Create model card."""
    card = f"""# TomatoCare Model

## Info
- Architecture: MobileNetV2
- Classes: 10
- Input: 224x224 RGB
- Val Accuracy: {val_acc:.2f}%

## Exports
| Format | Size |
|--------|------|
| TorchScript | {sizes.get('pt', 0):.2f} MB |
| Quantized | {sizes.get('quantized', 0):.2f} MB |
| ONNX | {sizes.get('onnx', 0):.2f} MB |

## Performance
- CPU Inference: {inference_ms:.2f} ms

## Classes
0. Bacterial Spot
1. Early Blight
2. Late Blight
3. Leaf Mold
4. Septoria Leaf Spot
5. Spider Mites
6. Target Spot
7. Yellow Leaf Curl Virus
8. Mosaic Virus
9. Healthy
"""
    path = os.path.join(save_dir, 'MODEL_CARD.md')
    with open(path, 'w') as f:
        f.write(card)
    print(f"\n📄 Model card: {path}")


def export(checkpoint_path=None):
    """Run full export pipeline."""
    ensure_dirs()
    checkpoint_path = checkpoint_path or BEST_MODEL_PATH
    
    print("=" * 60)
    print("📱 TOMATOCARE MOBILE EXPORT")
    print("=" * 60)
    
    model, val_acc = load_model(checkpoint_path)
    sizes = {}
    
    sizes['pt'] = export_torchscript(model, os.path.join(MOBILE_DIR, 'tomato_classifier.pt'))
    sizes['quantized'] = export_quantized(model, os.path.join(MOBILE_DIR, 'tomato_classifier_quantized.pt'))
    sizes['onnx'] = export_onnx(model, os.path.join(MOBILE_DIR, 'tomato_classifier.onnx'))
    
    inference_ms = benchmark(model, 'cpu')
    if torch.cuda.is_available():
        benchmark(model, 'cuda')
    
    create_model_card(MOBILE_DIR, sizes, inference_ms, val_acc)
    
    print("\n" + "=" * 60)
    print("✅ EXPORT COMPLETE!")
    print("=" * 60)
    print(f"📁 Output: {MOBILE_DIR}")
    print(f"\n📊 Sizes:")
    print(f"   TorchScript: {sizes['pt']:.2f} MB")
    print(f"   Quantized:   {sizes['quantized']:.2f} MB")
    print(f"   ONNX:        {sizes['onnx']:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    export(args.checkpoint)
