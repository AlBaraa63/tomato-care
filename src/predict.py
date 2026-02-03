"""
predict.py - Single Image Prediction

Predict disease from a single tomato leaf image.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse

import torch
from PIL import Image

from configs.config import (
    DEVICE, BEST_MODEL_PATH, DISEASE_INFO_PATH, NUM_CLASSES,
    get_device_info
)
from src.data.transforms import get_inference_transforms
from src.models.classifier import get_model


def load_model(checkpoint_path=None, device=None):
    """Load trained model."""
    checkpoint_path = checkpoint_path or BEST_MODEL_PATH
    device = device or DEVICE
    
    model = get_model(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_disease_info():
    """Load disease information database."""
    if os.path.exists(DISEASE_INFO_PATH):
        with open(DISEASE_INFO_PATH, 'r') as f:
            return json.load(f)
    return {}


def predict(image_path, model=None, device=None, top_k=3):
    """
    Predict disease from image.
    
    Args:
        image_path: Path to leaf image
        model: Loaded model (optional, will load if None)
        device: Device to use
        top_k: Number of top predictions to return
    
    Returns:
        dict: Prediction results with class, confidence, and info
    """
    device = device or DEVICE
    
    # Load model if not provided
    if model is None:
        model = load_model(device=device)
    
    # Load and transform image
    transform = get_inference_transforms()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = probs.topk(top_k)
    
    # Get class names
    class_names = [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]
    
    # Load disease info
    disease_info = load_disease_info()
    
    # Build results
    results = {
        'image': image_path,
        'predictions': []
    }
    
    for i in range(top_k):
        class_idx = top_indices[0][i].item()
        class_name = class_names[class_idx]
        confidence = top_probs[0][i].item() * 100
        
        pred = {
            'class': class_name,
            'display_name': class_name.replace('Tomato___', '').replace('_', ' '),
            'confidence': confidence,
            'rank': i + 1
        }
        
        # Add disease info if available
        if class_name in disease_info:
            info = disease_info[class_name]
            pred['info'] = {
                'name': info.get('name'),
                'symptoms': info.get('symptoms', []),
                'treatment': info.get('treatment', {}),
                'prevention': info.get('prevention', []),
                'uae_notes': info.get('uae_notes')
            }
        
        results['predictions'].append(pred)
    
    return results


def print_prediction(results):
    """Print prediction results."""
    print("\n" + "=" * 60)
    print("🍅 TOMATOCARE PREDICTION")
    print("=" * 60)
    print(f"Image: {results['image']}")
    print("-" * 60)
    
    for pred in results['predictions']:
        icon = "🏆" if pred['rank'] == 1 else "  "
        print(f"{icon} #{pred['rank']}: {pred['display_name']}")
        print(f"      Confidence: {pred['confidence']:.1f}%")
        
        if pred['rank'] == 1 and 'info' in pred:
            info = pred['info']
            print(f"\n📋 Disease Info:")
            if info.get('symptoms'):
                print(f"   Symptoms: {info['symptoms'][0]}")
            if info.get('treatment', {}).get('organic'):
                print(f"   Treatment: {info['treatment']['organic'][0]}")
            if info.get('uae_notes'):
                print(f"   UAE Note: {info['uae_notes'][:80]}...")
        print()
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Predict tomato disease from image')
    parser.add_argument('image', type=str, help='Path to leaf image')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint')
    parser.add_argument('--top-k', type=int, default=3, help='Top K predictions')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    get_device_info()
    
    results = predict(
        args.image,
        model=load_model(args.checkpoint) if args.checkpoint else None,
        top_k=args.top_k
    )
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_prediction(results)


if __name__ == "__main__":
    main()
