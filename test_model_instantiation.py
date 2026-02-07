
import torch
from src.model import TomatoCareNet, count_parameters

def test_model():
    print("Initializing model...")
    model = TomatoCareNet(num_classes=10)
    print("Model initialized.")
    
    count_parameters(model)
    model.eval()
    
    x = torch.randn(1, 3, 224, 224)
    print(f"Input shape: {x.shape}")
    
    y = model(x)
    print(f"Output shape: {y.shape}")
    assert y.shape == (1, 10)
    print("Forward pass successful!")

if __name__ == "__main__":
    test_model()
