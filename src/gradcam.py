"""
Grad-CAM: Visual explanations for TomatoCareNet predictions.

Shows which regions of the leaf image the model focuses on when
making a disease classification, useful for the capstone report.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into a target conv layer, captures activations on the forward pass
    and gradients on the backward pass, then produces a heatmap showing
    which spatial regions contributed most to a given class prediction.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor [1, 3, H, W] on device
            target_class: class index to explain (None = predicted class)

        Returns:
            cam: normalized heatmap as numpy array [H, W] in [0, 1]
            target_class: the class that was explained
            output: raw model logits
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        # Global average pool the gradients to get per-channel importance
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy(), target_class, output


def preprocess_image(image_path):
    """Load and preprocess a single image for GradCAM."""
    transform = A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=image)["image"].unsqueeze(0).to(Config.DEVICE)
    return image, tensor


def visualize_gradcam(model, image_path, save_path=None):
    """
    Generate and display a Grad-CAM visualization for one image.

    Targets the last conv layer in block4 (deepest features).
    """
    # Hook into the last conv block's second BN (right before residual add)
    target_layer = model.block4.conv2[-1]
    grad_cam = GradCAM(model, target_layer)

    original_image, input_tensor = preprocess_image(image_path)
    heatmap, pred_class, logits = grad_cam.generate(input_tensor)

    probs = torch.softmax(logits, dim=1).squeeze().cpu().detach().numpy()
    pred_name = Config.CLASS_NAMES[pred_class]
    confidence = probs[pred_class] * 100

    # Resize heatmap to match original image
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (original_image.shape[1], original_image.shape[0]),
            Image.BILINEAR
        )
    ) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    # Overlay
    overlay = original_image.astype(np.float32) / 255.0
    heatmap_color = plt.cm.jet(heatmap_resized)[:, :, :3]
    blended = 0.6 * overlay + 0.4 * heatmap_color
    blended = np.clip(blended, 0, 1)
    axes[2].imshow(blended)
    axes[2].set_title(f"Prediction: {pred_name} ({confidence:.1f}%)")
    axes[2].axis("off")

    plt.suptitle("TomatoCareNet — Grad-CAM Explanation", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM saved to: {save_path}")
    plt.show()
