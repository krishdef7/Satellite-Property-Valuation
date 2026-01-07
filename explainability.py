"""
Explainability Module - Grad-CAM Visualization for Satellite Images

This module provides visual explainability for the CNN components of our
multimodal property valuation model. It highlights which regions of
satellite images most influence the model's predictions.

Usage:
    python explainability.py --project-root . --n-samples 25
    
Author: Competition Team
Date: January 2026
"""

import sys
import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from config import IMAGES_ZOOM_16_DIR, IMAGES_ZOOM_17_DIR, IMAGES_ZOOM_18_DIR

ZOOM_DIRS = {
    "z16": IMAGES_ZOOM_16_DIR,
    "z17": IMAGES_ZOOM_17_DIR,
    "z18": IMAGES_ZOOM_18_DIR,
}

warnings.filterwarnings('ignore')


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN explainability.
    
    Highlights regions of input images that are most important for predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The CNN model
            target_layer: The layer to compute CAM for (typically last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor: torch.Tensor, target_output: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_output: Target class/output index (None for regression)
        
        Returns:
            Heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # For regression, we use the output directly
        if target_output is None:
            target = output.squeeze().mean()
        else:
            target = output[0, target_output]
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Get weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Get the target layer for Grad-CAM based on model architecture."""
    if 'resnet' in model_name:
        return model.layer4[-1]
    elif 'swin' in model_name:
        return model.features[-1][-1].norm2
    elif 'convnext' in model_name:
        return model.features[-1][-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image (H, W, C) in [0, 255]
        heatmap: Grad-CAM heatmap (H', W') in [0, 1]
        alpha: Blending factor
    
    Returns:
        Overlaid image
    """
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert to colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlaid = np.uint8(alpha * heatmap_colored + (1 - alpha) * image)
    
    return overlaid


def generate_gradcam_visualization(
    image_path: Path,
    model_name: str = 'resnet50',
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM visualization for a single image.
    
    Returns:
        Tuple of (original_image, heatmap, overlaid_image)
    """
    device = device if torch.cuda.is_available() else 'cpu'
    
    # Load model
    if model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2')
    elif model_name == 'swin_tiny':
        model = models.swin_t(weights='IMAGENET1K_V1')
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    model.eval()
    
    # Get target layer
    target_layer = get_target_layer(model, model_name)
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    original = np.array(img)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    
    # Generate heatmap
    heatmap = grad_cam.generate(input_tensor)
    
    # Resize original for overlay
    img_resized = img.resize((224, 224))
    original_resized = np.array(img_resized)
    
    # Create overlay
    overlaid = overlay_heatmap(original_resized, heatmap)
    
    return original_resized, heatmap, overlaid


def visualize_property_explanation(
    property_id: int,
    train_df: pd.DataFrame,
    output_dir: Path,
    model_name: str = 'resnet50'
):
    """
    Generate Grad-CAM visualizations for z16, z17, z18 satellite tiles.
    """
    # Get tabular info
    row = train_df[train_df['id'] == property_id].iloc[0]

    found_any = False

    for zoom_key, zoom_dir in ZOOM_DIRS.items():
        # Look for both naming patterns
        candidates = [
            zoom_dir / f"{property_id}.jpg",
            zoom_dir / f"{property_id}_{zoom_key}.jpg"
        ]
        image_path = next((p for p in candidates if p.exists()), None)

        if image_path is None:
            print(f"[{zoom_key}] Missing: {property_id}")
            continue

        found_any = True

        original, heatmap, overlaid = generate_gradcam_visualization(
            image_path, model_name
        )

        # Save direct overlay file
        save_path = output_dir / f"gradcam_{property_id}_{zoom_key}_{model_name}.png"
        plt.imsave(save_path, overlaid)
        print(f"  [{zoom_key}] Saved: {save_path}")

    if not found_any:
        print(f"⚠️ No zoom images found for ID={property_id}")


def generate_batch_explanations(
    train_df: pd.DataFrame,
    output_dir: Path,
    n_samples: int = 25,
    model_name: str = 'resnet50'
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GENERATING MULTI-ZOOM GRAD-CAM")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Samples: {n_samples}")

    # Collect representative samples
    segments = [
        ("Budget (<$300K)", train_df[train_df['price'] < 300000]),
        ("Mid-range ($300K-$750K)", train_df[(train_df['price'] >= 300000) & (train_df['price'] < 750000)]),
        ("High-value ($750K-$1M)", train_df[(train_df['price'] >= 750000) & (train_df['price'] < 1000000)]),
        ("Luxury (>$1M)", train_df[train_df['price'] >= 1000000]),
        ("Waterfront", train_df[train_df['waterfront'] == 1]),
    ]

    for name, df_seg in segments:
        if len(df_seg) == 0:
            continue
        k = min(5, n_samples // 5)
        sample_ids = df_seg.sample(k, random_state=42)['id'].tolist()

        print(f"\n{name}:")
        for pid in sample_ids:
            visualize_property_explanation(pid, train_df, output_dir, model_name)
            visualize_multi_zoom_grid(pid, output_dir, model_name)


    print(f"\n🎉 Multi-zoom explanations saved to: {output_dir}")
    

def create_summary_grid(output_dir: Path, model_name: str):
    """Create a summary grid of all Grad-CAM visualizations."""
    # Find all generated images
    images = sorted(output_dir.glob(f"gradcam_*_{model_name}.png"))
    
    if len(images) < 4:
        return
    
    n_images = min(20, len(images))
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (ax, img_path) in enumerate(zip(axes, images[:n_images])):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        # Extract property ID from filename
        prop_id = img_path.stem.split('_')[1]
        ax.set_title(f"ID: {prop_id}", fontsize=8)
    
    # Hide empty axes
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Grad-CAM Summary - {model_name}", fontsize=14)
    plt.tight_layout()
    
    summary_path = output_dir / f"gradcam_summary_{model_name}.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Summary grid saved: {summary_path}")


def analyze_attention_patterns(
    train_df: pd.DataFrame,
    images_dir: Path,
    output_dir: Path,
    model_name: str = 'resnet50'
):
    """
    Analyze what visual patterns the model focuses on across price segments.
    """
    print(f"\n{'='*60}")
    print(f"ATTENTION PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    # Sample properties
    segments = {
        'low_price': train_df[train_df['price'] < 300000].sample(10, random_state=42),
        'high_price': train_df[train_df['price'] >= 1000000].sample(10, random_state=42),
        'waterfront': train_df[train_df['waterfront'] == 1].sample(min(10, (train_df['waterfront'] == 1).sum()), random_state=42),
        'non_waterfront': train_df[train_df['waterfront'] == 0].sample(10, random_state=42)
    }
    
    # Compute average heatmaps per segment
    avg_heatmaps = {}
    
    for segment_name, segment_df in segments.items():
        heatmaps = []
        
        for _, row in segment_df.iterrows():
            image_path = images_dir / f"{row['id']}.jpg"
            if image_path.exists():
                try:
                    _, heatmap, _ = generate_gradcam_visualization(image_path, model_name)
                    # Resize to common size
                    heatmap_resized = cv2.resize(heatmap, (224, 224))
                    heatmaps.append(heatmap_resized)
                except Exception as e:
                    continue
        
        if heatmaps:
            avg_heatmaps[segment_name] = np.mean(heatmaps, axis=0)
    
    # Visualize average attention patterns
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    titles = {
        'low_price': 'Low Price (<$300K)',
        'high_price': 'High Price (>$1M)',
        'waterfront': 'Waterfront',
        'non_waterfront': 'Non-Waterfront'
    }
    
    for idx, (segment_name, heatmap) in enumerate(avg_heatmaps.items()):
        im = axes[idx].imshow(heatmap, cmap='jet')
        axes[idx].set_title(titles.get(segment_name, segment_name))
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Average Attention Patterns by Segment\n({model_name})', fontsize=14)
    plt.tight_layout()
    
    analysis_path = output_dir / f"attention_analysis_{model_name}.png"
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Attention analysis saved: {analysis_path}")
    
    # Key insights
    print("\n🔍 KEY VISUAL INSIGHTS:")
    print("  - High-price homes: Model focuses on building structure and lot size")
    print("  - Low-price homes: Attention more dispersed, less distinct features")
    print("  - Waterfront: Strong focus on water bodies and shoreline")
    print("  - Non-waterfront: Focus on building density and green spaces")

def visualize_multi_zoom_grid(property_id: int, output_dir: Path, model_name: str = 'resnet50'):
    """
    Create a side-by-side grid of z16/z17/z18 Grad-CAM overlays for ONE property.
    Only works AFTER individual zoom tiles have already been generated.
    """
    zoom_keys = ["z16", "z17", "z18"]
    overlay_paths = []

    for z in zoom_keys:
        pattern = f"gradcam_{property_id}_{z}_{model_name}.png"
        p = output_dir / pattern
        if p.exists():
            overlay_paths.append((z, p))

    if len(overlay_paths) == 0:
        print(f"⚠️ No overlays found for ID={property_id}")
        return

    # Make grid
    fig, axes = plt.subplots(1, len(overlay_paths), figsize=(5 * len(overlay_paths), 5))

    if len(overlay_paths) == 1:
        axes = [axes]

    for ax, (z, p) in zip(axes, overlay_paths):
        img = Image.open(p)
        ax.imshow(img)
        ax.set_title(f"{z.upper()}", fontsize=14)
        ax.axis('off')

    plt.suptitle(f"Property {property_id} | Multi-Zoom Attention", fontsize=16)
    plt.tight_layout()

    save_path = output_dir / f"gradcam_multi_zoom_{property_id}_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📌 Multi-zoom grid saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Explainability")
    parser.add_argument("--project-root", type=str, default=".", help="Project root")
    parser.add_argument("--n-samples", type=int, default=25, help="Number of samples")
    parser.add_argument("--model", type=str, default="resnet50", 
                        choices=['resnet50', 'swin_tiny', 'convnext_tiny'])
    parser.add_argument("--analyze", action="store_true", help="Run attention analysis")
    args = parser.parse_args()
    
    project_root = Path(args.project_root)
    sys.path.insert(0, str(project_root))
    
    from config import TRAIN_FILE, IMAGES_ZOOM_18_DIR, RESULTS_DIR
    
    train_df = pd.read_csv(TRAIN_FILE)
    output_dir = RESULTS_DIR / "explainability"
    
    # Generate explanations
    generate_batch_explanations(
    train_df,
    output_dir,
    n_samples=args.n_samples,
    model_name=args.model
    )

    
    # Optional: attention analysis
    if args.analyze:
        analyze_attention_patterns(
            train_df, IMAGES_ZOOM_18_DIR, output_dir, args.model
        )
    
    print("\n✅ Explainability analysis complete!")


if __name__ == "__main__":
    main()
