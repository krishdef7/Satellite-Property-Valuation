"""
PyTorch Dataset Classes for Property Valuation.

This module provides Dataset classes for:
- Tabular-only data
- Image-only data (Z16, Z17, Z18)
- Multimodal data (tabular + multi-scale images)

Supports data augmentation for training and proper transforms for validation.

Author: Competition Submission
Date: December 2024
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    IMAGES_ZOOM_16_DIR, IMAGES_ZOOM_17_DIR, IMAGES_ZOOM_18_DIR, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, AUGMENTATION_ENABLED,
    IMAGE_TRAIN_TRANSFORMS
)


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular features only.
    
    Used for LightGBM baseline comparison with neural networks.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        ids: Optional[np.ndarray] = None
    ):
        """
        Initialize tabular dataset.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            targets: Target values of shape (n_samples,), None for test set
            ids: Property IDs for tracking
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.ids = ids
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {'features': self.features[idx]}
        
        if self.targets is not None:
            item['target'] = self.targets[idx]
        
        if self.ids is not None:
            item['id'] = self.ids[idx]
            
        return item


class ImageDataset(Dataset):
    """
    PyTorch Dataset for satellite images only.
    
    Supports three-scale images (zoom 16, 17, and 18).
    """
    
    def __init__(
        self,
        property_ids: List[str],
        targets: Optional[np.ndarray] = None,
        images_dir_z16: Path = IMAGES_ZOOM_16_DIR,
        images_dir_z17: Path = IMAGES_ZOOM_17_DIR,
        images_dir_z18: Path = IMAGES_ZOOM_18_DIR,
        transform: Optional[Callable] = None,
        augment: bool = False
    ):
        """
        Initialize image dataset.
        
        Args:
            property_ids: List of property IDs
            targets: Target values, None for test set
            images_dir_z16: Directory for zoom 16 images
            images_dir_z17: Directory for zoom 17 images
            images_dir_z18: Directory for zoom 18 images
            transform: Torchvision transforms to apply
            augment: Whether to apply data augmentation
        """
        self.property_ids = property_ids
        self.targets = targets
        self.images_dir_z16 = Path(images_dir_z16)
        self.images_dir_z17 = Path(images_dir_z17)
        self.images_dir_z18 = Path(images_dir_z18)
        self.augment = augment
        
        # Default transform
        if transform is None:
            self.transform = self._get_default_transform(augment)
        else:
            self.transform = transform
            
    def _get_default_transform(self, augment: bool) -> T.Compose:
        """Get default image transforms."""
        transforms = []
        
        if augment and AUGMENTATION_ENABLED:
            transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(IMAGE_TRAIN_TRANSFORMS['rotation']),
                T.ColorJitter(
                    brightness=IMAGE_TRAIN_TRANSFORMS['color_jitter']['brightness'],
                    contrast=IMAGE_TRAIN_TRANSFORMS['color_jitter']['contrast'],
                    saturation=IMAGE_TRAIN_TRANSFORMS['color_jitter']['saturation'],
                    hue=IMAGE_TRAIN_TRANSFORMS['color_jitter']['hue']
                ),
            ])
        
        transforms.extend([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        return T.Compose(transforms)
    
    def _load_image(self, path: Path) -> Optional[Image.Image]:
        """Load image from path, return None if not found."""
        if path.exists():
            try:
                return Image.open(path).convert('RGB')
            except Exception:
                return None
        return None
    
    def _get_placeholder_image(self) -> Image.Image:
        """Create a placeholder image for missing data."""
        # Gray placeholder
        return Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
    
    def __len__(self) -> int:
        return len(self.property_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prop_id = str(self.property_ids[idx])
        
        # Load zoom 16 image (regional context)
        z16_path = self.images_dir_z16 / f"{prop_id}_z16.jpg"
        img_z16 = self._load_image(z16_path)
        if img_z16 is None:
            img_z16 = self._get_placeholder_image()
        
        # Load zoom 17 image (neighborhood context)
        z17_path = self.images_dir_z17 / f"{prop_id}_z17.jpg"
        img_z17 = self._load_image(z17_path)
        if img_z17 is None:
            img_z17 = self._get_placeholder_image()
        
        # Load zoom 18 image (property detail)
        z18_path = self.images_dir_z18 / f"{prop_id}_z18.jpg"
        img_z18 = self._load_image(z18_path)
        if img_z18 is None:
            img_z18 = self._get_placeholder_image()
        
        # Apply transforms
        img_z16 = self.transform(img_z16)
        img_z17 = self.transform(img_z17)
        img_z18 = self.transform(img_z18)
        
        item = {
            'img_z16': img_z16,
            'img_z17': img_z17,
            'img_z18': img_z18,
            'id': prop_id
        }
        
        if self.targets is not None:
            item['target'] = torch.FloatTensor([self.targets[idx]])
        
        return item


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal data (tabular + three-scale images).
    
    Combines tabular features with satellite imagery at three zoom levels
    for the fusion models.
    """
    
    def __init__(
        self,
        property_ids: List[str],
        tabular_features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        baseline_predictions: Optional[np.ndarray] = None,
        images_dir_z16: Path = IMAGES_ZOOM_16_DIR,
        images_dir_z17: Path = IMAGES_ZOOM_17_DIR,
        images_dir_z18: Path = IMAGES_ZOOM_18_DIR,
        transform: Optional[Callable] = None,
        augment: bool = False
    ):
        """
        Initialize multimodal dataset.
        
        Args:
            property_ids: List of property IDs
            tabular_features: Tabular feature matrix
            targets: Target values, None for test set
            baseline_predictions: LightGBM baseline predictions for residual learning
            images_dir_z16: Directory for zoom 16 images
            images_dir_z17: Directory for zoom 17 images
            images_dir_z18: Directory for zoom 18 images
            transform: Torchvision transforms
            augment: Whether to apply data augmentation
        """
        self.property_ids = [str(pid) for pid in property_ids]
        self.tabular_features = torch.FloatTensor(tabular_features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.baseline_preds = torch.FloatTensor(baseline_predictions) if baseline_predictions is not None else None
        
        self.images_dir_z16 = Path(images_dir_z16)
        self.images_dir_z17 = Path(images_dir_z17)
        self.images_dir_z18 = Path(images_dir_z18)
        self.augment = augment
        
        if transform is None:
            self.transform = self._get_default_transform(augment)
        else:
            self.transform = transform
    
    def _get_default_transform(self, augment: bool) -> T.Compose:
        """Get default image transforms."""
        transforms = []
        
        if augment and AUGMENTATION_ENABLED:
            transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(IMAGE_TRAIN_TRANSFORMS['rotation']),
                T.ColorJitter(
                    brightness=IMAGE_TRAIN_TRANSFORMS['color_jitter']['brightness'],
                    contrast=IMAGE_TRAIN_TRANSFORMS['color_jitter']['contrast'],
                    saturation=IMAGE_TRAIN_TRANSFORMS['color_jitter']['saturation'],
                    hue=IMAGE_TRAIN_TRANSFORMS['color_jitter']['hue']
                ),
            ])
        
        transforms.extend([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        return T.Compose(transforms)
    
    def _load_image(self, path: Path) -> Optional[Image.Image]:
        """Load image from path."""
        if path.exists():
            try:
                return Image.open(path).convert('RGB')
            except Exception:
                return None
        return None
    
    def _get_placeholder_image(self) -> Image.Image:
        """Create placeholder for missing images."""
        return Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
    
    def __len__(self) -> int:
        return len(self.property_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prop_id = self.property_ids[idx]
        
        # Load zoom 16 image (regional context)
        z16_path = self.images_dir_z16 / f"{prop_id}_z16.jpg"
        img_z16 = self._load_image(z16_path)
        if img_z16 is None:
            img_z16 = self._get_placeholder_image()
        
        # Load zoom 17 image (neighborhood context)
        z17_path = self.images_dir_z17 / f"{prop_id}_z17.jpg"
        img_z17 = self._load_image(z17_path)
        if img_z17 is None:
            img_z17 = self._get_placeholder_image()
        
        # Load zoom 18 image (property detail)
        z18_path = self.images_dir_z18 / f"{prop_id}_z18.jpg"
        img_z18 = self._load_image(z18_path)
        if img_z18 is None:
            img_z18 = self._get_placeholder_image()
        
        # Apply transforms
        img_z16 = self.transform(img_z16)
        img_z17 = self.transform(img_z17)
        img_z18 = self.transform(img_z18)
        
        item = {
            'img_z16': img_z16,
            'img_z17': img_z17,
            'img_z18': img_z18,
            'tabular': self.tabular_features[idx],
            'id': prop_id
        }
        
        if self.targets is not None:
            item['target'] = self.targets[idx]
        
        if self.baseline_preds is not None:
            item['baseline_pred'] = self.baseline_preds[idx]
        
        return item


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_image_availability(
    property_ids: List[str],
    images_dir_z16: Path = IMAGES_ZOOM_16_DIR,
    images_dir_z17: Path = IMAGES_ZOOM_17_DIR,
    images_dir_z18: Path = IMAGES_ZOOM_18_DIR
) -> Dict[str, bool]:
    """
    Check which properties have all three zoom level images available.
    
    Args:
        property_ids: List of property IDs to check
        images_dir_z16: Zoom 16 images directory
        images_dir_z17: Zoom 17 images directory
        images_dir_z18: Zoom 18 images directory
        
    Returns:
        Dictionary mapping property_id -> has_all_images
    """
    availability = {}
    
    for prop_id in property_ids:
        prop_id = str(prop_id)
        z16_exists = (images_dir_z16 / f"{prop_id}_z16.jpg").exists()
        z17_exists = (images_dir_z17 / f"{prop_id}_z17.jpg").exists()
        z18_exists = (images_dir_z18 / f"{prop_id}_z18.jpg").exists()
        availability[prop_id] = z16_exists and z17_exists and z18_exists
    
    return availability


if __name__ == "__main__":
    """Test dataset classes."""
    import pandas as pd
    from config import TRAIN_FILE
    
    print("="*60)
    print("Testing Dataset Classes (3-Scale)")
    print("="*60)
    
    # Load sample data
    train_df = pd.read_csv(TRAIN_FILE)
    sample_df = train_df.head(100)
    
    print(f"\nSample size: {len(sample_df)}")
    
    # Check image availability
    availability = get_image_availability(sample_df['id'].tolist())
    has_images = sum(availability.values())
    print(f"Properties with all 3 zoom images: {has_images}/{len(sample_df)}")
    
    # Test tabular dataset
    print("\n--- Tabular Dataset ---")
    features = sample_df[['sqft_living', 'bedrooms', 'bathrooms', 'grade']].values
    targets = sample_df['price'].values
    
    tab_dataset = TabularDataset(features, targets, sample_df['id'].values)
    print(f"Dataset length: {len(tab_dataset)}")
    
    sample = tab_dataset[0]
    print(f"Sample features shape: {sample['features'].shape}")
    print(f"Sample target: {sample['target']:.2f}")
    
    # Test multimodal dataset
    print("\n--- Multimodal Dataset (3-Scale) ---")
    mm_dataset = MultimodalDataset(
        property_ids=sample_df['id'].tolist(),
        tabular_features=features,
        targets=targets,
        augment=False
    )
    
    print(f"Dataset length: {len(mm_dataset)}")
    
    sample = mm_dataset[0]
    print(f"Image z16 shape: {sample['img_z16'].shape}")
    print(f"Image z17 shape: {sample['img_z17'].shape}")
    print(f"Image z18 shape: {sample['img_z18'].shape}")
    print(f"Tabular shape: {sample['tabular'].shape}")
    
    # Test data loaders
    print("\n--- Data Loaders ---")
    train_loader, val_loader = create_data_loaders(
        mm_dataset, mm_dataset,
        batch_size=16,
        num_workers=0
    )
    
    batch = next(iter(train_loader))
    print(f"Batch img_z16 shape: {batch['img_z16'].shape}")
    print(f"Batch img_z17 shape: {batch['img_z17'].shape}")
    print(f"Batch img_z18 shape: {batch['img_z18'].shape}")
    print(f"Batch tabular shape: {batch['tabular'].shape}")
    print(f"Batch target shape: {batch['target'].shape}")
    
    print("\n✅ All dataset tests passed!")