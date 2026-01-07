"""
Configuration file for Satellite Imagery-Based Property Valuation.

This file contains all hyperparameters, paths, and settings for the project.
Centralizing configuration ensures reproducibility and easy experimentation.

Author: Competition Submission
Date: December 2024
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"
IMAGES_ZOOM_16_DIR = IMAGES_DIR / "zoom_16"  # NEW: Wide area context
IMAGES_ZOOM_17_DIR = IMAGES_DIR / "zoom_17"
IMAGES_ZOOM_18_DIR = IMAGES_DIR / "zoom_18"

# Model and results directories
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
REPORT_DIR = PROJECT_ROOT / "report"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGES_ZOOM_16_DIR,
                 IMAGES_ZOOM_17_DIR, IMAGES_ZOOM_18_DIR, MODELS_DIR, 
                 RESULTS_DIR, VISUALIZATIONS_DIR, REPORT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA FILES
# =============================================================================

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"

# Processed data files
TRAIN_PROCESSED_FILE = PROCESSED_DATA_DIR / "train_processed.parquet"
TEST_PROCESSED_FILE = PROCESSED_DATA_DIR / "test_processed.parquet"

# =============================================================================
# API CONFIGURATION - GOOGLE MAPS
# =============================================================================

# Set your Google Maps API key as environment variable:
# export GOOGLE_MAPS_API_KEY="your_key_here"
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# Image settings
IMAGE_SIZE = 512  # 512x512 pixels
ZOOM_LEVELS = [16, 17, 18]  # Regional (16) + Neighborhood (17) + Property detail (18)
IMAGE_FORMAT = "jpg"
MAP_TYPE = "satellite"  # Google Maps satellite imagery

# Zoom level descriptions for documentation
ZOOM_LEVEL_INFO = {
    16: "Regional context - ~2.4km coverage, shows district/area patterns",
    17: "Neighborhood context - ~1.2km coverage, shows street-level patterns",
    18: "Property detail - ~600m coverage, shows building-level features"
}

# Google Maps Static API endpoint
GOOGLE_MAPS_BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"

# Rate limiting - Google allows 25,000 free requests/day
# Conservative: ~400/min to stay safe
GOOGLE_REQUESTS_PER_MINUTE = 400
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2
RETRY_BACKOFF_MULTIPLIER = 2

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

BASE_FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above',
    'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
    'sqft_living15', 'sqft_lot15'
]

N_SPATIAL_CLUSTERS = 40
K_NEIGHBORS_PRICE = 10  # Used consistently everywhere

# Reference coordinates
SEATTLE_CENTER_LAT = 47.6062
SEATTLE_CENTER_LONG = -122.3321
CURRENT_YEAR = 2025

# =============================================================================
# LIGHTGBM CONFIGURATION
# =============================================================================

LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1
}

LIGHTGBM_NUM_ROUNDS = 2000
LIGHTGBM_EARLY_STOPPING = 100

# =============================================================================
# XGBOOST CONFIGURATION (Required by tech stack)
# =============================================================================

XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1
}

XGBOOST_NUM_ROUNDS = 2000
XGBOOST_EARLY_STOPPING = 100

# =============================================================================
# NEURAL NETWORK CONFIGURATION
# =============================================================================

EFFICIENTNET_MODEL = 'efficientnet_b3'
EFFICIENTNET_EMBEDDING_DIM = 1536
VISUAL_EMBEDDING_DIM = 256
TABULAR_EMBEDDING_DIM = 64
TABULAR_HIDDEN_DIM = 128

# Multi-scale fusion configuration
FUSION_HIDDEN_DIM = 512
FUSION_DROPOUT = 0.4

# Per-scale attention weights (learned during training)
MULTI_SCALE_ATTENTION = True

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

N_FOLDS = 5
RANDOM_SEED = 42

# Batch sizes - adjusted for 3 images per property
BATCH_SIZE = 24  # Reduced from 32 due to 3x image memory
BATCH_SIZE_FALLBACK = 4  # If GPU memory issues

# Optimizer
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 1.0

# Epochs and stopping
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5

# Augmentation
AUGMENTATION_ENABLED = True
AUGMENTATION_PROB = 0.5

# Mixed precision
USE_AMP = True

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_TRAIN_TRANSFORMS = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotation': 15,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.1,
        'hue': 0.05
    }
}

# =============================================================================
# EXPLAINABILITY
# =============================================================================

GRADCAM_TARGET_LAYER = 'features'  # Dynamic detection
GRADCAM_NUM_EXAMPLES = 25
SHAP_NUM_SAMPLES = 1000

# =============================================================================
# ENSEMBLE
# =============================================================================

ENSEMBLE_MODELS = ['lightgbm', 'xgboost', 'residual_fusion']

ENSEMBLE_WEIGHTS = {
    'lightgbm': 0.35,
    'xgboost': 0.15,
    'residual_fusion': 0.50
}

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "training.log"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_all_seeds(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def get_device():
    """Get best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
    except ImportError:
        pass
    return 'cpu'

DEVICE = get_device()

def get_optimal_batch_size():
    """Get optimal batch size based on GPU memory (adjusted for 3 images)."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_mem < 8:
                return BATCH_SIZE_FALLBACK
            elif gpu_mem < 12:
                return 12  # Lower due to 3x images
            elif gpu_mem < 16:
                return 16
            else:
                return BATCH_SIZE
    except:
        pass
    return BATCH_SIZE_FALLBACK

def validate_config():
    """Validate configuration settings."""
    errors = []
    warnings = []
    
    if not GOOGLE_MAPS_API_KEY:
        warnings.append("GOOGLE_MAPS_API_KEY not set - needed for image fetching")
    
    if not TRAIN_FILE.exists():
        errors.append(f"Training file not found: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        errors.append(f"Test file not found: {TEST_FILE}")
    
    # Check all zoom levels
    z16_count = len(list(IMAGES_ZOOM_16_DIR.glob('*.jpg')))
    z17_count = len(list(IMAGES_ZOOM_17_DIR.glob('*.jpg')))
    z18_count = len(list(IMAGES_ZOOM_18_DIR.glob('*.jpg')))
    
    if z16_count == 0:
        warnings.append(f"No zoom-16 images found")
    if z17_count == 0:
        warnings.append(f"No zoom-17 images found")
    if z18_count == 0:
        warnings.append(f"No zoom-18 images found")
    
    return errors, warnings

if __name__ == "__main__":
    print("="*60)
    print("CONFIGURATION VALIDATION")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"Optimal batch size: {get_optimal_batch_size()}")
    print(f"\nZoom levels: {ZOOM_LEVELS}")
    for zoom, info in ZOOM_LEVEL_INFO.items():
        print(f"  Z{zoom}: {info}")
    
    errors, warnings = validate_config()
    if warnings:
        print("\n⚠️  Warnings:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("\n❌ Errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n✅ Configuration OK!")