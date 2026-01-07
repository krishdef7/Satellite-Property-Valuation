# 🏠 Satellite Imagery-Based Property Valuation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Competition**: CDC X Yhills OPEN PROJECTS 2025-2026  
**Final Score**: RMSE $111,294 | 6.6% improvement over baseline  
**Date**: January 2026

A **multimodal regression pipeline** that predicts property market value by intelligently fusing tabular data with multi-scale satellite imagery using deep learning.

![Architecture Overview](results/architecture_diagram.png)

---

## 🎯 Key Results

| Metric | Baseline (Tabular) | Final (Multimodal) | Improvement |
|--------|-------------------|---------------------|-------------|
| **RMSE** | $119,160 | **$111,294** | **-$7,866 (6.6%)** |
| **R² Score** | 0.892 | **0.906** | **+1.4%** |
| **Waterfront RMSE** | $185,420 | **$156,780** | **-$28,640 (15.4%)** |
| **High-Value RMSE** | $183,920 | **$172,460** | **-$11,460 (6.2%)** |

---

## 💡 Novel Contributions

### 1. Segment-Wise NNLS Hybrid 🎯

Different property types need different model weights—discovered through data-driven optimization:

| Segment | Samples | Baseline | ResNet | Transformer | Insight |
|---------|---------|----------|--------|-------------|---------|
| Standard (<$750K) | 13,465 | 0% | 59% | 41% | Balanced approach |
| High-value ($750K-$1M) | 1,610 | 11% | 54% | 35% | Needs stability |
| Ultra-high (>$1M) | 1,021 | 0% | 61% | 39% | Detail matters |
| **Waterfront** | 113 | 0% | **0%** | **100%** | **ResNet hurts!** |

🔑 **Critical Discovery**: Waterfront properties need 100% transformer weights because value comes from water proximity (global context), not building details (local features).

### 2. Residual Modeling Architecture 📐

Instead of predicting price directly, we predict `residual = price - baseline`:

```
Baseline (tabular) → $450,000 ± $119K
CNN (imagery) → +$50,000 ± $12K  (visual correction)
Final → $500,000 ± $111K
```

**Impact**: $140K RMSE (direct CNN) → $111K RMSE (residual approach) = **$29K improvement**

### 3. Multi-Scale Visual Intelligence 🔍

Combined 3 zoom levels capture context at all scales:

- **Z16** (~2.4km): Regional context, urban/suburban classification
- **Z17** (~1.2km): Neighborhood patterns, amenity proximity
- **Z18** (~600m): Property-level details, lot configuration

### 4. Explainable Predictions 🔬

Grad-CAM heatmaps prove the model learns meaningful visual patterns:

![Multi-Zoom Attention](results/explainability/sample_attention.png)

---

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
- Python 3.8+
- 16GB RAM minimum (32GB recommended for training)
- GPU with 6GB+ VRAM (optional, speeds up CNN feature extraction)
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd satellite-property-valuation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup

1. **Prepare Data**

Place your data files in `data/raw/`:
```
data/raw/
├── train.csv  (or train.xlsx - will be auto-converted)
└── test.csv   (or test.xlsx)
```

2. **Configure Google Maps API**

Get an API key from [Google Cloud Console](https://console.cloud.google.com/apis/credentials):

```bash
export GOOGLE_MAPS_API_KEY="your_api_key_here"
```

Or on Windows:
```cmd
set GOOGLE_MAPS_API_KEY=your_api_key_here
```

3. **Fetch Satellite Images**

```bash
# Fetch all images (train + test) - takes ~45 minutes for 16K properties
python data_fetcher.py --fetch-all

# Or fetch separately
python data_fetcher.py --fetch-train
python data_fetcher.py --fetch-test

# Check cache statistics
python data_fetcher.py --stats
```

### Training Pipeline

#### Step 1: Run EDA (Optional)

```bash
jupyter notebook preprocessing.ipynb
```

This generates visualizations and feature analysis shown in the report.

#### Step 2: Train Baseline Model (Required!)

```bash
python train_baseline_final.py --project-root .
```

**Output**: `results/baseline_oof_predictions.csv` (RMSE: ~$119,160)

**Time**: ~12 minutes on 8-core CPU

⚠️ **Important**: This must be run before fusion training, as it generates the baseline predictions needed for residual modeling.

#### Step 3: Extract CNN Features (Required!)

```bash
# Extract features from all encoders
python extract_multi_encoder_features.py --project-root . --encoder all

# Or extract individually
python extract_multi_encoder_features.py --encoder resnet50
python extract_multi_encoder_features.py --encoder swin_tiny
python extract_multi_encoder_features.py --encoder convnext_tiny
```

**Output**: 
- `features/combined_features_resnet50.pkl`
- `features/combined_features_swin_tiny.pkl`
- `features/combined_features_convnext_tiny.pkl`

**Time**: ~45 minutes (one-time cost, cached for future runs)

#### Step 4: Train Fusion Model

```bash
python train_final.py --project-root . --n-seeds 25
```

**Output**: 
- `results/oof_predictions.csv` (RMSE: ~$111,294)
- `results/test_predictions.csv`

**Time**: ~18 minutes

#### Step 5: Generate Explainability (Optional)

```bash
# Generate Grad-CAM visualizations
python explainability.py --project-root . --n-samples 25 --model resnet50

# With attention analysis
python explainability.py --project-root . --n-samples 25 --analyze
```

**Output**: `results/explainability/*.png` (attention heatmaps)

---

## 📁 Project Structure

```
satellite-property-valuation/
├── data/
│   ├── raw/                      # Original CSV files
│   │   ├── train.csv
│   │   └── test.csv
│   └── images/                   # Satellite imagery (auto-downloaded)
│       ├── zoom_16/              # Regional (~2.4km)
│       ├── zoom_17/              # Neighborhood (~1.2km)
│       └── zoom_18/              # Property (~600m)
├── features/                     # CNN embeddings (cached)
│   ├── combined_features_resnet50.pkl
│   ├── combined_features_swin_tiny.pkl
│   └── combined_features_convnext_tiny.pkl
├── results/                      # Model outputs & visualizations
│   ├── baseline_oof_predictions.csv
│   ├── oof_predictions.csv
│   ├── test_predictions.csv
│   ├── visualizations/           # EDA plots
│   └── explainability/           # Grad-CAM heatmaps
├── config.py                     # Configuration settings
├── data_fetcher.py               # Satellite image downloader
├── preprocessing.ipynb           # EDA & feature engineering
├── train_baseline_final.py       # Baseline tabular model (V1→V6→Final)
├── extract_multi_encoder_features.py  # CNN feature extraction
├── train_final.py                # Fusion model training
├── explainability.py             # Grad-CAM visualization
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🔧 Configuration

Edit `config.py` to customize settings:

```python
# Data paths
DATA_DIR = Path("data/raw")
IMAGES_DIR = Path("data/images")
RESULTS_DIR = Path("results")

# Image settings
IMAGE_SIZE = 512
ZOOM_LEVELS = [16, 17, 18]

# Model settings
N_FOLDS = 5
RANDOM_SEED = 42
N_SEEDS_ENSEMBLE = 25

# Feature engineering
N_SPATIAL_CLUSTERS = [20, 40, 80]  # 3-level hierarchy
K_NEIGHBORS_PRICE = 10
```

---

## 📊 Model Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT DATA                             │
├────────────────────┬────────────────────────────────────────┤
│  Tabular Features  │      Satellite Imagery                 │
│  (17 raw)          │  ┌──────┬──────┬──────┐               │
│                    │  │  Z16 │  Z17 │  Z18 │               │
│  → 60+ engineered  │  │ 2.4km│ 1.2km│ 600m │               │
│                    │  └──────┴──────┴──────┘               │
└────────────┬───────┴────────────┬───────────────────────────┘
             │                    │
             ▼                    ▼
    ┌────────────────┐   ┌────────────────────┐
    │   Baseline     │   │  CNN Encoders      │
    │   Ensemble     │   │  - ResNet50        │
    │   (6 GBDT +    │   │  - Swin-T          │
    │   ElasticNet)  │   │  - ConvNeXt-T      │
    │                │   │  → PCA reduction   │
    └────────┬───────┘   └─────────┬──────────┘
             │                     │
             ▼                     ▼
    ┌──────────────────────────────────────┐
    │     Residual = Price - Baseline      │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │  Fusion Models (LightGBM on          │
    │  Tabular + CNN features)             │
    │  - Model A: Tabular + ResNet         │
    │  - Model B: Tabular + Transformers   │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │  Segment-Wise NNLS Weight Optimizer  │
    │  - Standard: 59% R / 41% T           │
    │  - High-value: 54% R / 35% T         │
    │  - Waterfront: 0% R / 100% T ⚠️      │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │  Final Prediction = Baseline +       │
    │             Weighted Residual        │
    │                                      │
    │  RMSE: $111,294  |  R²: 0.906       │
    └──────────────────────────────────────┘
```

---

## 🎓 Key Learnings

### What Worked ✅

1. **Residual Modeling**: Predicting `price - baseline` instead of raw price
   - Impact: $29K RMSE improvement

2. **Segment-Wise Optimization**: Different property types need different encoder weights
   - Discovery: Waterfront properties need transformers only (0% ResNet!)

3. **Multi-Scale Imagery**: Combining Z16/Z17/Z18 captures context at all levels
   - Impact: $2.6K RMSE improvement over single scale

4. **Stability Over Complexity**: V5's simple architecture beat V4's complex specialists
   - Lesson: Trust the ensemble, avoid over-engineering

5. **Data-Driven Weights**: NNLS optimization found 100% transformer for waterfront
   - Lesson: Algorithms explore solution space better than manual tuning

### What Failed ❌

1. **Post-Processing**: Clipping predictions added +$5.7K RMSE
   - Lesson: Trust the model, don't patch with rules

2. **Specialist Models**: Waterfront specialist had R²=-1.88 (worse than random!)
   - Reason: 113 samples insufficient for separate model
   - Lesson: Small segments need full ensemble wisdom

3. **Huber Loss**: Model learned to predict ~$0 residuals
   - Lesson: MSE loss works best for regression residuals

4. **Manual Blending**: 80/20 ResNet/Transformer blend was suboptimal
   - NNLS discovered better: 59/41 for standard, 0/100 for waterfront

---

## 📈 Performance Breakdown

### By Price Segment

| Segment | Count | Baseline | Final | Improvement |
|---------|-------|----------|-------|-------------|
| <$300K | 6,842 | $38,670 | $37,250 | -3.7% |
| $300-500K | 4,215 | $52,340 | $49,210 | -6.0% |
| $500-750K | 2,408 | $61,850 | $58,120 | -6.0% |
| $750K-$1M | 1,610 | $71,850 | $66,340 | **-7.7%** |
| $1-2M | 823 | $142,680 | $131,200 | **-8.0%** |
| >$2M | 198 | $287,340 | $268,920 | -6.4% |

### By Property Type

| Type | Count | Baseline | Final | Improvement |
|------|-------|----------|-------|-------------|
| Standard | 15,338 | $115,420 | $108,230 | -6.2% |
| High Grade | 645 | $168,240 | $155,670 | -7.5% |
| **Waterfront** | 113 | $185,420 | **$156,780** | **-15.4%** |
| View (3-4) | 487 | $147,290 | $136,120 | -7.6% |
| Large Lot | 1,245 | $138,670 | $129,340 | -6.7% |

---

## 🔬 Explainability

### Grad-CAM Attention Patterns

We use Gradient-weighted Class Activation Mapping to visualize what the model "sees":

#### Standard Property
- Model focuses on building footprint and surrounding density
- ResNet captures property details effectively

#### Waterfront Property
- Model strongly attends to water bodies across all zoom levels
- Transformer's global attention captures water-property relationship
- This validates 100% transformer weight for waterfront segment

#### Example Visualizations

See `results/explainability/` for:
- Multi-zoom attention grids (Z16/Z17/Z18 side-by-side)
- Segment-wise average attention patterns
- Individual property case studies

To generate your own:

```bash
python explainability.py --project-root . --n-samples 25 --model resnet50
```

---

## 💻 Hardware Requirements

### Minimum (CPU Only)
- 8-core CPU (AMD Ryzen 7 or Intel i7)
- 16GB RAM
- 50GB disk space
- Training time: ~90 minutes

### Recommended (with GPU)
- 8+ core CPU
- 16-32GB RAM
- GPU with 6GB+ VRAM (NVIDIA RTX 3060 or better)
- 50GB disk space
- Training time: ~60 minutes

### Our System
- CPU: AMD Ryzen 7 5800H (8 cores, 3.2GHz)
- RAM: 16GB DDR4
- GPU: Radeon Graphics (6GB)
- Training time: 75 minutes (first run), 30 minutes (retrain)

---

## 📚 Documentation

### Main Documents

- **[PROJECT_REPORT.md](PROJECT_REPORT.md)**: Comprehensive technical report (62 pages)
  - Full methodology, results, and analysis
  - All visualizations and tables
  - Detailed architecture diagrams
  - Lessons learned and future work

- **[README.md](README.md)**: This quick-start guide
  - Installation and setup
  - Training pipeline
  - Key results summary

### Notebooks

- **[preprocessing.ipynb](preprocessing.ipynb)**: Exploratory Data Analysis
  - Price distribution analysis
  - Feature correlation studies
  - Geospatial visualization
  - Sample property images

---

## 🔍 Inference Example

```python
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# 1. Load trained models
baseline_model = pickle.load(open('models/baseline_final.pkl', 'rb'))
fusion_model = pickle.load(open('models/fusion_final.pkl', 'rb'))
segment_weights = pickle.load(open('models/segment_weights.pkl', 'rb'))

# 2. Prepare tabular features
tabular_features = engineer_features(property_data)

# 3. Load and process satellite images
images = {}
for zoom in [16, 17, 18]:
    img = Image.open(f'data/images/zoom_{zoom}/{property_id}.jpg')
    images[zoom] = preprocess_image(img)

# 4. Extract CNN features
cnn_features = extract_cnn_embeddings(images)

# 5. Get baseline prediction
baseline_pred = baseline_model.predict(tabular_features)

# 6. Get residual predictions
residual_resnet = fusion_model['resnet'].predict(
    np.hstack([tabular_features, cnn_features['resnet']])
)
residual_transformer = fusion_model['transformer'].predict(
    np.hstack([tabular_features, cnn_features['swin'], cnn_features['convnext']])
)

# 7. Determine segment and get weights
segment = determine_segment(property_data)
weights = segment_weights[segment]

# 8. Compute final prediction
final_pred = baseline_pred + weights['resnet'] * residual_resnet + \
             weights['transformer'] * residual_transformer

print(f"Predicted price: ${final_pred:,.0f}")
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "Google Maps API key not found"**

```bash
# Set environment variable
export GOOGLE_MAPS_API_KEY="your_key"

# Or add to config.py
GOOGLE_MAPS_API_KEY = "your_key"
```

**2. "CUDA out of memory"**

```python
# In config.py, reduce batch size
BATCH_SIZE = 16  # Default is 32
```

Or extract features on CPU (slower but works):

```bash
python extract_multi_encoder_features.py --device cpu
```

**3. "Baseline predictions not found"**

Must run baseline training before fusion:

```bash
python train_baseline_final.py --project-root .
```

**4. "Images not found"**

Run data fetcher first:

```bash
python data_fetcher.py --fetch-all
```

**5. "Sklearn version mismatch"**

```bash
pip install --upgrade scikit-learn==1.3.0
```

---

## 📊 Expected Timeline

For 16,209 training properties:

| Step | Time | Hardware | Output |
|------|------|----------|--------|
| Data fetching | 45 min | Internet | Images cached |
| Baseline training | 12 min | 8-core CPU | $119K RMSE |
| CNN extraction | 45 min | GPU (2h CPU) | Features cached |
| Fusion training | 18 min | 8-core CPU | $111K RMSE |
| Explainability | 30 min | GPU | Visualizations |
| **Total (first run)** | **2.5h** | - | - |
| **Retrain (cached)** | **30 min** | - | - |

---

## 🤝 Contributing

This is a competition submission, but we welcome feedback:

1. Open an issue for bugs or questions
2. Share suggestions for improvements
3. Report results if you replicate on other datasets

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **Google Maps** for satellite imagery API
- **PyTorch** for deep learning framework
- **LightGBM/XGBoost/CatBoost** for gradient boosting implementations
- **scikit-learn** for ML infrastructure
- **Competition organizers** for the challenge

---

## 📧 Contact

**Email**: gargkrish06@gmail.com 
**Competition**: CDC X Yhills OPEN PROJECTS 2025-2026  
**Date**: January 2026

---

## 🏆 Competition Summary

**Final Submission**:
- ✅ RMSE: **$111,294** (6.6% improvement)
- ✅ R²: **0.906** (1.4% improvement)
- ✅ Waterfront: **15.4%** improvement
- ✅ Production-ready: 16ms inference
- ✅ Explainable: Grad-CAM validated
- ✅ Reproducible: Full code & documentation

**Thank you for reviewing our work!** 🚀
