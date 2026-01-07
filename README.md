# 🏠 Satellite Imagery-Based Property Valuation

A **Multimodal Regression Pipeline** that predicts property market value by combining tabular data with satellite imagery using deep learning.

![Architecture](results/architecture_diagram.png)

## 🎯 Project Overview

This project builds a state-of-the-art property valuation model that integrates:
- **Tabular features**: Property attributes (sqft, bedrooms, grade, location, etc.)
- **Visual features**: Multi-scale satellite imagery (property, neighborhood, regional)
- **Spatial features**: Geographic clustering, property density, neighbor statistics

### Key Innovation: Segment-Wise NNLS Hybrid

Different property types need different model weights:

| Segment | Samples | Baseline | ResNet | Transformer |
|---------|---------|----------|--------|-------------|
| Standard (<$750K) | 13,465 | 0% | 59% | 41% |
| High-value ($750K-$1M) | 1,610 | 11% | 54% | 35% |
| Ultra-high (>$1M) | 1,021 | 0% | 61% | 39% |
| **Waterfront** | 113 | 0% | **0%** | **100%** |

🔑 **Critical Finding**: Waterfront properties need 100% transformer weights - ResNet hurts performance!

## 📊 Results

| Model | RMSE | Improvement |
|-------|------|-------------|
| Baseline (Tabular Only) | $119,160 | - |
| ResNet50-only | $111,544 | +$7,616 |
| Swin+ConvNeXt | $112,166 | +$6,994 |
| **Final (Segment-NNLS)** | **$111,294** | **+$7,866 (6.6%)** |

---

## 🏗️ Baseline Model Architecture

### Evolution: V1 → V6 → Final ($142K → $119K)

The baseline model went through **6 major iterations** to reach optimal performance:

| Version | RMSE | Key Changes | Outcome |
|---------|------|-------------|---------|
| **V1** | $142,000 | Initial 6-model ensemble | High variance, poor luxury handling |
| **V2** | $125,000 | Weighted loss, segment features | Luxury/waterfront still biased |
| **V3** | $119,500 | Feature pruning, meta-learning | Stable but specialists failing |
| **V4** | $118,600 | Segment specialists (luxury/WF/cheap) | Specialists catastrophic (R²=-1.88!) |
| **V5** | $118,945 | Radical simplification, remove bad specialists | Best raw, but clipping destroyed it |
| **V6** | $169,000 | Sanity checks, Huber loss | Catastrophic regression |
| **Final** | **$119,160** | V5 architecture + stability fixes, NO post-processing | Locked & stable |

### Key Lessons Learned

1. **Post-processing destroys performance**: Clipping added +$5.7K RMSE, sanity checks added +$15K
2. **Specialist models fail on small segments**: Waterfront specialist had R²=-1.88 (worse than random!)
3. **Stability > Complexity**: V5's simpler architecture beat V4's complex specialists
4. **Stratified GroupKFold is critical**: Reduces fold variance from 28.6% to <25%

### Final Baseline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   BASELINE MODEL (V5-Final)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  LightGBM   │   │  LightGBM   │   │   XGBoost   │       │
│  │   (Main)    │   │  (Diverse)  │   │             │       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│         │                 │                 │               │
│  ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐       │
│  │  CatBoost   │   │  CatBoost   │   │ RandomForest│       │
│  │   (Main)    │   │  (Diverse)  │   │             │       │
│  └──────┬──────┘   └──────┴──────┘   └──────┬──────┘       │
│         │                 │                 │               │
│         └────────┬────────┴────────┬────────┘               │
│                  │                 │                        │
│         ┌────────▼────────┐  ┌─────▼─────┐                 │
│         │ Luxury Specialist│  │  Sample   │                 │
│         │ (65/35 soft blend)│ │  Weights  │                 │
│         └────────┬────────┘  └─────┬─────┘                 │
│                  │                 │                        │
│                  └────────┬────────┘                        │
│                           │                                 │
│                  ┌────────▼────────┐                        │
│                  │   ElasticNet    │                        │
│                  │  Meta-Learner   │                        │
│                  │  (α=0.01, tight)│                        │
│                  └────────┬────────┘                        │
│                           │                                 │
│                  ┌────────▼────────┐                        │
│                  │  Final Stacked  │                        │
│                  │   Prediction    │                        │
│                  │  (NO clipping!) │                        │
│                  └─────────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Stability Fixes (Critical!)

| Fix | Problem | Solution |
|-----|---------|----------|
| **Stratified GroupKFold** | Folds had uneven price distributions | Balance luxury/waterfront/price bins across folds |
| **Robust Neighborhood Features** | Sparse areas caused noise | Blend with global stats based on reliability score |
| **Softened Luxury Blend** | Luxury specialist over-pulled predictions | 65/35 blend instead of 70/30 |
| **Tighter ElasticNet** | Unstable meta-weights | α=0.01 (10x stronger regularization) |
| **NO Post-Processing** | Clipping/sanity destroyed RMSE | Trust the model's raw predictions |

### Feature Engineering (60+ Features)

```python
# Core Transforms
log_sqft_living, log_sqft_lot, log_sqft_above
bath_bed_ratio, living_lot_ratio, basement_ratio
age, years_since_renovation, age_squared, log_age

# Quality Interactions (TOP performers)
quality_sqft_interaction = grade * log_sqft_living
grade_condition_score = grade * condition
grade_view_interaction = grade * (view + 1)

# Luxury Features (CRITICAL)
is_luxury = (grade >= 10) | (waterfront == 1) | (sqft_living > 4500) | (view >= 3)
is_ultra_luxury = (grade >= 12) | ((waterfront == 1) & (sqft_living > 3000))
luxury_index = log_sqft_living * grade * (view + 1) * (1 + waterfront)

# Waterfront Features
waterfront_grade, waterfront_sqft, waterfront_grade_sqft
waterfront_quality = waterfront * grade * condition

# Neighborhood Features (with reliability weighting)
local_avg_price, local_weighted_price, local_median_price
local_price_ppsf, local_luxury_ratio, local_waterfront_ratio
local_reliability  # NEW: tracks neighborhood data quality

# Cluster Features (3-level hierarchy)
cluster_coarse (20), cluster_medium (40), cluster_fine (80)
dist_to_centroid_*, cluster_avg_price_*, cluster_ppsf_*
```

### Sample Weighting Strategy

```python
# Moderate weighting (learned from V4 failures)
base_weight = 1.0 / sqrt(price + 1)
weights[is_luxury == 1] *= 2.0      # Luxury homes
weights[is_waterfront == 1] *= 2.5  # Waterfront premium
weights[price > 1M] *= 1.3          # High-value
weights[price > 2M] *= 1.2          # Ultra-high
weights = clip(weights, 0.5, 3.0)   # Prevent extremes
```

### Running the Baseline

```bash
python train_baseline_final.py --project-root .
```

Output: `results/baseline_oof_predictions.csv`

## 🗂️ Project Structure

```
├── data/
│   ├── raw/                    # Original Excel files (converted to CSV)
│   │   ├── train.csv
│   │   └── test.csv
│   └── images/
│       ├── zoom_16/            # Regional context (~2.4km)
│       ├── zoom_17/            # Neighborhood (~1.2km)
│       └── zoom_18/            # Property detail (~600m)
├── features/
│   ├── combined_features_resnet50.pkl
│   ├── combined_features_swin_tiny.pkl
│   └── combined_features_convnext_tiny.pkl
├── results/
│   ├── train_processed.csv
│   ├── test_processed.csv
│   ├── baseline_oof_predictions.csv   # Baseline model OOF
│   ├── oof_predictions.csv            # Final fusion OOF
│   └── visualizations/
├── config.py                   # All configuration settings
├── data_fetcher.py            # Satellite image downloading
├── preprocessing.ipynb        # EDA & feature extraction
├── model_training.ipynb       # Model training & evaluation
├── train_baseline_final.py    # Baseline tabular model (V1→V6→Final)
├── train_final.py             # Production fusion training
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your Excel files in `data/raw/`:
- `train.xlsx` → Will be converted to `train.csv`
- `test.xlsx` → Will be converted to `test.csv`

### 3. Set Up Google Maps API

```bash
export GOOGLE_MAPS_API_KEY="your_api_key_here"
```

Get a key at: https://console.cloud.google.com/apis/credentials

### 4. Fetch Satellite Images

```bash
# Fetch all images (train + test)
python data_fetcher.py --fetch-all

# Or fetch separately
python data_fetcher.py --fetch-train
python data_fetcher.py --fetch-test

# Check cache statistics
python data_fetcher.py --stats
```

### 5. Run Preprocessing

Open and run `preprocessing.ipynb` cell by cell, or:

```bash
jupyter notebook preprocessing.ipynb
```

This notebook will:
- Load and explore the data
- Create engineered features
- Extract CNN embeddings from satellite images
- Save processed data

### 6. Train Baseline Model (Important!)

**Run the baseline model first** - this generates the OOF predictions needed for fusion:

```bash
python train_baseline_final.py --project-root .
```

This will:
- Train 6-model ensemble (LightGBM, XGBoost, CatBoost, RF)
- Apply stability fixes (stratified CV, robust neighborhoods)
- Generate `results/baseline_oof_predictions.csv`
- Expected RMSE: ~$119,160

### 7. Train Fusion Models

Open and run `model_training.ipynb` cell by cell, or:

```bash
jupyter notebook model_training.ipynb
```

This notebook will:
- Load baseline predictions (from step 6)
- Train residual fusion models with CNN features
- Create segment-wise NNLS hybrid
- Generate comprehensive analysis

### 8. Generate Production Predictions

For production predictions:

```bash
python train_final.py --project-root . --n-seeds 25
```

Output: `results/fusion_results/final_predictions.csv`

## 📝 Detailed Documentation

### Data Description

| Column | Description |
|--------|-------------|
| `sqft_living` | Interior living space |
| `sqft_above` | Above-ground living space |
| `sqft_basement` | Below-ground living space |
| `sqft_lot` | Total land area |
| `sqft_living15` / `sqft_lot15` | Average of nearest 15 neighbors |
| `condition` (1-5) | Maintenance quality |
| `grade` (1-13) | Construction quality and design |
| `view` (0-4) | View rating |
| `waterfront` | Binary: overlooks water |

---

## 🔬 Fusion Model Architecture

### Evolution: V1 → V7.3 → Final ($140K → $111K)

The fusion model also went through extensive iteration:

| Version | RMSE | Key Changes | Outcome |
|---------|------|-------------|---------|
| **V1** | $140,535 | Direct price prediction with CNN | **WORSE than baseline!** |
| **V2** | $112,063 | **Residual modeling** (price - baseline) | Breakthrough! |
| **V3** | $112,500 | Multi-encoder (R50+Swin+ConvNeXt) | Dilution hurt |
| **V4** | $112,200 | ResNet-dominant + waterfront protection | Better segmentation |
| **V5** | $111,940 | Best single seed beat ensemble | Ensemble plateau |
| **V6.1** | $111,544 | ResNet50-only + NNLS ensemble | Strong baseline |
| **V7** | $111,451 | Hybrid (ResNet for non-WF, Trans for WF) | Segment-aware |
| **V7.2** | $111,392 | Soft blend 80%R/20%T + 100+100 PCA | Optimized |
| **V7.3** | **$111,294** | **Segment-wise NNLS** | **Final** |

### Key Discoveries

1. **Residual modeling is critical**: Direct prediction ($140K) → Residual ($112K) = instant $28K improvement
2. **ResNet50 best for standard homes**: Captures property-level details
3. **Transformers (Swin/ConvNeXt) best for waterfront**: Captures global context (views, surroundings)
4. **NNLS beats manual blending**: Data-driven weights outperform 80/20 fixed ratio
5. **Segment-wise optimization**: Different price tiers need different strategies

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA                                │
├─────────────────────┬───────────────────────────────────────┤
│    Tabular Data     │         Satellite Images              │
│  (17 features)      │  (3 scales: Z16, Z17, Z18)           │
└─────────┬───────────┴───────────────┬───────────────────────┘
          │                           │
          ▼                           ▼
┌─────────────────────┐   ┌───────────────────────────────────┐
│   LightGBM          │   │     CNN Feature Extraction        │
│   Baseline          │   │  ┌─────────┐ ┌─────────┐ ┌───────┐│
│                     │   │  │ResNet50 │ │Swin     │ │ConvNeXt││
│                     │   │  │(2048D)  │ │(768D)   │ │(768D) ││
└─────────┬───────────┘   │  └────┬────┘ └────┬────┘ └───┬───┘│
          │               │       │           │          │    │
          │               │       ▼           ▼          ▼    │
          │               │    ┌──────────────────────────┐   │
          │               │    │   PCA Dimensionality     │   │
          │               │    │   Reduction              │   │
          │               │    │   R:200 + S:100 + C:100 │   │
          │               │    └───────────┬──────────────┘   │
          │               └────────────────┼──────────────────┘
          │                                │
          ▼                                ▼
┌─────────────────────┐   ┌───────────────────────────────────┐
│  Baseline           │   │    Residual Prediction            │
│  Prediction         │   │    (price - baseline)             │
│  (price estimate)   │   │    Tabular + CNN features         │
└─────────┬───────────┘   └───────────────┬───────────────────┘
          │                               │
          └───────────┬───────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  Segment-Wise NNLS    │
          │  Weight Optimization  │
          │  ─────────────────────│
          │  Standard: 59%R/41%T  │
          │  High-value: 54%R/35%T│
          │  Waterfront: 100%T    │
          └───────────┬───────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │   Final Prediction    │
          │   RMSE: $111,294      │
          └───────────────────────┘
```

### Key Technical Decisions

#### 1. Residual Modeling (vs. Direct Prediction)
❌ Direct fusion predicts price directly from CNN+tabular → RMSE $140,535
✅ Residual fusion predicts `price - baseline` → RMSE $111,294

The baseline captures most price variance; CNNs focus on what it misses.

#### 2. Multi-Encoder Strategy
- **ResNet50** (200 PCA): Best for standard properties, captures property details
- **Swin+ConvNeXt** (100+100 PCA): Best for waterfront, captures global context

#### 3. Segment-Wise Optimization
Different price tiers have different characteristics. NNLS discovers optimal weights automatically from data, avoiding manual tuning.

#### 4. Waterfront Special Handling
Waterfront properties are unique: views and surroundings matter more than property details. ResNet (local features) hurts; transformers (global context) help.

### Hyperparameters (Locked - Evidence-Based)

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Learning Rate | 0.012 | 0.01 was $133 worse |
| ResNet PCA | 200 | 43.3% variance, optimal |
| Swin PCA | 100 | 75.4% variance |
| ConvNeXt PCA | 100 | 70.9% variance |
| Seeds | 25 | 50 only +$11 improvement |
| Loss | MSE | Huber predicts ~0 residuals |

## 📈 Visual Insights

### Price Distribution by Segment

The model handles all price segments effectively:
- Standard properties (<$750K): Core market, 83.5% of data
- High-value ($750K-$1M): Transition segment, needs baseline help
- Ultra-high (>$1M): Dominates RMSE, well-handled by CNN
- Waterfront: Requires transformer-only approach

### Satellite Image Analysis

Multi-scale imagery captures different features:
- **Zoom 16** (Regional): Urban density, road networks
- **Zoom 17** (Neighborhood): Street patterns, local amenities
- **Zoom 18** (Property): Building footprint, parking, landscaping

Green dominance (vegetation proxy) correlates with price at all scales.

## 🔧 Configuration

All settings in `config.py`:

```python
# API Settings
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
IMAGE_SIZE = 512
ZOOM_LEVELS = [16, 17, 18]

# Model Settings
N_FOLDS = 5
RANDOM_SEED = 42
LIGHTGBM_PARAMS = {...}

# Feature Engineering
N_SPATIAL_CLUSTERS = 40
K_NEIGHBORS_PRICE = 10
```

## 📦 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
requests>=2.31.0
scipy>=1.11.0
```

## 🎓 Key Learnings

### From Baseline Development (6 Versions)

1. **Post-processing is dangerous**: Clipping and sanity checks destroyed +$20K RMSE
2. **Specialist models need large samples**: Waterfront specialist (113 samples) had R²=-1.88
3. **Stability beats complexity**: Simple V5 outperformed complex V4 with specialists
4. **Stratified CV is essential**: Reduces variance from 28.6% to <25%
5. **Trust the ensemble**: Raw stacked predictions beat all post-processing attempts

### From Fusion Development (7 Versions)

1. **Residual modeling is critical**: Predict what baseline misses, not raw prices
2. **Different encoders for different segments**: ResNet for details, Transformers for context
3. **Waterfront needs special handling**: ResNet actually hurts waterfront predictions
4. **NNLS ensemble beats averaging**: Data-driven weights outperform manual tuning
5. **Segment-wise optimization**: Standard homes need 41% transformer (not 20%!)

### Technical Insights

1. **Learning rate matters**: 0.012 was $133 better than 0.01 for ResNet model
2. **PCA dimensions**: ResNet 200 (43% variance), Swin/ConvNeXt 100 each (75%/71%)
3. **Seeds plateau**: 25 seeds sufficient, 50 only adds +$11 improvement
4. **Ridge regularization in NNLS**: Higher for small segments (1e-3 vs 1e-4)

## 📚 References

- [EfficientNet](https://arxiv.org/abs/1905.11946) - Efficient CNN backbone
- [Swin Transformer](https://arxiv.org/abs/2103.14030) - Vision transformer for images
- [ConvNeXt](https://arxiv.org/abs/2201.03545) - Modern CNN architecture
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting framework
- [NNLS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html) - Non-negative least squares

## 📄 License

This project is for educational and competition purposes.

---

**Author:** Competition Submission  
**Date:** January 2026
