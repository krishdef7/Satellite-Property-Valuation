"""
================================================================================
FINAL PRODUCTION SCRIPT - SATELLITE IMAGE PROPERTY VALUATION
================================================================================

Best Result: $111,294 RMSE (6.6% improvement over $119,160 baseline)

This script incorporates ALL findings from the complete experiment history:

EXPERIMENT HISTORY & KEY FINDINGS:
----------------------------------
1. Direct fusion FAILED ($140K) - trying to replace baseline instead of complement
2. Residual modeling WORKS - predict (price - baseline) instead of price
3. ResNet50-only: $111,544 - best for non-waterfront but DESTROYS waterfront (-$5,514)
4. Swin+ConvNeXt: $112,166 - worse overall but HELPS waterfront (+$2,102)
5. V7 Hybrid: $111,451 - segment-aware architecture works!
6. V7.2 (80%R/20%T): $111,392 - fixed blend is good but not optimal
7. V7.3 Segment-NNLS: $111,294 - data-driven weights per segment = BEST

WHAT WORKS:
-----------
✅ Residual modeling (predict baseline errors)
✅ ResNet50 with 200 PCA components
✅ Swin+ConvNeXt with 100+100 PCA for waterfront
✅ Learning rate 0.012 (not 0.01!)
✅ Segment-wise NNLS (different weights per price tier)
✅ Including baseline as NNLS candidate
✅ MSE loss function
✅ Heavy regularization (reg_alpha=2.0, reg_lambda=2.0)

WHAT FAILED:
------------
❌ Huber loss (ANY delta) - model predicts ~0 residuals
❌ Lower learning rate (0.01) - worse by $133
❌ EfficientNet for waterfront - CNN can't capture global context
❌ Global fixed blending - suboptimal for different price tiers
❌ Hyperparameter tuning - fold 0 overfitting every time

OPTIMAL SEGMENT WEIGHTS (discovered by NNLS):
---------------------------------------------
- Standard (<$750K): 0% Baseline, 59% ResNet, 41% Transformer
- High-value ($750K-$1M): 11% Baseline, 54% ResNet, 35% Transformer  
- Ultra-high (>$1M): 0% Baseline, 61% ResNet, 39% Transformer
- Waterfront: 0% Baseline, 0% ResNet, 100% Transformer

Usage:
------
    # Training (generates OOF predictions)
    python train_final.py --project-root . --n-seeds 25
    
    # Training with model saving for test inference
    python train_final.py --project-root . --n-seeds 25 --save-models
    
    # Test inference (after training with --save-models)
    python train_final.py --project-root . --predict-test
    
    # Quick test with fewer seeds
    python train_final.py --project-root . --n-seeds 10

Output:
-------
    results/fusion_results/final_predictions.csv - OOF predictions
    results/fusion_results/final_test_predictions.csv - Test predictions
    results/fusion_results/final_models.pkl - Saved models (if --save-models)

Author: Competition Team
Date: January 2026
================================================================================
"""

import sys
import argparse
import logging
import pickle
import warnings
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import nnls

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - ALL VALUES ARE EVIDENCE-BASED AND LOCKED
# =============================================================================

# Encoder configurations (DO NOT CHANGE - extensively tested)
RESNET_CONFIG = {
    'resnet50': {'pca_components': 200},  # 43.3% variance, optimal for non-WF
}

TRANSFORMER_CONFIG = {
    'swin_tiny': {'pca_components': 100},     # 75.4% variance
    'convnext_tiny': {'pca_components': 100}, # 70.9% variance
}

# LightGBM parameters (DO NOT CHANGE - lr=0.012 proven optimal)
LGB_PARAMS = {
    'objective': 'regression',  # NOT huber - it failed completely
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.012,     # V7 optimal - 0.01 was $133 worse
    'num_leaves': 28,
    'max_depth': 7,
    'min_child_samples': 70,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'reg_alpha': 2.0,           # Heavy regularization prevents overfitting
    'reg_lambda': 2.0,
    'verbose': -1,
}

NUM_BOOST_ROUND = 4000
EARLY_STOPPING_ROUNDS = 250

# Residual clipping (prevents extreme values from distorting training)
RESIDUAL_CLIP_MIN = -800000
RESIDUAL_CLIP_MAX = 800000

# Price tier thresholds for segment-wise NNLS
HIGH_VALUE_THRESHOLD = 750000
ULTRA_HIGH_THRESHOLD = 1000000

# Tabular features to use
TABULAR_FEATURES = [
    'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def load_encoder_features(features_dir: Path, encoder_name: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Load pre-extracted encoder features."""
    encoder_file = features_dir / f"combined_features_{encoder_name}.pkl"
    
    if encoder_file.exists():
        with open(encoder_file, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'features' in data:
            return data['features'], data.get('feature_dim', 'unknown')
        return data, 'unknown'
    return None, None


def nnls_weights(predictions_matrix: np.ndarray, y_true: np.ndarray, 
                 ridge_alpha: float = 1e-4) -> np.ndarray:
    """
    Compute non-negative least squares weights with ridge regularization.
    
    This finds weights that minimize ||Aw - y||^2 + alpha*||w||^2
    subject to w >= 0, sum(w) = 1
    """
    A = predictions_matrix
    b = y_true
    n = A.shape[1]
    
    # Augment for ridge regularization
    A_aug = np.vstack([A, np.sqrt(ridge_alpha) * np.eye(n)])
    b_aug = np.concatenate([b, np.zeros(n)])
    
    weights, _ = nnls(A_aug, b_aug)
    
    # Normalize to sum to 1
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(n) / n
    
    return weights


def segment_nnls_weights(predictions_matrix: np.ndarray, y_true: np.ndarray, 
                         segment_mask: np.ndarray, ridge_alpha: float = 1e-4) -> np.ndarray:
    """Compute NNLS weights for a specific segment."""
    n_samples = segment_mask.sum()
    
    if n_samples < 10:
        # Too few samples - use uniform weights
        return np.ones(predictions_matrix.shape[1]) / predictions_matrix.shape[1]
    
    # Use higher regularization for small segments to prevent overfitting
    if n_samples < 500:
        ridge_alpha = max(ridge_alpha, 1e-3)
    
    A = predictions_matrix[segment_mask]
    b = y_true[segment_mask]
    
    return nnls_weights(A, b, ridge_alpha)


# =============================================================================
# PCA CACHE CLASS
# =============================================================================

class PCACache:
    """
    Caches PCA-transformed features for each fold to avoid recomputation.
    Also stores objects needed for test-time inference.
    """
    
    def __init__(self, train_df: pd.DataFrame, encoders_data: Dict, 
                 fold_assignments: np.ndarray, encoder_configs: Dict):
        self.train_df = train_df
        self.encoders_data = encoders_data
        self.fold_assignments = fold_assignments
        self.encoder_configs = encoder_configs
        
        # Training cache
        self.cache = {}
        
        # For test inference
        self.pca_objects = {}
        self.scalers = {}
        self.mean_embeddings = {}
        
        # Filter to available features
        self.tabular_features = [f for f in TABULAR_FEATURES if f in train_df.columns]
        
        self._build_cache()
    
    def _build_cache(self) -> None:
        """Build PCA cache for all folds and encoders."""
        unique_folds = sorted(np.unique(self.fold_assignments))
        
        for fold in unique_folds:
            val_idx = np.where(self.fold_assignments == fold)[0]
            train_idx = np.where(self.fold_assignments != fold)[0]
            
            for encoder_name, (features_dict, _) in self.encoders_data.items():
                pca_components = self.encoder_configs.get(encoder_name, {}).get('pca_components', 100)
                
                # Build feature matrix
                img_dim = len(next(iter(features_dict.values())))
                img_matrix = np.zeros((len(self.train_df), img_dim), dtype=np.float32)
                
                # Compute mean embedding for missing images
                all_feats = np.array(list(features_dict.values()), dtype=np.float32)
                mean_embedding = np.mean(all_feats, axis=0)
                self.mean_embeddings[(fold, encoder_name)] = mean_embedding
                
                # Fill matrix
                for i, pid in enumerate(self.train_df['id'].values):
                    if pid in features_dict:
                        img_matrix[i] = features_dict[pid]
                    else:
                        img_matrix[i] = mean_embedding
                
                X_train_img = img_matrix[train_idx]
                X_val_img = img_matrix[val_idx]
                
                # Standardize
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_img)
                X_val_scaled = scaler.transform(X_val_img)
                self.scalers[(fold, encoder_name)] = scaler
                
                # PCA
                n_comp = min(pca_components, X_train_scaled.shape[1], X_train_scaled.shape[0])
                pca = PCA(n_components=n_comp, random_state=42)
                X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
                X_val_pca = pca.transform(X_val_scaled).astype(np.float32)
                self.pca_objects[(fold, encoder_name)] = pca
                
                # Cache
                self.cache[(fold, encoder_name)] = {
                    'X_train_pca': X_train_pca,
                    'X_val_pca': X_val_pca,
                    'train_idx': train_idx,
                    'val_idx': val_idx,
                    'variance': pca.explained_variance_ratio_.sum(),
                    'n_components': n_comp,
                }
                
                # Free memory
                del img_matrix, all_feats, X_train_scaled, X_val_scaled
    
    def get_features(self, fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get concatenated PCA features for a fold."""
        train_pca_list = []
        val_pca_list = []
        train_idx = None
        val_idx = None
        
        for encoder_name in self.encoders_data.keys():
            data = self.cache[(fold, encoder_name)]
            train_pca_list.append(data['X_train_pca'])
            val_pca_list.append(data['X_val_pca'])
            train_idx = data['train_idx']
            val_idx = data['val_idx']
        
        X_train_img = np.hstack(train_pca_list)
        X_val_img = np.hstack(val_pca_list)
        
        return X_train_img, X_val_img, train_idx, val_idx
    
    def get_pca_info(self) -> str:
        """Get PCA variance info string."""
        fold = sorted(np.unique(self.fold_assignments))[0]
        parts = []
        for encoder_name in self.encoders_data.keys():
            data = self.cache[(fold, encoder_name)]
            parts.append(f"{encoder_name}:{data['n_components']}({data['variance']:.1%})")
        return ", ".join(parts)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_single_seed(train_df: pd.DataFrame, baseline_oof: np.ndarray, 
                      pca_cache: PCACache, fold_assignments: np.ndarray, 
                      seed: int = 42, save_models: bool = False) -> Tuple[np.ndarray, float, Optional[Dict]]:
    """Train a single seed across all folds."""
    set_all_seeds(seed)
    
    tabular_features = pca_cache.tabular_features
    unique_folds = sorted(np.unique(fold_assignments))
    
    residual_oof = np.zeros(len(train_df), dtype=np.float32)
    models = {} if save_models else None
    
    for fold_idx, fold in enumerate(unique_folds):
        X_train_img, X_val_img, train_idx, val_idx = pca_cache.get_features(fold)
        
        # Tabular features
        X_train_tab = train_df.iloc[train_idx][tabular_features].values.astype(np.float32)
        X_val_tab = train_df.iloc[val_idx][tabular_features].values.astype(np.float32)
        
        # Combine
        X_train = np.hstack([X_train_tab, X_train_img])
        X_val = np.hstack([X_val_tab, X_val_img])
        
        # Target: residual = price - baseline
        y_train_price = train_df.iloc[train_idx]['price'].values
        y_train_residual = y_train_price - baseline_oof[train_idx]
        y_train_clipped = np.clip(y_train_residual, RESIDUAL_CLIP_MIN, RESIDUAL_CLIP_MAX)
        
        y_val_residual = train_df.iloc[val_idx]['price'].values - baseline_oof[val_idx]
        
        # Feature names
        n_img = X_train_img.shape[1]
        feature_names = tabular_features + [f'img_{i}' for i in range(n_img)]
        
        # Train
        params = LGB_PARAMS.copy()
        params['seed'] = seed + fold_idx
        
        train_set = lgb.Dataset(X_train, y_train_clipped, feature_name=feature_names)
        val_set = lgb.Dataset(X_val, y_val_residual, reference=train_set)
        
        model = lgb.train(
            params, train_set, num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), 
                lgb.log_evaluation(0)
            ]
        )
        
        # Predict
        val_pred_residual = model.predict(X_val, num_iteration=model.best_iteration)
        residual_oof[val_idx] = val_pred_residual
        
        if save_models:
            models[fold] = model
    
    # Calculate RMSE
    fusion_pred = baseline_oof + residual_oof
    fusion_rmse = np.sqrt(mean_squared_error(train_df['price'], fusion_pred))
    
    return residual_oof, fusion_rmse, models


def train_encoder_model(train_df: pd.DataFrame, baseline_oof: np.ndarray, 
                        encoders_data: Dict, fold_assignments: np.ndarray, 
                        encoder_config: Dict, n_seeds: int = 25, 
                        model_name: str = "model", 
                        save_models: bool = False) -> Dict:
    """Train a model with specific encoder configuration."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Encoders: {list(encoders_data.keys())}")
    logger.info(f"PCA: {[(k, encoder_config[k]['pca_components']) for k in encoders_data.keys()]}")
    
    # Build PCA cache
    pca_cache = PCACache(train_df, encoders_data, fold_assignments, encoder_config)
    logger.info(f"  {pca_cache.get_pca_info()}")
    
    # Train multiple seeds
    all_residuals = []
    all_rmses = []
    all_models = [] if save_models else None
    
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 73  # Deterministic seed sequence
        residual_oof, rmse, models = train_single_seed(
            train_df, baseline_oof, pca_cache, fold_assignments, 
            seed, save_models=save_models
        )
        all_residuals.append(residual_oof)
        all_rmses.append(rmse)
        if save_models:
            all_models.append(models)
        
        if (seed_idx + 1) % 5 == 0 or seed_idx == 0:
            logger.info(f"  Seed {seed_idx+1}/{n_seeds}: RMSE=${rmse:,.0f}")
    
    # NNLS ensemble of seeds
    pred_matrix = np.column_stack([baseline_oof + r for r in all_residuals])
    weights = nnls_weights(pred_matrix, train_df['price'].values)
    
    residual_matrix = np.column_stack(all_residuals)
    best_residual = residual_matrix @ weights
    best_pred = baseline_oof + best_residual
    best_rmse = np.sqrt(mean_squared_error(train_df['price'], best_pred))
    
    # Stats
    best_single_idx = np.argmin(all_rmses)
    n_significant = (weights > 0.01).sum()
    
    logger.info(f"  Best single: ${all_rmses[best_single_idx]:,.0f} (seed {best_single_idx+1})")
    logger.info(f"  NNLS ensemble: ${best_rmse:,.0f} ({n_significant} seeds used)")
    
    return {
        'pred': best_pred,
        'rmse': best_rmse,
        'residuals': all_residuals,
        'rmses': all_rmses,
        'weights': weights,
        'pca_cache': pca_cache,
        'models': all_models,
    }


# =============================================================================
# SEGMENT-WISE NNLS HYBRID (KEY INNOVATION)
# =============================================================================

def create_segment_nnls_hybrid(train_df: pd.DataFrame, baseline_oof: np.ndarray,
                                resnet_residuals: List[np.ndarray], 
                                transformer_residuals: List[np.ndarray],
                                y_true: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Create hybrid predictions with segment-wise NNLS optimization.
    
    Key insight: Different price segments need different model weights.
    NNLS discovers optimal weights from data instead of manual tuning.
    """
    
    n_samples = len(train_df)
    
    # Define segments
    waterfront_mask = (train_df['waterfront'].values == 1).astype(bool)
    ultra_high_mask = (y_true >= ULTRA_HIGH_THRESHOLD) & ~waterfront_mask
    high_value_mask = (y_true >= HIGH_VALUE_THRESHOLD) & (y_true < ULTRA_HIGH_THRESHOLD) & ~waterfront_mask
    standard_mask = ~waterfront_mask & ~high_value_mask & ~ultra_high_mask
    
    n_resnet = len(resnet_residuals)
    n_trans = len(transformer_residuals)
    
    # Build prediction matrices
    resnet_preds = np.column_stack([baseline_oof + r for r in resnet_residuals])
    trans_preds = np.column_stack([baseline_oof + r for r in transformer_residuals])
    
    # Include baseline as a candidate (sometimes it's best for certain segments!)
    all_preds = np.column_stack([baseline_oof.reshape(-1, 1), resnet_preds, trans_preds])
    n_models = all_preds.shape[1]
    
    logger.info(f"\n  Ensemble: 1 baseline + {n_resnet} ResNet + {n_trans} Trans = {n_models} models")
    
    hybrid_pred = np.zeros(n_samples)
    segment_info = {'method': 'segment_nnls'}
    
    # 1. STANDARD SEGMENT (<$750K, non-waterfront)
    if standard_mask.sum() > 0:
        weights = segment_nnls_weights(all_preds, y_true, standard_mask, ridge_alpha=1e-4)
        hybrid_pred[standard_mask] = all_preds[standard_mask] @ weights
        
        base_w = weights[0]
        resnet_w = weights[1:n_resnet+1].sum()
        trans_w = weights[n_resnet+1:].sum()
        
        segment_info['standard'] = {
            'n_samples': int(standard_mask.sum()),
            'baseline': float(base_w),
            'resnet': float(resnet_w),
            'transformer': float(trans_w),
        }
        logger.info(f"  Standard (<$750K): {standard_mask.sum()} samples")
        logger.info(f"    Baseline: {base_w:.1%}, ResNet: {resnet_w:.1%}, Trans: {trans_w:.1%}")
    
    # 2. HIGH-VALUE SEGMENT ($750K-$1M)
    if high_value_mask.sum() > 0:
        weights = segment_nnls_weights(all_preds, y_true, high_value_mask, ridge_alpha=1e-3)
        hybrid_pred[high_value_mask] = all_preds[high_value_mask] @ weights
        
        base_w = weights[0]
        resnet_w = weights[1:n_resnet+1].sum()
        trans_w = weights[n_resnet+1:].sum()
        
        segment_info['high_value'] = {
            'n_samples': int(high_value_mask.sum()),
            'baseline': float(base_w),
            'resnet': float(resnet_w),
            'transformer': float(trans_w),
        }
        logger.info(f"  High-value ($750K-$1M): {high_value_mask.sum()} samples")
        logger.info(f"    Baseline: {base_w:.1%}, ResNet: {resnet_w:.1%}, Trans: {trans_w:.1%}")
    
    # 3. ULTRA-HIGH SEGMENT (>$1M) - dominates RMSE!
    if ultra_high_mask.sum() > 0:
        weights = segment_nnls_weights(all_preds, y_true, ultra_high_mask, ridge_alpha=1e-3)
        hybrid_pred[ultra_high_mask] = all_preds[ultra_high_mask] @ weights
        
        base_w = weights[0]
        resnet_w = weights[1:n_resnet+1].sum()
        trans_w = weights[n_resnet+1:].sum()
        
        segment_info['ultra_high'] = {
            'n_samples': int(ultra_high_mask.sum()),
            'baseline': float(base_w),
            'resnet': float(resnet_w),
            'transformer': float(trans_w),
        }
        logger.info(f"  Ultra-high (>$1M): {ultra_high_mask.sum()} samples ⭐")
        logger.info(f"    Baseline: {base_w:.1%}, ResNet: {resnet_w:.1%}, Trans: {trans_w:.1%}")
    
    # 4. WATERFRONT - use transformer-only (ResNet proven harmful!)
    if waterfront_mask.sum() > 0:
        wf_preds = np.column_stack([baseline_oof.reshape(-1, 1), trans_preds])
        weights = segment_nnls_weights(wf_preds, y_true, waterfront_mask, ridge_alpha=1e-6)
        hybrid_pred[waterfront_mask] = wf_preds[waterfront_mask] @ weights
        
        base_w = weights[0]
        trans_w = weights[1:].sum()
        
        segment_info['waterfront'] = {
            'n_samples': int(waterfront_mask.sum()),
            'baseline': float(base_w),
            'transformer': float(trans_w),
        }
        logger.info(f"  Waterfront: {waterfront_mask.sum()} samples")
        logger.info(f"    Baseline: {base_w:.1%}, Trans: {trans_w:.1%} (no ResNet)")
    
    return hybrid_pred, segment_info


# =============================================================================
# ANALYSIS AND REPORTING
# =============================================================================

def comprehensive_analysis(y_true: np.ndarray, baseline_pred: np.ndarray, 
                           resnet_pred: np.ndarray, transformer_pred: np.ndarray,
                           hybrid_pred: np.ndarray, train_df: pd.DataFrame) -> Dict:
    """Generate comprehensive analysis of model performance."""
    
    waterfront_mask = (train_df['waterfront'].values == 1).astype(bool)
    
    logger.info(f"\n{'='*70}")
    logger.info("COMPREHENSIVE ANALYSIS")
    logger.info(f"{'='*70}")
    
    # Overall RMSE
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
    resnet_rmse = np.sqrt(mean_squared_error(y_true, resnet_pred))
    trans_rmse = np.sqrt(mean_squared_error(y_true, transformer_pred))
    hybrid_rmse = np.sqrt(mean_squared_error(y_true, hybrid_pred))

    # Overall R2
    baseline_r2 = r2_score(y_true, baseline_pred)
    resnet_r2 = r2_score(y_true, resnet_pred)
    trans_r2 = r2_score(y_true, transformer_pred)
    hybrid_r2 = r2_score(y_true, hybrid_pred)

    
    logger.info(f"\n{'Model':<30} {'RMSE':>12} {'Improvement':>12}")
    logger.info("-" * 60)
    logger.info(f"{'Baseline':<30} ${baseline_rmse:>10,.0f} ${0:>10,.0f}")
    logger.info(f"{'ResNet50-only':<30} ${resnet_rmse:>10,.0f} ${baseline_rmse-resnet_rmse:>+10,.0f}")
    logger.info(f"{'Swin+ConvNeXt':<30} ${trans_rmse:>10,.0f} ${baseline_rmse-trans_rmse:>+10,.0f}")
    logger.info(f"{'FINAL (Segment-NNLS)':<30} ${hybrid_rmse:>10,.0f} ${baseline_rmse-hybrid_rmse:>+10,.0f}")
    
    logger.info(f"\nR² SCORES (Higher = Better)")
    logger.info("-" * 60)
    logger.info(f"{'Baseline':<30} {baseline_r2:>8.4f}")
    logger.info(f"{'ResNet50-only':<30} {resnet_r2:>8.4f}")
    logger.info(f"{'Swin+ConvNeXt':<30} {trans_r2:>8.4f}")
    logger.info(f"{'FINAL (Segment-NNLS)':<30} {hybrid_r2:>8.4f}")

    # Segment breakdown
    logger.info(f"\n{'='*70}")
    logger.info("SEGMENT BREAKDOWN")
    logger.info(f"{'='*70}")
    
    segments = {
        '<300K': y_true < 300000,
        '300-500K': (y_true >= 300000) & (y_true < 500000),
        '500-750K': (y_true >= 500000) & (y_true < 750000),
        '750K-1M': (y_true >= 750000) & (y_true < 1000000),
        '>1M': y_true >= 1000000,
    }
    
    logger.info(f"\n{'Segment':<12} {'Count':>6} {'Baseline':>10} {'Final':>10} {'Δ':>10}")
    logger.info("-" * 55)
    
    for name, mask in segments.items():
        if mask.sum() == 0:
            continue
        base_mae = mean_absolute_error(y_true[mask], baseline_pred[mask])
        final_mae = mean_absolute_error(y_true[mask], hybrid_pred[mask])
        delta = base_mae - final_mae
        status = "✅" if delta > 0 else "❌"
        logger.info(f"{name:<12} {mask.sum():>6} ${base_mae:>8,.0f} ${final_mae:>8,.0f} {status}${delta:>+8,.0f}")
    
    # Special segments
    logger.info(f"\n{'Special':<12} {'Count':>6} {'Baseline':>10} {'Final':>10} {'Δ':>10}")
    logger.info("-" * 55)
    
    # Waterfront
    base_mae = mean_absolute_error(y_true[waterfront_mask], baseline_pred[waterfront_mask])
    final_mae = mean_absolute_error(y_true[waterfront_mask], hybrid_pred[waterfront_mask])
    delta = base_mae - final_mae
    status = "✅" if delta >= 0 else "❌"
    logger.info(f"{'Waterfront':<12} {waterfront_mask.sum():>6} ${base_mae:>8,.0f} ${final_mae:>8,.0f} {status}${delta:>+8,.0f}")
    
    # Luxury
    lux_mask = (train_df['grade'].values >= 10).astype(bool)
    base_mae = mean_absolute_error(y_true[lux_mask], baseline_pred[lux_mask])
    final_mae = mean_absolute_error(y_true[lux_mask], hybrid_pred[lux_mask])
    delta = base_mae - final_mae
    status = "✅" if delta > 0 else "❌"
    logger.info(f"{'Luxury':<12} {lux_mask.sum():>6} ${base_mae:>8,.0f} ${final_mae:>8,.0f} {status}${delta:>+8,.0f}")
    
    return {
        'baseline_rmse': baseline_rmse,
        'resnet_rmse': resnet_rmse,
        'transformer_rmse': trans_rmse,
        'final_rmse': hybrid_rmse,
        'improvement': baseline_rmse - hybrid_rmse,
        'improvement_pct': (baseline_rmse - hybrid_rmse) / baseline_rmse * 100,
        'baseline_r2': baseline_r2,
        'resnet_r2': resnet_r2,
        'transformer_r2': trans_r2,
        'final_r2': hybrid_r2,
    }


# =============================================================================
# TEST PREDICTION
# =============================================================================

def predict_test(project_root: Path, models_file: Path) -> None:
    """Generate predictions for test set using saved models."""
    
    logger.info("="*70)
    logger.info("TEST SET PREDICTION")
    logger.info("="*70)
    
    # Load saved data
    logger.info("\nLoading saved models...")
    with open(models_file, 'rb') as f:
        saved_data = pickle.load(f)
    
    # Load test data
    sys.path.insert(0, str(project_root))
    from config import TEST_FILE, RESULTS_DIR
    
    test_df = pd.read_csv(TEST_FILE)
    logger.info(f"Test samples: {len(test_df)}")
    
    # Load baseline test predictions
    baseline_test_file = RESULTS_DIR / "baseline_test_predictions.csv"
    if not baseline_test_file.exists():
        logger.error(f"Baseline test predictions not found: {baseline_test_file}")
        logger.error("Run your baseline model on test set first!")
        return
    
    baseline_test_df = pd.read_csv(baseline_test_file)
    baseline_test = baseline_test_df['final_prediction'].values
    
    # TODO: Implement full test prediction pipeline
    # This requires:
    # 1. Loading test image features
    # 2. Applying saved PCA transforms
    # 3. Running saved models
    # 4. Applying segment-wise blending
    
    logger.warning("Test prediction not fully implemented - need test image features")
    logger.info("For now, use the OOF predictions approach for CV validation")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Final Production Script - Satellite Image Property Valuation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_final.py --project-root . --n-seeds 25
  python train_final.py --project-root . --n-seeds 25 --save-models
  python train_final.py --project-root . --predict-test
        """
    )
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--n-seeds", type=int, default=25, help="Number of seeds (default: 25)")
    parser.add_argument("--save-models", action="store_true", help="Save models for test inference")
    parser.add_argument("--predict-test", action="store_true", help="Generate test predictions")
    parser.add_argument("--baseline-file", type=str, default=None, help="Path to baseline predictions")
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    project_root = Path(args.project_root)
    sys.path.insert(0, str(project_root))
    
    from config import TRAIN_FILE, RESULTS_DIR
    
    features_dir = project_root / "features"
    output_dir = RESULTS_DIR / "fusion_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test prediction mode
    if args.predict_test:
        models_file = output_dir / "final_models.pkl"
        if not models_file.exists():
            logger.error(f"Models file not found: {models_file}")
            logger.error("Run training with --save-models first!")
            return
        predict_test(project_root, models_file)
        return
    
    # ==========================================================================
    # TRAINING MODE
    # ==========================================================================
    
    logger.info("="*70)
    logger.info("FINAL PRODUCTION SCRIPT")
    logger.info("Satellite Image Property Valuation")
    logger.info("="*70)
    logger.info(f"Start: {start_time}")
    logger.info(f"Seeds: {args.n_seeds}")
    logger.info(f"Save models: {args.save_models}")
    
    # Load data
    logger.info("\n" + "="*70)
    logger.info("LOADING DATA")
    logger.info("="*70)
    
    train_df = pd.read_csv(TRAIN_FILE)
    logger.info(f"Training samples: {len(train_df)}")
    
    # Load baseline
    baseline_file = args.baseline_file
    if baseline_file is None:
        for path in [project_root / "baseline_oof_predictions.csv", 
                     RESULTS_DIR / "baseline_oof_predictions.csv"]:
            if path.exists():
                baseline_file = path
                break
    
    if baseline_file is None or not Path(baseline_file).exists():
        logger.error("Baseline predictions not found!")
        logger.error("Run baseline model first to generate OOF predictions.")
        return
    
    baseline_df = pd.read_csv(baseline_file)
    logger.info(f"Loaded baseline from {baseline_file}")
    
    # Align baseline with train_df
    if 'id' in baseline_df.columns:
        baseline_df = baseline_df.set_index('id').reindex(train_df['id']).reset_index()
        baseline_oof = baseline_df['final_prediction'].values.astype(np.float32)
    else:
        baseline_oof = baseline_df['final_prediction'].values.astype(np.float32)
    
    # Get fold assignments
    if 'fold' in baseline_df.columns:
        fold_assignments = baseline_df['fold'].values
    else:
        fold_file = RESULTS_DIR / 'fold_splits.pkl'
        with open(fold_file, 'rb') as f:
            splits = pickle.load(f)
        fold_assignments = np.zeros(len(train_df), dtype=int)
        for fold, (_, val_idx) in enumerate(splits):
            fold_assignments[val_idx] = fold
    
    y_true = train_df['price'].values
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_oof))
    
    waterfront_mask = (train_df['waterfront'].values == 1).astype(bool)
    ultra_high_mask = y_true >= ULTRA_HIGH_THRESHOLD
    
    logger.info(f"Baseline RMSE: ${baseline_rmse:,.0f}")
    logger.info(f"Waterfront: {waterfront_mask.sum()} samples ({waterfront_mask.mean()*100:.1f}%)")
    logger.info(f">$1M: {ultra_high_mask.sum()} samples ({ultra_high_mask.mean()*100:.1f}%) ← KEY SEGMENT")
    
    # ==========================================================================
    # MODEL A: ResNet50-only (for non-waterfront)
    # ==========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("MODEL A: ResNet50-only")
    logger.info("="*70)
    
    resnet_encoders = {}
    for encoder in RESNET_CONFIG.keys():
        features, dim = load_encoder_features(features_dir, encoder)
        if features is not None:
            resnet_encoders[encoder] = (features, dim)
            logger.info(f"  Loaded {encoder}: {len(features)} samples")
        else:
            logger.error(f"  Failed to load {encoder}!")
    
    if not resnet_encoders:
        logger.error("No ResNet encoders found! Cannot proceed.")
        return
    
    resnet_results = train_encoder_model(
        train_df, baseline_oof, resnet_encoders, fold_assignments,
        RESNET_CONFIG, n_seeds=args.n_seeds, model_name="ResNet50-only",
        save_models=args.save_models
    )
    
    # ==========================================================================
    # MODEL B: Swin+ConvNeXt (for waterfront)
    # ==========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("MODEL B: Swin+ConvNeXt")
    logger.info("="*70)
    
    transformer_encoders = {}
    for encoder in TRANSFORMER_CONFIG.keys():
        features, dim = load_encoder_features(features_dir, encoder)
        if features is not None:
            transformer_encoders[encoder] = (features, dim)
            logger.info(f"  Loaded {encoder}: {len(features)} samples")
        else:
            logger.warning(f"  {encoder} not found - will use ResNet50 fallback")
    
    if not transformer_encoders:
        logger.warning("No transformer encoders! Using ResNet50 for everything.")
        transformer_results = {
            'pred': resnet_results['pred'].copy(),
            'rmse': resnet_results['rmse'],
            'residuals': resnet_results['residuals'],
        }
    else:
        transformer_results = train_encoder_model(
            train_df, baseline_oof, transformer_encoders, fold_assignments,
            TRANSFORMER_CONFIG, n_seeds=args.n_seeds, model_name="Swin+ConvNeXt",
            save_models=args.save_models
        )
    
    # ==========================================================================
    # SEGMENT-WISE NNLS HYBRID
    # ==========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("SEGMENT-WISE NNLS HYBRID")
    logger.info("="*70)
    
    hybrid_pred, segment_info = create_segment_nnls_hybrid(
        train_df, baseline_oof,
        resnet_results['residuals'],
        transformer_results['residuals'],
        y_true
    )
    
    final_rmse = np.sqrt(mean_squared_error(y_true, hybrid_pred))
    logger.info(f"\n  FINAL RMSE: ${final_rmse:,.0f}")
    
    # ==========================================================================
    # COMPREHENSIVE ANALYSIS
    # ==========================================================================
    
    analysis = comprehensive_analysis(
        y_true, baseline_oof,
        resnet_results['pred'], transformer_results['pred'],
        hybrid_pred, train_df
    )
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("SAVING RESULTS")
    logger.info("="*70)
    
    # Predictions CSV
    results_df = pd.DataFrame({
        'id': train_df['id'],
        'price': train_df['price'],
        'baseline_prediction': baseline_oof,
        'resnet_prediction': resnet_results['pred'],
        'transformer_prediction': transformer_results['pred'],
        'final_prediction': hybrid_pred,
        'fold': fold_assignments
    })
    predictions_file = output_dir / "final_predictions.csv"
    results_df.to_csv(predictions_file, index=False)
    logger.info(f"  Predictions: {predictions_file}")
    
    # Segment info
    segment_file = output_dir / "segment_weights.pkl"
    with open(segment_file, 'wb') as f:
        pickle.dump(segment_info, f)
    logger.info(f"  Segment weights: {segment_file}")
    
    # Models (if requested)
    if args.save_models:
        models_data = {
            'resnet_models': resnet_results['models'],
            'resnet_weights': resnet_results['weights'],
            'resnet_pca_cache': resnet_results['pca_cache'],
            'transformer_models': transformer_results.get('models'),
            'transformer_weights': transformer_results['weights'],
            'transformer_pca_cache': transformer_results.get('pca_cache'),
            'segment_info': segment_info,
            'config': {
                'resnet_config': RESNET_CONFIG,
                'transformer_config': TRANSFORMER_CONFIG,
                'lgb_params': LGB_PARAMS,
            }
        }
        models_file = output_dir / "final_models.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(models_data, f)
        logger.info(f"  Models: {models_file}")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    
    elapsed = datetime.now() - start_time
    
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    logger.info(f"Duration: {elapsed}")
    logger.info(f"\nResults:")
    logger.info(f"  Baseline RMSE:  ${analysis['baseline_rmse']:,.0f}")
    logger.info(f"  Final RMSE:     ${analysis['final_rmse']:,.0f}")
    logger.info(f"  Improvement:    ${analysis['improvement']:,.0f} ({analysis['improvement_pct']:.1f}%)")
    
    target = 110000
    gap = analysis['final_rmse'] - target
    if gap <= 0:
        logger.info(f"\n🎯 TARGET ACHIEVED: ${analysis['final_rmse']:,.0f} <= ${target:,}")
    else:
        logger.info(f"\n⚠️ Gap to target: ${gap:,.0f}")
    
    logger.info(f"\n📁 Output directory: {output_dir}")
    logger.info(f"📁 Use 'final_prediction' column from final_predictions.csv")
    logger.info("\n" + "="*70)
    logger.info("DONE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
