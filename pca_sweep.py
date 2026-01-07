#!/usr/bin/env python3
"""
PCA Component Sweep Script

Quickly tests different PCA component values to find optimal setting.

Usage:
    python pca_sweep.py --project-root . --pca-values 50,75,100,150,200

Expected runtime: ~15-30 min per configuration (on pre-extracted features)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pca_sweep(project_root, pca_values, n_folds=3):
    """Run sweep over PCA component values."""
    
    project_root = Path(project_root)
    sys.path.insert(0, str(project_root))
    
    from config import TRAIN_FILE, RESULTS_DIR
    
    # Load training data
    train_df = pd.read_csv(TRAIN_FILE)
    logger.info(f"Loaded {len(train_df)} training samples")
    
    # Load image features
    features_dir = project_root / "features"
    combined_files = list(features_dir.glob("combined_features*.pkl"))
    
    if not combined_files:
        logger.error("No feature files found. Run feature extraction first.")
        return
    
    # Use first available feature file
    with open(combined_files[0], 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'features' in data:
        image_features = data['features']
        logger.info(f"Loaded features from {combined_files[0].name}")
    else:
        image_features = data
    
    # Build image matrix
    img_feature_dim = len(next(iter(image_features.values())))
    img_matrix = np.zeros((len(train_df), img_feature_dim))
    
    all_feats = np.array(list(image_features.values()))
    mean_embedding = np.mean(all_feats, axis=0)
    
    for i, pid in enumerate(train_df['id'].values):
        img_matrix[i] = image_features.get(pid, mean_embedding)
    
    logger.info(f"Image matrix: {img_matrix.shape}")
    
    # Create simple CV splits (faster than full pipeline)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = []
    
    for n_components in pca_values:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing PCA components = {n_components}")
        logger.info(f"{'='*50}")
        
        fold_rmses = []
        fold_maes = []
        variance_explained_list = []
        
        start_time = datetime.now()
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            # Scale and PCA
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(img_matrix[train_idx])
            X_val_scaled = scaler.transform(img_matrix[val_idx])
            
            pca = PCA(n_components=min(n_components, X_train_scaled.shape[1]))
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_val_pca = pca.transform(X_val_scaled)
            
            variance_explained = pca.explained_variance_ratio_.sum()
            variance_explained_list.append(variance_explained)
            
            # Train simple LGB
            y_train = np.log1p(train_df.iloc[train_idx]['price'].values)
            y_val = train_df.iloc[val_idx]['price'].values
            
            train_set = lgb.Dataset(X_train_pca, y_train)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'verbose': -1,
                'seed': 42 + fold
            }
            
            model = lgb.train(params, train_set, num_boost_round=500)
            
            val_pred = np.expm1(model.predict(X_val_pca))
            
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            mae = mean_absolute_error(y_val, val_pred)
            
            fold_rmses.append(rmse)
            fold_maes.append(mae)
            
            logger.info(f"  Fold {fold+1}: RMSE=${rmse:,.0f}, Variance={variance_explained:.1%}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        result = {
            'pca_components': n_components,
            'mean_rmse': np.mean(fold_rmses),
            'std_rmse': np.std(fold_rmses),
            'mean_mae': np.mean(fold_maes),
            'mean_variance': np.mean(variance_explained_list),
            'time_sec': elapsed
        }
        results.append(result)
        
        logger.info(f"  Mean RMSE: ${result['mean_rmse']:,.0f} ± ${result['std_rmse']:,.0f}")
        logger.info(f"  Mean Variance: {result['mean_variance']:.1%}")
    
    # Summary
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_rmse')
    
    logger.info(f"\n{'='*60}")
    logger.info("PCA SWEEP RESULTS")
    logger.info(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    # Best config
    best = results_df.iloc[0]
    logger.info(f"\nBest: PCA={int(best['pca_components'])} with RMSE=${best['mean_rmse']:,.0f}")
    logger.info(f"Variance explained: {best['mean_variance']:.1%}")
    
    # Save results
    output_file = project_root / "pca_sweep_results.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to {output_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Sweep PCA components")
    parser.add_argument("--project-root", type=str, default=".", help="Project root")
    parser.add_argument("--pca-values", type=str, default="50,75,100,150,200",
                        help="Comma-separated PCA values to test")
    parser.add_argument("--n-folds", type=int, default=3, help="Number of CV folds")
    args = parser.parse_args()
    
    pca_values = [int(x) for x in args.pca_values.split(',')]
    run_pca_sweep(args.project_root, pca_values, args.n_folds)


if __name__ == "__main__":
    main()
