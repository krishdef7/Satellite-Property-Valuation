"""
ULTIMATE COMPETITION BASELINE - Maximum Leaderboard Performance

Philosophy: KEEP V5 POWER + FIX INSTABILITY = WIN LEADERBOARD

Key Insights from V3-V6 Experiments:
- V5 Stacked RMSE: $118,945 (BEST raw performance)
- Post-processing DESTROYED performance (clipping +$5.7K, sanity +$15K)
- High fold variance = leaderboard lottery
- Solution: Stabilize V5, NO post-processing

MANDATORY FIXES (from expert analysis):
1. Stratified GroupKFold - balance price distribution across folds
2. Robust neighborhood features - min neighbor requirements + fallbacks
3. Softened luxury specialist - 65/35 blend instead of 70/30
4. Tighter ElasticNet regularization - more stable meta weights
5. NO clipping, NO sanity correction, NO heuristic overrides

Target: ~$119K RMSE with STABLE folds (CV < 25%) = reliable leaderboard

Architecture (unchanged from V5):
- 6 Base Models: LGB, LGB_Diverse, XGB, CatBoost, CatBoost_Diverse, RF
- 1 Specialist: Luxury (with softened blending)
- Meta-learner: ElasticNet (tighter regularization)
- Final = Stacked (NO post-processing)
"""

import sys
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_FILE, TEST_FILE, MODELS_DIR, RESULTS_DIR,
    N_FOLDS, RANDOM_SEED, K_NEIGHBORS_PRICE, set_all_seeds,
    IMAGES_ZOOM_16_DIR, IMAGES_ZOOM_17_DIR, IMAGES_ZOOM_18_DIR,
    SEATTLE_CENTER_LAT, SEATTLE_CENTER_LONG, CURRENT_YEAR
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ANALYSIS_DIR = RESULTS_DIR / "analysis_ultimate"
PLOTS_DIR = ANALYSIS_DIR / "plots"
for d in [ANALYSIS_DIR, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STABILITY FIX #1: Stratified GroupKFold
# ============================================================================

def stratified_group_kfold(df, group_col, target_col, n_splits=5, n_bins=10, random_state=RANDOM_SEED):
    """
    CRITICAL STABILITY FIX: Balance price distribution across folds.
    
    This ensures every fold has:
    - Similar proportion of cheap/mid/luxury homes
    - Similar waterfront distribution
    - No fold gets "lucky" or "unlucky" with price concentration
    
    This is the SINGLE BIGGEST variance reducer.
    """
    np.random.seed(random_state)
    
    # Create price bins for stratification
    bins = pd.qcut(df[target_col], q=n_bins, duplicates='drop', labels=False)
    df_temp = df.copy()
    df_temp['_price_bin'] = bins.fillna(0).astype(int)
    
    groups = df_temp[group_col].values
    bin_ids = df_temp['_price_bin'].values
    
    # Map each group to its bin distribution
    group_bins = defaultdict(lambda: np.zeros(n_bins, dtype=int))
    group_idx = defaultdict(list)
    for i, g in enumerate(groups):
        b = int(bin_ids[i])
        group_bins[g][b] += 1
        group_idx[g].append(i)
    
    # Greedily allocate groups to folds to balance bin distributions
    fold_bins = [np.zeros(n_bins, dtype=int) for _ in range(n_splits)]
    fold_groups = [set() for _ in range(n_splits)]
    
    # Sort groups by size (allocate largest first for better balance)
    group_order = sorted(group_bins.items(), key=lambda x: -x[1].sum())
    
    for g, counts in group_order:
        # Find fold that minimizes imbalance after adding this group
        best_fold = None
        best_score = float('inf')
        
        for f in range(n_splits):
            # Compute imbalance score
            tmp_bins = [fold_bins[i] + (counts if i == f else np.zeros(n_bins)) for i in range(n_splits)]
            
            # Score = variance in fold sizes + variance in bin distributions
            size_var = np.var([b.sum() for b in tmp_bins])
            bin_var = np.mean([np.var([tmp_bins[i][j] for i in range(n_splits)]) for j in range(n_bins)])
            score = size_var + bin_var * 0.5
            
            if score < best_score:
                best_score = score
                best_fold = f
        
        fold_groups[best_fold].add(g)
        fold_bins[best_fold] += counts
    
    # Generate splits
    splits = []
    for f in range(n_splits):
        val_mask = df[group_col].isin(fold_groups[f])
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        splits.append((train_idx, val_idx))
    
    # Log fold balance statistics
    logger.info("Stratified GroupKFold - Fold Balance:")
    for f, (train_idx, val_idx) in enumerate(splits):
        train_prices = df.iloc[train_idx][target_col]
        val_prices = df.iloc[val_idx][target_col]
        val_luxury = df.iloc[val_idx]['is_luxury'].sum() if 'is_luxury' in df.columns else 0
        val_wf = df.iloc[val_idx]['waterfront'].sum() if 'waterfront' in df.columns else 0
        logger.info(f"  Fold {f+1}: Val={len(val_idx)}, Median=${val_prices.median():,.0f}, "
                   f"Luxury={val_luxury}, WF={val_wf}")
    
    return splits


def detect_gpu():
    gpu_status = {'lightgbm_gpu': False, 'xgboost_gpu': False, 'catboost_gpu': False}
    try:
        test_data = xgb.DMatrix([[1,2],[3,4]], label=[1,2])
        xgb.train({'tree_method':'gpu_hist','gpu_id':0}, test_data, num_boost_round=1, verbose_eval=False)
        gpu_status['xgboost_gpu'] = True
    except: pass
    try:
        lgb.train({'device':'gpu','verbose':-1}, lgb.Dataset([[1,2],[3,4]], label=[1,2]), num_boost_round=1)
        gpu_status['lightgbm_gpu'] = True
    except: pass
    try:
        CatBoostRegressor(iterations=1, task_type='GPU', devices='0', verbose=False).fit([[1,2],[3,4]], [1,2])
        gpu_status['catboost_gpu'] = True
    except: pass
    return gpu_status

GPU_STATUS = detect_gpu()


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [np.asarray(lat1), np.asarray(lon1), 
                                               np.asarray(lat2), np.asarray(lon2)])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ============================================================================
# FEATURE ENGINEERING (V5 Architecture - Unchanged)
# ============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features - V5 architecture (proven powerful)."""
    df = df.copy()
    
    for col, fallback in [('sqft_living15', 'sqft_living'), ('sqft_lot15', 'sqft_lot')]:
        if col not in df.columns:
            df[col] = df[fallback]
    
    # Core transforms
    df['log_sqft_living'] = np.log1p(df['sqft_living'])
    df['log_sqft_lot'] = np.log1p(df['sqft_lot'])
    df['log_sqft_above'] = np.log1p(df['sqft_above'])
    
    df['bath_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1)
    df['living_lot_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)
    df['basement_ratio'] = df['sqft_basement'] / (df['sqft_living'] + 1)
    df['above_ratio'] = df['sqft_above'] / (df['sqft_living'] + 1)
    
    df['age'] = CURRENT_YEAR - df['yr_built']
    df['years_since_renovation'] = np.where(df['yr_renovated'] > 0, 
                                            CURRENT_YEAR - df['yr_renovated'], df['age'])
    
    df['age_squared'] = df['age'] ** 2
    df['log_age'] = np.log1p(df['age'])
    
    # Quality interactions (TOP performers)
    df['quality_sqft_interaction'] = df['grade'] * df['log_sqft_living']
    df['quality_sqft_interaction_sq'] = df['quality_sqft_interaction'] ** 2
    df['grade_condition_score'] = df['grade'] * df['condition']
    df['grade_view_interaction'] = df['grade'] * (df['view'] + 1)
    df['avg_floor_sqft'] = df['sqft_living'] / (df['floors'] + 0.5)
    
    # Neighbor comparisons
    df['living_diff_neighbors'] = df['sqft_living'] - df['sqft_living15']
    df['lot_diff_neighbors'] = df['sqft_lot'] - df['sqft_lot15']
    df['living_ratio_neighbors'] = df['sqft_living'] / (df['sqft_living15'] + 1)
    
    # Location
    df['dist_to_seattle_center'] = haversine_distance(
        df['lat'].values, df['long'].values, SEATTLE_CENTER_LAT, SEATTLE_CENTER_LONG)
    df['lat_squared'] = df['lat'] ** 2
    df['long_squared'] = df['long'] ** 2
    
    # Luxury features (CRITICAL)
    df['is_luxury'] = ((df['grade'] >= 10) | (df['waterfront'] == 1) | 
                       (df['sqft_living'] > 4500) | (df['view'] >= 3)).astype(int)
    df['is_ultra_luxury'] = ((df['grade'] >= 12) | 
                             ((df['waterfront'] == 1) & (df['sqft_living'] > 3000)) |
                             (df['sqft_living'] > 7000)).astype(int)
    
    df['luxury_index'] = df['log_sqft_living'] * df['grade'] * (df['view'] + 1) * (1 + df['waterfront'])
    df['luxury_index_log'] = np.log1p(df['luxury_index'])
    df['luxury_index_sq'] = df['luxury_index'] ** 0.5
    
    df['luxury_sqft'] = df['is_luxury'] * df['sqft_living']
    df['luxury_sqft_log'] = df['is_luxury'] * df['log_sqft_living']
    df['luxury_grade'] = df['is_luxury'] * df['grade']
    df['luxury_grade_cond'] = df['is_luxury'] * df['grade'] * df['condition']
    df['luxury_dist'] = df['is_luxury'] * df['dist_to_seattle_center']
    
    df['ultra_luxury_sqft'] = df['is_ultra_luxury'] * df['sqft_living']
    df['ultra_luxury_grade'] = df['is_ultra_luxury'] * df['grade']
    
    # Waterfront
    df['waterfront'] = df['waterfront'].astype(int)
    df['waterfront_grade'] = df['waterfront'] * df['grade']
    df['waterfront_sqft'] = df['waterfront'] * df['sqft_living']
    df['waterfront_sqft_log'] = df['waterfront'] * df['log_sqft_living']
    df['waterfront_grade_sqft'] = df['waterfront'] * df['grade'] * df['log_sqft_living']
    df['waterfront_quality'] = df['waterfront'] * df['grade'] * df['condition']
    
    # View
    df['view_value'] = df['view'] * df['grade']
    df['view_sqft'] = df['view'] * df['sqft_living']
    
    # Size
    df['sqft_living_squared'] = df['sqft_living'] ** 2
    df['log_sqft_living_squared'] = df['log_sqft_living'] ** 2
    df['sqft_percentile'] = df['sqft_living'].rank(pct=True)
    df['lot_percentile'] = df['sqft_lot'].rank(pct=True)
    df['is_large_home'] = (df['sqft_percentile'] > 0.9).astype(int)
    
    # Segment indicators
    df['size_tier'] = pd.cut(df['sqft_living'], bins=[0, 1500, 2500, 4000, 100000], 
                             labels=[0, 1, 2, 3]).astype(float).fillna(1)
    df['grade_tier'] = pd.cut(df['grade'], bins=[0, 6, 8, 10, 15], 
                              labels=[0, 1, 2, 3]).astype(float).fillna(1)
    
    logger.info(f"Created {len(df.columns)} features")
    return df


def create_clusters(df):
    """Create 3-level hierarchical clusters."""
    df = df.copy()
    coords = df[['lat', 'long']].values
    
    kmeans_models = {}
    for n_clusters, level in [(20, 'coarse'), (40, 'medium'), (80, 'fine')]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        df[f'cluster_{level}'] = kmeans.fit_predict(coords)
        kmeans_models[level] = kmeans
        
        centroids = kmeans.cluster_centers_
        idx_vals = df[f'cluster_{level}'].astype(int).values
        df[f'dist_to_centroid_{level}'] = haversine_distance(
            df['lat'].values, df['long'].values,
            centroids[idx_vals, 0], centroids[idx_vals, 1])
    
    logger.info("Created 3-level hierarchical clusters")
    return df, kmeans_models


def compute_property_density(train_df, test_df):
    train_df, test_df = train_df.copy(), test_df.copy()
    all_coords = pd.concat([train_df[['lat', 'long']], test_df[['lat', 'long']]], ignore_index=True)
    tree = BallTree(np.radians(all_coords.values), metric='haversine')
    
    radius_rad = 1.0 / 6371.0
    train_df['density_1km'] = tree.query_radius(
        np.radians(train_df[['lat', 'long']].values), r=radius_rad, count_only=True) - 1
    test_df['density_1km'] = tree.query_radius(
        np.radians(test_df[['lat', 'long']].values), r=radius_rad, count_only=True) - 1
    
    return train_df, test_df


def add_image_features(df):
    """Add satellite image features with robust fallbacks."""
    from data_fetcher import compute_image_stats
    
    df = df.copy()
    feature_names = ['green_dominance', 'brightness', 'contrast']
    
    for zoom in [16, 17, 18]:
        for feat in feature_names:
            df[f'{feat}_z{zoom}'] = np.nan
    
    df['has_images'] = False
    zoom_dirs = {16: IMAGES_ZOOM_16_DIR, 17: IMAGES_ZOOM_17_DIR, 18: IMAGES_ZOOM_18_DIR}
    
    count = 0
    for idx, row in df.iterrows():
        prop_id = str(row['id'])
        all_exist = True
        for zoom, zoom_dir in zoom_dirs.items():
            img_path = zoom_dir / f"{prop_id}_z{zoom}.jpg"
            if img_path.exists():
                stats = compute_image_stats(img_path)
                if stats:
                    for feat in feature_names:
                        if feat in stats:
                            df.loc[idx, f'{feat}_z{zoom}'] = stats[feat]
            else:
                all_exist = False
        if all_exist:
            count += 1
        df.loc[idx, 'has_images'] = all_exist
    
    # Robust fallbacks
    for zoom in [16, 17, 18]:
        for feat in feature_names:
            col = f'{feat}_z{zoom}'
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.5 if 'dominance' in col else 128 if 'brightness' in col else 50
            df[col].fillna(median_val, inplace=True)
    
    df['green_diff_z17_z16'] = df['green_dominance_z17'] - df['green_dominance_z16']
    df['brightness_avg'] = (df['brightness_z16'] + df['brightness_z17'] + df['brightness_z18']) / 3
    
    logger.info(f"Added image features for {count}/{len(df)} properties")
    return df


# ============================================================================
# STABILITY FIX #2: Robust Neighborhood Features
# ============================================================================

def compute_neighborhood_features_robust(df, train_idx, val_idx, k=K_NEIGHBORS_PRICE, min_neighbors=5):
    """
    STABILITY FIX: Robust neighborhood features with minimum neighbor requirements.
    
    When neighbors are sparse:
    - Reduce trust in neighborhood features
    - Fall back to cluster-level statistics
    - Apply gentle smoothing to prevent extreme values
    
    This prevents model from chasing noise in sparse areas.
    """
    train_coords = df.loc[train_idx, ['lat', 'long']].values
    train_prices = df.loc[train_idx, 'price'].values
    train_sqft = df.loc[train_idx, 'sqft_living'].values
    train_grade = df.loc[train_idx, 'grade'].values
    train_waterfront = df.loc[train_idx, 'waterfront'].values
    train_ppsf = train_prices / (train_sqft + 1)
    train_is_luxury = df.loc[train_idx, 'is_luxury'].values
    
    # Global fallbacks for sparse neighborhoods
    global_avg_price = np.mean(train_prices)
    global_median_price = np.median(train_prices)
    global_ppsf = np.mean(train_ppsf)
    global_std = np.std(train_prices)
    
    knn = NearestNeighbors(n_neighbors=k, metric='haversine', algorithm='ball_tree')
    knn.fit(np.radians(train_coords))
    
    all_idx = np.concatenate([train_idx, val_idx])
    all_coords = df.loc[all_idx, ['lat', 'long']].values
    distances, indices = knn.kneighbors(np.radians(all_coords))
    
    feature_names = [
        'local_avg_price', 'local_weighted_price', 'local_median_price',
        'local_max_price', 'local_min_price', 'local_price_range', 'local_price_std',
        'local_price_p75', 'local_price_p90', 'local_price_p95',
        'local_price_ppsf', 'local_weighted_ppsf',
        'local_sqft_ratio', 'local_avg_grade', 'local_grade_premium',
        'local_luxury_ratio', 'local_waterfront_ratio',
        'local_top1_price', 'local_top3_avg_price', 'local_top5_avg_price',
        'local_neighbor_count', 'local_reliability'  # NEW: track reliability
    ]
    features = {name: pd.Series(index=all_idx, dtype=float) for name in feature_names}
    
    for i, idx in enumerate(all_idx):
        neighbor_idx = indices[i]
        neighbor_dist = distances[i]
        
        # Remove self from neighbors if present
        if idx in train_idx and len(neighbor_dist) > 0 and neighbor_dist[0] < 1e-10:
            neighbor_idx, neighbor_dist = neighbor_idx[1:], neighbor_dist[1:]
        
        n_neighbors = len(neighbor_idx)
        
        if n_neighbors == 0:
            # No neighbors - use global fallbacks
            features['local_avg_price'].loc[idx] = global_avg_price
            features['local_weighted_price'].loc[idx] = global_avg_price
            features['local_median_price'].loc[idx] = global_median_price
            features['local_price_ppsf'].loc[idx] = global_ppsf
            features['local_neighbor_count'].loc[idx] = 0
            features['local_reliability'].loc[idx] = 0.0
            continue
        
        neighbor_prices = train_prices[neighbor_idx]
        neighbor_ppsf = train_ppsf[neighbor_idx]
        neighbor_sqft = train_sqft[neighbor_idx]
        neighbor_grade = train_grade[neighbor_idx]
        neighbor_waterfront = train_waterfront[neighbor_idx]
        neighbor_luxury = train_is_luxury[neighbor_idx]
        
        # Distance-based weights
        safe_dist = np.maximum(neighbor_dist, 1e-3)
        weights = 1 / safe_dist
        weights = weights / (weights.sum() + 1e-10)
        
        # STABILITY: Compute reliability score based on neighbor count and distance spread
        reliability = min(n_neighbors / min_neighbors, 1.0)  # 1.0 if >= min_neighbors
        avg_dist_km = np.mean(neighbor_dist) * 6371  # Convert to km
        if avg_dist_km > 2.0:  # If neighbors are far, reduce reliability
            reliability *= 0.8
        
        # Compute raw features
        raw_avg = np.mean(neighbor_prices)
        raw_weighted = np.sum(weights * neighbor_prices)
        raw_median = np.median(neighbor_prices)
        
        # STABILITY: Blend with global based on reliability
        features['local_avg_price'].loc[idx] = reliability * raw_avg + (1 - reliability) * global_avg_price
        features['local_weighted_price'].loc[idx] = reliability * raw_weighted + (1 - reliability) * global_avg_price
        features['local_median_price'].loc[idx] = reliability * raw_median + (1 - reliability) * global_median_price
        
        features['local_max_price'].loc[idx] = np.max(neighbor_prices)
        features['local_min_price'].loc[idx] = np.min(neighbor_prices)
        features['local_price_range'].loc[idx] = np.ptp(neighbor_prices)
        features['local_price_std'].loc[idx] = np.std(neighbor_prices) if n_neighbors > 1 else global_std
        features['local_price_p75'].loc[idx] = np.percentile(neighbor_prices, 75)
        features['local_price_p90'].loc[idx] = np.percentile(neighbor_prices, 90)
        features['local_price_p95'].loc[idx] = np.percentile(neighbor_prices, 95)
        
        raw_ppsf = np.mean(neighbor_ppsf)
        features['local_price_ppsf'].loc[idx] = reliability * raw_ppsf + (1 - reliability) * global_ppsf
        features['local_weighted_ppsf'].loc[idx] = np.sum(weights * neighbor_ppsf)
        
        features['local_sqft_ratio'].loc[idx] = df.loc[idx, 'sqft_living'] / (np.mean(neighbor_sqft) + 1)
        features['local_avg_grade'].loc[idx] = np.mean(neighbor_grade)
        features['local_grade_premium'].loc[idx] = (df.loc[idx, 'grade'] - np.mean(neighbor_grade)) * raw_avg
        features['local_luxury_ratio'].loc[idx] = np.mean(neighbor_luxury)
        features['local_waterfront_ratio'].loc[idx] = np.mean(neighbor_waterfront)
        
        sorted_prices = np.sort(neighbor_prices)[::-1]
        features['local_top1_price'].loc[idx] = sorted_prices[0]
        features['local_top3_avg_price'].loc[idx] = np.mean(sorted_prices[:min(3, len(sorted_prices))])
        features['local_top5_avg_price'].loc[idx] = np.mean(sorted_prices[:min(5, len(sorted_prices))])
        
        features['local_neighbor_count'].loc[idx] = n_neighbors
        features['local_reliability'].loc[idx] = reliability
    
    return features


def compute_cluster_features(df, train_idx, val_idx):
    """Compute cluster-based features (fallback for sparse neighborhoods)."""
    features = {}
    
    for level in ['medium', 'fine']:
        cluster_col = f'cluster_{level}'
        train_data = df.loc[train_idx].copy()
        train_data['ppsf'] = train_data['price'] / (train_data['sqft_living'] + 1)
        
        train_stats = train_data.groupby(cluster_col).agg({
            'price': ['mean', 'std', 'median', 'count'],
            'ppsf': 'mean'
        })
        train_stats.columns = ['price_mean', 'price_std', 'price_median', 'count', 'ppsf_mean']
        
        p90 = train_data.groupby(cluster_col)['price'].quantile(0.90)
        train_stats['price_p90'] = p90
        
        all_idx = np.concatenate([train_idx, val_idx])
        result_df = df.loc[all_idx, [cluster_col]].copy()
        result_df = result_df.join(train_stats, on=cluster_col)
        
        overall_mean = train_data['price'].mean()
        overall_std = train_data['price'].std()
        overall_ppsf = train_data['ppsf'].mean()
        
        result_df['price_mean'].fillna(overall_mean, inplace=True)
        result_df['price_std'].fillna(overall_std, inplace=True)
        result_df['price_median'].fillna(overall_mean, inplace=True)
        result_df['count'].fillna(1, inplace=True)
        result_df['ppsf_mean'].fillna(overall_ppsf, inplace=True)
        result_df['price_p90'].fillna(overall_mean, inplace=True)
        
        features[f'cluster_avg_price_{level}'] = result_df['price_mean']
        features[f'cluster_price_std_{level}'] = result_df['price_std']
        features[f'cluster_median_price_{level}'] = result_df['price_median']
        features[f'cluster_count_{level}'] = result_df['count']
        features[f'cluster_ppsf_{level}'] = result_df['ppsf_mean']
        features[f'cluster_price_p90_{level}'] = result_df['price_p90']
    
    return features


def add_target_encoding(df, train_idx, val_idx, smooth=15.0):
    """Target encoding for clusters."""
    features = {}
    global_mean = df.loc[train_idx, 'price'].mean()
    
    for col in ['cluster_medium', 'cluster_fine']:
        stats = df.loc[train_idx].groupby(col)['price'].agg(['mean', 'count'])
        stats['smooth_mean'] = (stats['count'] * stats['mean'] + smooth * global_mean) / (stats['count'] + smooth)
        
        all_idx = np.concatenate([train_idx, val_idx])
        result = df.loc[all_idx, col].map(stats['smooth_mean']).fillna(global_mean)
        features[f'{col}_target_enc'] = result
    
    return features


# ============================================================================
# MODEL PARAMETERS (V5 Configuration)
# ============================================================================

def get_lgb_params(model_type='main'):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'verbose': -1, 'seed': RANDOM_SEED, 'n_jobs': -1, 'force_col_wise': True
    }
    if GPU_STATUS.get('lightgbm_gpu'):
        params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
    
    if model_type == 'main':
        params.update({
            'learning_rate': 0.015,
            'num_leaves': 50,
            'max_depth': 10,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 5,
            'lambda_l1': 0.3,
            'lambda_l2': 1.0
        })
    elif model_type == 'diverse':
        params.update({
            'learning_rate': 0.025,
            'num_leaves': 35,
            'max_depth': 7,
            'min_data_in_leaf': 25,
            'feature_fraction': 0.65,
            'bagging_fraction': 0.7,
            'bagging_freq': 3,
            'lambda_l1': 0.5,
            'lambda_l2': 0.8
        })
    elif model_type == 'luxury':
        params.update({
            'learning_rate': 0.012,
            'num_leaves': 45,
            'max_depth': 9,
            'min_data_in_leaf': 12,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 3,
            'lambda_l1': 0.1,
            'lambda_l2': 0.5
        })
    return params


def get_xgb_params():
    params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'random_state': RANDOM_SEED, 'n_jobs': -1,
        'max_depth': 9,
        'learning_rate': 0.015,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.3,
        'reg_lambda': 1.0,
        'min_child_weight': 8
    }
    if GPU_STATUS.get('xgboost_gpu'):
        params.update({'tree_method': 'gpu_hist', 'gpu_id': 0})
    else:
        params['tree_method'] = 'hist'
    return params


def get_cat_params(model_type='main'):
    params = {'loss_function': 'RMSE', 'random_seed': RANDOM_SEED, 'verbose': False}
    if GPU_STATUS.get('catboost_gpu'):
        params.update({'task_type': 'GPU', 'devices': '0'})
    else:
        params.update({'task_type': 'CPU', 'thread_count': -1})
    
    if model_type == 'main':
        params.update({
            'learning_rate': 0.015,
            'depth': 9,
            'l2_leaf_reg': 8,
            'min_data_in_leaf': 20
        })
    elif model_type == 'diverse':
        params.update({
            'learning_rate': 0.025,
            'depth': 7,
            'l2_leaf_reg': 12,
            'min_data_in_leaf': 30
        })
    return params


def compute_sample_weights(prices, is_luxury, is_waterfront):
    """Moderate sample weighting."""
    prices = np.asarray(prices, dtype=float)
    is_luxury = np.asarray(is_luxury, dtype=int)
    is_waterfront = np.asarray(is_waterfront, dtype=int)
    
    weights = 1.0 / np.sqrt(prices + 1)
    weights = weights / weights.mean()
    
    weights[is_luxury == 1] *= 2.0
    weights[is_waterfront == 1] *= 2.5
    weights[prices > 1000000] *= 1.3
    weights[prices > 2000000] *= 1.2
    weights[prices < 300000] *= 1.1
    
    weights = np.clip(weights, 0.5, 3.0)
    
    return weights


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_base_models(X_train, y_train, X_val, y_val, feature_names, 
                      train_prices, train_is_luxury, train_is_waterfront):
    """Train 6 base models."""
    results = {}
    weights = compute_sample_weights(train_prices, train_is_luxury, train_is_waterfront)
    
    # LightGBM Main
    lgb_params = get_lgb_params('main')
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
    
    lgb_model = lgb.train(lgb_params, train_data, num_boost_round=4000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    results['lgb_model'] = lgb_model
    results['lgb_pred'] = np.expm1(lgb_model.predict(X_val))
    
    # LightGBM Diverse
    lgb_params_div = get_lgb_params('diverse')
    lgb_model_div = lgb.train(lgb_params_div, train_data, num_boost_round=3000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    results['lgb_div_model'] = lgb_model_div
    results['lgb_div_pred'] = np.expm1(lgb_model_div.predict(X_val))
    
    # XGBoost
    xgb_params = get_xgb_params()
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=4000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=200, verbose_eval=False)
    results['xgb_model'] = xgb_model
    results['xgb_pred'] = np.expm1(xgb_model.predict(dval))
    
    # CatBoost Main
    cat_params = get_cat_params('main')
    train_pool = Pool(X_train, y_train, feature_names=feature_names, weight=weights)
    val_pool = Pool(X_val, y_val, feature_names=feature_names)
    cat_model = CatBoostRegressor(**cat_params, iterations=4000, early_stopping_rounds=200)
    cat_model.fit(train_pool, eval_set=val_pool, verbose=False)
    results['cat_model'] = cat_model
    results['cat_pred'] = np.expm1(cat_model.predict(X_val))
    
    # CatBoost Diverse
    cat_params_div = get_cat_params('diverse')
    train_pool_div = Pool(X_train, y_train, feature_names=feature_names)
    cat_model_div = CatBoostRegressor(**cat_params_div, iterations=3000, early_stopping_rounds=150)
    cat_model_div.fit(train_pool_div, eval_set=val_pool, verbose=False)
    results['cat_div_model'] = cat_model_div
    results['cat_div_pred'] = np.expm1(cat_model_div.predict(X_val))
    
    # RandomForest
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=10,
        max_features=0.5,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    rf_model.fit(X_train, y_train, sample_weight=weights)
    results['rf_model'] = rf_model
    results['rf_pred'] = np.expm1(rf_model.predict(X_val))
    
    return results


def train_luxury_specialist(X_train, y_train, X_val, y_val, feature_names,
                            train_is_luxury, train_prices):
    """Train luxury specialist model."""
    luxury_mask = (train_is_luxury == 1) | (train_prices > 750000)
    
    if luxury_mask.sum() < 300:
        logger.warning(f"Luxury specialist: only {luxury_mask.sum()} samples")
        return None, None
    
    X_lux = X_train[luxury_mask]
    y_lux = y_train[luxury_mask]
    
    lux_params = get_lgb_params('luxury')
    train_data = lgb.Dataset(X_lux, label=y_lux, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
    
    lux_model = lgb.train(lux_params, train_data, num_boost_round=3500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(175, verbose=False), lgb.log_evaluation(0)])
    
    lux_pred = np.expm1(lux_model.predict(X_val))
    return lux_model, lux_pred


# ============================================================================
# STABILITY FIX #3: Softened Luxury Blending (65/35 instead of 70/30)
# ============================================================================

def blend_with_luxury_softened(base_pred, lux_pred, is_luxury, local_reliability=None):
    """
    STABILITY FIX: Softer luxury blending to prevent over-pull.
    
    - Default: 65% base + 35% luxury (was 70/30)
    - If neighborhood is sparse (low reliability), reduce luxury influence further
    - Prevents luxury specialist from hijacking predictions
    """
    if lux_pred is None:
        return base_pred
    
    blended = base_pred.copy()
    lux_mask = is_luxury == 1
    
    if lux_mask.sum() > 0:
        # Base blend: 65/35 (softer than 70/30)
        base_weight = 0.65
        lux_weight = 0.35
        
        # If we have reliability scores, adjust blend
        if local_reliability is not None:
            # In sparse areas (low reliability), trust base more
            reliability = local_reliability[lux_mask]
            adjusted_lux_weight = lux_weight * reliability
            adjusted_base_weight = 1 - adjusted_lux_weight
            
            blended[lux_mask] = (adjusted_base_weight * base_pred[lux_mask] + 
                                 adjusted_lux_weight * lux_pred[lux_mask])
        else:
            blended[lux_mask] = base_weight * base_pred[lux_mask] + lux_weight * lux_pred[lux_mask]
    
    return blended


# ============================================================================
# STABILITY FIX #4: Tighter ElasticNet Regularization
# ============================================================================

def train_meta_learner_stable(base_predictions, y_true, is_luxury, is_waterfront, price_percentile):
    """
    STABILITY FIX: Tighter regularization for more stable meta weights.
    
    - Higher alpha (0.01 instead of 0.001) = more regularization
    - Prevents extreme reliance on single model
    - More stable across folds
    """
    meta_features = np.column_stack([
        base_predictions,
        is_luxury.reshape(-1, 1),
        is_waterfront.reshape(-1, 1),
        price_percentile.reshape(-1, 1)
    ])
    
    # TIGHTER regularization (alpha=0.01 instead of 0.001)
    meta_model = ElasticNet(
        alpha=0.01,  # 10x stronger regularization
        l1_ratio=0.5,
        random_state=RANDOM_SEED,
        max_iter=5000
    )
    meta_model.fit(meta_features, y_true)
    
    return meta_model


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_feature_importance(models_dict, feature_names):
    lgb_imp = np.mean([m.feature_importance(importance_type='gain') for m in models_dict['lgb']], axis=0)
    lgb_div_imp = np.mean([m.feature_importance(importance_type='gain') for m in models_dict['lgb_div']], axis=0)
    
    xgb_scores = {f: [] for f in feature_names}
    for model in models_dict['xgb']:
        score = model.get_score(importance_type='gain')
        mapped = {}
        for k, v in score.items():
            if k.startswith('f') and k[1:].isdigit():
                idx = int(k[1:])
                if idx < len(feature_names):
                    mapped[feature_names[idx]] = v
            else:
                mapped[k] = v
        for f in feature_names:
            xgb_scores[f].append(mapped.get(f, 0))
    xgb_imp = np.array([np.mean(xgb_scores[f]) for f in feature_names])
    
    cat_imp = np.mean([m.get_feature_importance() for m in models_dict['cat']], axis=0)
    cat_div_imp = np.mean([m.get_feature_importance() for m in models_dict['cat_div']], axis=0)
    rf_imp = np.mean([m.feature_importances_ for m in models_dict['rf']], axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'lgb': lgb_imp, 'lgb_div': lgb_div_imp, 'xgb': xgb_imp,
        'cat': cat_imp, 'cat_div': cat_div_imp, 'rf': rf_imp
    })
    
    for col in ['lgb', 'lgb_div', 'xgb', 'cat', 'cat_div', 'rf']:
        importance_df[f'{col}_pct'] = importance_df[col] / (importance_df[col].sum() + 1e-6) * 100
    
    importance_df['avg_importance'] = importance_df[['lgb_pct', 'lgb_div_pct', 'xgb_pct', 
                                                      'cat_pct', 'cat_div_pct', 'rf_pct']].mean(axis=1)
    importance_df = importance_df.sort_values('avg_importance', ascending=False)
    
    return importance_df


def analyze_residuals(train_df, oof_predictions):
    preds = oof_predictions['final']
    
    df = pd.DataFrame({
        'price': train_df['price'], 'pred': preds,
        'error': preds - train_df['price'],
        'abs_error': np.abs(preds - train_df['price']),
        'pct_error': np.abs(preds - train_df['price']) / train_df['price'] * 100,
        'price_segment': pd.cut(train_df['price'], bins=[0, 300000, 500000, 750000, 1000000, np.inf],
                                labels=['<300K', '300-500K', '500-750K', '750K-1M', '>1M']),
        'is_luxury': train_df['is_luxury'],
        'is_waterfront': train_df['waterfront'],
        'cluster': train_df['cluster_medium']
    })
    
    seg_stats = df.groupby('price_segment').agg({
        'abs_error': ['mean', 'median'], 'pct_error': ['mean', 'median'],
        'error': 'mean', 'price': 'count'
    })
    seg_stats.columns = ['mae', 'med_error', 'mape', 'med_pct', 'bias', 'count']
    seg_stats.to_csv(ANALYSIS_DIR / 'residuals_by_segment.csv')
    
    lux_stats = df.groupby('is_luxury').agg({
        'abs_error': ['mean', 'median'], 'error': 'mean', 'price': ['mean', 'count']
    })
    lux_stats.columns = ['mae', 'med_error', 'bias', 'avg_price', 'count']
    lux_stats.to_csv(ANALYSIS_DIR / 'residuals_by_luxury.csv')
    
    wf_stats = df.groupby('is_waterfront').agg({
        'abs_error': ['mean', 'median'], 'error': 'mean', 'price': ['mean', 'count']
    })
    wf_stats.columns = ['mae', 'med_error', 'bias', 'avg_price', 'count']
    wf_stats.to_csv(ANALYSIS_DIR / 'residuals_by_waterfront.csv')
    
    return seg_stats, lux_stats, wf_stats


def create_diagnostic_plots(train_df, oof_predictions, feature_importance, fold_rmses):
    preds = oof_predictions['final']
    actual = train_df['price'].values
    residuals = preds - actual
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Actual vs Predicted
    axes[0,0].scatter(actual, preds, alpha=0.3, s=8)
    axes[0,0].plot([0, actual.max()], [0, actual.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Actual'); axes[0,0].set_ylabel('Predicted')
    axes[0,0].set_title('Actual vs Predicted')
    
    # Residuals histogram
    axes[0,1].hist(residuals, bins=100, alpha=0.7)
    axes[0,1].axvline(0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Residual')
    axes[0,1].set_title(f'Residuals (mean={residuals.mean():.0f})')
    
    # Fold stability
    axes[0,2].bar(range(1, len(fold_rmses)+1), fold_rmses)
    axes[0,2].axhline(np.mean(fold_rmses), color='r', linestyle='--', label=f'Mean: ${np.mean(fold_rmses):,.0f}')
    axes[0,2].set_xlabel('Fold')
    axes[0,2].set_ylabel('RMSE')
    fold_cv = np.std(fold_rmses) / np.mean(fold_rmses) * 100
    axes[0,2].set_title(f'Fold Stability (CV={fold_cv:.1f}%)')
    axes[0,2].legend()
    
    # % Error distribution
    pct_err = np.abs(residuals) / actual * 100
    axes[1,0].hist(pct_err[pct_err < 100], bins=50, alpha=0.7)
    axes[1,0].axvline(np.median(pct_err), color='r', linestyle='--')
    axes[1,0].set_xlabel('% Error')
    axes[1,0].set_title(f'% Error (median={np.median(pct_err):.1f}%)')
    
    # Error by luxury
    is_lux = train_df['is_luxury'].values
    axes[1,1].boxplot([np.abs(residuals[is_lux==0]), np.abs(residuals[is_lux==1])],
                      labels=['Non-Luxury', 'Luxury'])
    axes[1,1].set_ylabel('Absolute Error')
    axes[1,1].set_title('Error by Luxury Status')
    
    # Feature importance (top 15)
    top = feature_importance.head(15)
    axes[1,2].barh(range(len(top)), top['avg_importance'].values)
    axes[1,2].set_yticks(range(len(top)))
    axes[1,2].set_yticklabels(top['feature'].values)
    axes[1,2].invert_yaxis()
    axes[1,2].set_xlabel('Avg Importance (%)')
    axes[1,2].set_title('Top 15 Features')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'diagnostic_plots.png', dpi=150)
    plt.close()


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def prepare_data():
    logger.info("Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    logger.info("Creating features...")
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    logger.info("Creating clusters...")
    train_df, kmeans_models = create_clusters(train_df)
    
    for level, kmeans in kmeans_models.items():
        centroids = kmeans.cluster_centers_
        test_df[f'cluster_{level}'] = kmeans.predict(test_df[['lat', 'long']].values)
        idx_vals = test_df[f'cluster_{level}'].astype(int).values
        test_df[f'dist_to_centroid_{level}'] = haversine_distance(
            test_df['lat'].values, test_df['long'].values,
            centroids[idx_vals, 0], centroids[idx_vals, 1])
    
    logger.info("Computing density...")
    train_df, test_df = compute_property_density(train_df, test_df)
    
    logger.info("Adding image features...")
    train_df = add_image_features(train_df)
    test_df = add_image_features(test_df)
    
    train_df['log_price'] = np.log1p(train_df['price'])
    
    return train_df, test_df, kmeans_models


def train_stacked_ensemble(train_df, n_folds=N_FOLDS):
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING ULTIMATE BASELINE - {n_folds}-FOLD CV")
    logger.info(f"{'='*60}")
    logger.info("STABILITY FIXES ENABLED:")
    logger.info("  1. Stratified GroupKFold (balanced price distribution)")
    logger.info("  2. Robust neighborhood features (min neighbors + fallbacks)")
    logger.info("  3. Softened luxury blend (65/35 instead of 70/30)")
    logger.info("  4. Tighter ElasticNet (alpha=0.01)")
    logger.info("  5. NO post-processing (no clipping, no sanity)")
    
    exclude_cols = {'id', 'date', 'price', 'log_price', 'has_images', 'zipcode',
                    'yr_built', 'yr_renovated', 'lat', 'long'}
    base_features = [col for col in train_df.columns if col not in exclude_cols 
                     and not col.startswith('cluster_')]
    
    logger.info(f"Using {len(base_features)} base features")
    
    # STABILITY FIX #1: Use stratified GroupKFold
    logger.info("\nCreating STRATIFIED GroupKFold splits:")
    splits = stratified_group_kfold(train_df, 'cluster_medium', 'price', n_splits=n_folds)
    
    all_models = {
        'lgb': [], 'lgb_div': [], 'xgb': [], 'cat': [], 'cat_div': [], 'rf': [],
        'lux': [], 'meta': []
    }
    oof_predictions = {
        'lgb': np.zeros(len(train_df)), 'lgb_div': np.zeros(len(train_df)),
        'xgb': np.zeros(len(train_df)), 'cat': np.zeros(len(train_df)),
        'cat_div': np.zeros(len(train_df)), 'rf': np.zeros(len(train_df)),
        'lux': np.zeros(len(train_df)), 'blended': np.zeros(len(train_df)),
        'stacked': np.zeros(len(train_df)), 'final': np.zeros(len(train_df))
    }
    
    fold_rmses = []
    fold_stats = []
    feature_names_final = None
    
    # Track fold assignments for fusion alignment (ChatGPT fix)
    fold_assignments = np.zeros(len(train_df), dtype=int)
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        # Record which fold each validation sample belongs to
        fold_assignments[val_idx] = fold
        
        logger.info(f"\n{'='*50}")
        logger.info(f"FOLD {fold+1}/{n_folds} - Train: {len(train_idx)}, Val: {len(val_idx)}")
        logger.info(f"{'='*50}")
        
        # STABILITY FIX #2: Robust neighborhood features
        nbr_features = compute_neighborhood_features_robust(train_df, train_idx, val_idx)
        cluster_features = compute_cluster_features(train_df, train_idx, val_idx)
        target_enc = add_target_encoding(train_df, train_idx, val_idx)
        
        df_fold = train_df.copy()
        for name, series in {**nbr_features, **cluster_features, **target_enc}.items():
            df_fold.loc[series.index, name] = series.values
        
        # Derived features
        df_fold['log_local_avg_price'] = np.log1p(df_fold['local_avg_price'].fillna(0))
        df_fold['log_local_median_price'] = np.log1p(df_fold['local_median_price'].fillna(0))
        df_fold['local_price_premium'] = df_fold['local_price_p90'] / (df_fold['local_avg_price'] + 1)
        df_fold['local_grade_vs_price'] = df_fold['local_avg_grade'] * df_fold['log_local_avg_price']
        df_fold['luxury_local_premium'] = df_fold['is_luxury'] * df_fold['local_price_p95']
        df_fold['waterfront_local'] = df_fold['waterfront'] * df_fold['local_price_p90']
        df_fold['ultra_local_premium'] = df_fold['is_ultra_luxury'] * df_fold['local_price_p95']
        df_fold['luxury_top_neighbor'] = df_fold['is_luxury'] * df_fold['local_top3_avg_price']
        
        all_features = base_features + list(nbr_features.keys()) + list(cluster_features.keys()) + \
                      list(target_enc.keys()) + [
                          'log_local_avg_price', 'log_local_median_price', 'local_price_premium',
                          'local_grade_vs_price', 'luxury_local_premium', 'waterfront_local',
                          'ultra_local_premium', 'luxury_top_neighbor'
                      ]
        all_features = [f for f in all_features if f in df_fold.columns]
        all_features = list(dict.fromkeys(all_features))
        feature_names_final = all_features
        
        logger.info(f"Total features: {len(all_features)}")
        
        X_train = np.nan_to_num(df_fold.loc[train_idx, all_features].values.astype(np.float32), nan=0)
        X_val = np.nan_to_num(df_fold.loc[val_idx, all_features].values.astype(np.float32), nan=0)
        y_train = df_fold.loc[train_idx, 'log_price'].values
        y_val = df_fold.loc[val_idx, 'log_price'].values
        y_val_real = df_fold.loc[val_idx, 'price'].values
        
        train_prices = df_fold.loc[train_idx, 'price'].values
        train_is_luxury = df_fold.loc[train_idx, 'is_luxury'].values.astype(int)
        train_is_waterfront = df_fold.loc[train_idx, 'waterfront'].values.astype(int)
        val_is_luxury = df_fold.loc[val_idx, 'is_luxury'].values.astype(int)
        val_is_waterfront = df_fold.loc[val_idx, 'waterfront'].values.astype(int)
        
        # Get reliability scores for blending
        val_reliability = df_fold.loc[val_idx, 'local_reliability'].values if 'local_reliability' in df_fold.columns else None
        
        # ==================== TRAIN BASE MODELS ====================
        base_results = train_base_models(
            X_train, y_train, X_val, y_val, all_features,
            train_prices, train_is_luxury, train_is_waterfront
        )
        
        for model_name in ['lgb', 'lgb_div', 'xgb', 'cat', 'cat_div', 'rf']:
            oof_predictions[model_name][val_idx] = base_results[f'{model_name}_pred']
            all_models[model_name].append(base_results[f'{model_name}_model'])
        
        # ==================== TRAIN LUXURY SPECIALIST ====================
        lux_model, lux_pred = train_luxury_specialist(
            X_train, y_train, X_val, y_val, all_features,
            train_is_luxury, train_prices
        )
        all_models['lux'].append(lux_model)
        if lux_pred is not None:
            oof_predictions['lux'][val_idx] = lux_pred
        
        # ==================== BLEND PREDICTIONS ====================
        base_pred_matrix = np.column_stack([
            base_results['lgb_pred'], base_results['lgb_div_pred'],
            base_results['xgb_pred'], base_results['cat_pred'],
            base_results['cat_div_pred'], base_results['rf_pred']
        ])
        avg_base = base_pred_matrix.mean(axis=1)
        
        # STABILITY FIX #3: Softened luxury blending
        blended_pred = blend_with_luxury_softened(avg_base, lux_pred, val_is_luxury, val_reliability)
        oof_predictions['blended'][val_idx] = blended_pred
        
        # ==================== META LEARNER ====================
        final_preds = np.column_stack([base_pred_matrix, blended_pred.reshape(-1, 1)])
        price_percentile = pd.Series(y_val_real).rank(pct=True).values
        
        # STABILITY FIX #4: Tighter ElasticNet regularization
        meta_model = train_meta_learner_stable(final_preds, y_val_real, val_is_luxury, 
                                               val_is_waterfront, price_percentile)
        all_models['meta'].append(meta_model)
        
        meta_features = np.column_stack([
            final_preds,
            val_is_luxury.reshape(-1, 1),
            val_is_waterfront.reshape(-1, 1),
            price_percentile.reshape(-1, 1)
        ])
        stacked_pred = meta_model.predict(meta_features)
        oof_predictions['stacked'][val_idx] = stacked_pred
        
        # ==================== FINAL = STACKED (NO POST-PROCESSING) ====================
        # STABILITY FIX #5: NO clipping, NO sanity correction, NO heuristic overrides
        final_pred = stacked_pred
        oof_predictions['final'][val_idx] = final_pred
        
        # ==================== FOLD METRICS ====================
        lgb_rmse = np.sqrt(mean_squared_error(y_val_real, base_results['lgb_pred']))
        stacked_rmse = np.sqrt(mean_squared_error(y_val_real, stacked_pred))
        final_rmse = np.sqrt(mean_squared_error(y_val_real, final_pred))
        fold_rmses.append(final_rmse)
        
        lux_mask = val_is_luxury == 1
        non_lux_mask = ~lux_mask
        wf_mask = val_is_waterfront == 1
        
        lux_mae = np.abs(y_val_real[lux_mask] - final_pred[lux_mask]).mean() if lux_mask.sum() > 0 else 0
        non_lux_mae = np.abs(y_val_real[non_lux_mask] - final_pred[non_lux_mask]).mean()
        wf_mae = np.abs(y_val_real[wf_mask] - final_pred[wf_mask]).mean() if wf_mask.sum() > 0 else 0
        
        fold_stats.append({
            'fold': fold + 1, 'lgb_rmse': lgb_rmse, 'stacked_rmse': stacked_rmse,
            'final_rmse': final_rmse, 'lux_mae': lux_mae, 'non_lux_mae': non_lux_mae, 'wf_mae': wf_mae
        })
        
        logger.info(f"\nFold {fold+1} Results:")
        logger.info(f"  LGB RMSE: ${lgb_rmse:,.0f}")
        logger.info(f"  Stacked RMSE: ${stacked_rmse:,.0f}")
        logger.info(f"  Final RMSE: ${final_rmse:,.0f}")
        logger.info(f"  Non-Luxury MAE: ${non_lux_mae:,.0f}")
        logger.info(f"  Luxury MAE: ${lux_mae:,.0f}")
        logger.info(f"  Waterfront MAE: ${wf_mae:,.0f}")
    
    # ==================== FINAL ANALYSIS ====================
    logger.info(f"\n{'='*60}")
    logger.info("ULTIMATE BASELINE - FINAL ANALYSIS")
    logger.info(f"{'='*60}")
    
    fold_mean = np.mean(fold_rmses)
    fold_std = np.std(fold_rmses)
    fold_cv = fold_std / fold_mean * 100
    
    logger.info(f"\n🎯 FOLD STABILITY (TARGET: CV < 25%):")
    logger.info(f"  RMSEs: {[f'${x:,.0f}' for x in fold_rmses]}")
    logger.info(f"  Mean: ${fold_mean:,.0f}, Std: ${fold_std:,.0f}")
    logger.info(f"  CV: {fold_cv:.1f}%")
    
    if fold_cv < 25:
        logger.info(f"  ✓ STABLE (CV < 25%) - Good for leaderboard!")
    else:
        logger.warning(f"  ⚠ HIGH VARIANCE (CV >= 25%) - Leaderboard risk!")
    
    logger.info("\n📊 OVERALL CV RESULTS:")
    for name, preds in oof_predictions.items():
        if np.any(preds != 0):
            rmse = np.sqrt(mean_squared_error(train_df['price'], preds))
            mae = mean_absolute_error(train_df['price'], preds)
            r2 = r2_score(train_df['price'], preds)
            marker = " <-- FINAL (submit this)" if name == 'final' else ""
            logger.info(f"  {name.upper():10s} RMSE: ${rmse:,.0f}, MAE: ${mae:,.0f}, R2: {r2:.4f}{marker}")
    
    feature_importance = analyze_feature_importance(all_models, feature_names_final)
    feature_importance.to_csv(ANALYSIS_DIR / 'feature_importance.csv', index=False)
    
    seg_stats, lux_stats, wf_stats = analyze_residuals(train_df, oof_predictions)
    
    logger.info("\n📈 RESIDUALS BY SEGMENT:")
    logger.info(seg_stats.to_string())
    
    logger.info("\n💎 RESIDUALS BY LUXURY:")
    logger.info(lux_stats.to_string())
    
    logger.info("\n🌊 RESIDUALS BY WATERFRONT:")
    logger.info(wf_stats.to_string())
    
    create_diagnostic_plots(train_df, oof_predictions, feature_importance, fold_rmses)
    
    pd.DataFrame(fold_stats).to_csv(ANALYSIS_DIR / 'fold_stats.csv', index=False)
    
    return {
        'models': all_models,
        'oof_predictions': oof_predictions,
        'feature_names': feature_names_final,
        'feature_importance': feature_importance,
        'splits': splits,
        'fold_rmses': fold_rmses,
        'fold_stats': fold_stats,
        'fold_assignments': fold_assignments  # For fusion alignment
    }


def save_results(results, train_df, kmeans_models):
    logger.info("\nSaving results...")
    
    for model_type in ['lgb', 'lgb_div', 'xgb', 'cat', 'cat_div']:
        for fold, model in enumerate(results['models'][model_type]):
            if model is None:
                continue
            if 'lgb' in model_type:
                model.save_model(str(MODELS_DIR / f'{model_type}_ultimate_fold_{fold}.txt'))
            elif model_type == 'xgb':
                model.save_model(str(MODELS_DIR / f'xgboost_ultimate_fold_{fold}.json'))
            elif 'cat' in model_type:
                model.save_model(str(MODELS_DIR / f'{model_type}_ultimate_fold_{fold}.cbm'))
    
    for fold, model in enumerate(results['models']['rf']):
        if model is not None:
            with open(MODELS_DIR / f'rf_ultimate_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(model, f)
    
    for fold, model in enumerate(results['models']['lux']):
        if model is not None:
            model.save_model(str(MODELS_DIR / f'lux_ultimate_fold_{fold}.txt'))
    
    for fold, model in enumerate(results['models']['meta']):
        if model is not None:
            with open(MODELS_DIR / f'meta_ultimate_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(model, f)
    
    with open(MODELS_DIR / 'kmeans_ultimate.pkl', 'wb') as f:
        pickle.dump(kmeans_models, f)
    
    # Save OOF predictions WITH FOLD ASSIGNMENTS (critical for fusion alignment)
    oof_df = pd.DataFrame({
        'id': train_df['id'], 
        'price': train_df['price'],
        'fold': results['fold_assignments'],  # Add fold column for fusion
        **{f'{k}_prediction': v for k, v in results['oof_predictions'].items()}
    })
    oof_df.to_csv(RESULTS_DIR / 'oof_predictions_ultimate.csv', index=False)
    oof_df.to_csv(RESULTS_DIR / 'baseline_oof_predictions.csv', index=False)  # For fusion
    
    # Save fold splits as pickle for fusion (backup method)
    with open(RESULTS_DIR / 'fold_splits.pkl', 'wb') as f:
        pickle.dump(results['splits'], f)
    logger.info(f"Saved fold splits to {RESULTS_DIR / 'fold_splits.pkl'}")
    
    with open(MODELS_DIR / 'feature_names_ultimate.pkl', 'wb') as f:
        pickle.dump(results['feature_names'], f)
    
    logger.info("Results saved")


def main():
    start = datetime.now()
    
    print("="*70)
    print("ULTIMATE COMPETITION BASELINE")
    print("="*70)
    print(f"Start: {start}")
    print("\n🎯 Philosophy: KEEP V5 POWER + FIX INSTABILITY = WIN LEADERBOARD")
    print("\n📋 STABILITY FIXES:")
    print("  1. Stratified GroupKFold - balance price across folds")
    print("  2. Robust neighborhood features - min neighbors + fallbacks")
    print("  3. Softened luxury blend - 65/35 (safer than 70/30)")
    print("  4. Tighter ElasticNet - alpha=0.01 (more regularization)")
    print("  5. NO post-processing - no clipping, no sanity, no hacks")
    print("\n🎯 Target: ~$119K RMSE with CV < 25% = RELIABLE LEADERBOARD")
    
    set_all_seeds(RANDOM_SEED)
    
    train_df, test_df, kmeans_models = prepare_data()
    results = train_stacked_ensemble(train_df, n_folds=N_FOLDS)
    save_results(results, train_df, kmeans_models)
    
    duration = datetime.now() - start
    
    final_rmse = np.sqrt(mean_squared_error(train_df['price'], results['oof_predictions']['final']))
    fold_cv = np.std(results['fold_rmses']) / np.mean(results['fold_rmses']) * 100
    
    print(f"\n{'='*70}")
    print("ULTIMATE BASELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Duration: {duration}")
    print(f"\n📊 RESULTS:")
    print(f"  Final RMSE: ${final_rmse:,.0f}")
    print(f"  Fold CV: {fold_cv:.1f}%")
    print(f"  Features: {len(results['feature_names'])}")
    
    if fold_cv < 25:
        print(f"\n✅ STABLE BASELINE - Safe for leaderboard submission!")
    else:
        print(f"\n⚠️  High variance - consider additional stabilization")
    
    print(f"\n📁 Output: {ANALYSIS_DIR}")
    print(f"📁 Models: {MODELS_DIR}")
    print(f"\n🚀 Ready for fusion: results/baseline_oof_predictions.csv")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        raise