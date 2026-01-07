#!/usr/bin/env python3
"""
Quick Encoder Comparison Script

Runs fast ablations on a subset of data to compare different encoders
before committing to full extraction.

Usage:
    python compare_encoders.py --project-root . --sample-size 5000

Recommended workflow:
1. Run this script with ~5000 samples (30-60 min)
2. Review results to pick best 1-2 encoders
3. Run full extraction with chosen encoder(s)
4. Train fusion with best encoder
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Encoders to compare (ordered by expected performance / compute tradeoff)
ENCODERS_TO_TEST = [
    'efficientnet_b3',   # Current baseline - fast
    'efficientnet_b4',   # Larger - better features
    'resnet50',          # Classic - fast
    'swin_tiny',         # Transformer - good for spatial
    'convnext_tiny',     # Modern CNN
]


def run_encoder_comparison(project_root, sample_size=5000, batch_size=32):
    """Run quick comparison of encoders on a sample of data."""
    
    project_root = Path(project_root)
    sys.path.insert(0, str(project_root))
    
    from config import TRAIN_FILE, RESULTS_DIR
    
    # Load training data
    train_df = pd.read_csv(TRAIN_FILE)
    logger.info(f"Total training samples: {len(train_df)}")
    
    # Sample for quick testing
    if sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=42)
        logger.info(f"Using sample of {sample_size} for comparison")
    
    # Import extraction module
    from extract_multi_encoder_features import MultiEncoderFeatureExtractor, ImageFeatureDataset, ENCODER_CONFIGS
    from torch.utils.data import DataLoader
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Prepare image directory
    try:
        from config import IMAGES_ZOOM_17_DIR
        image_dir = Path(IMAGES_ZOOM_17_DIR)
    except ImportError:
        image_dir = project_root / "data" / "images" / "zoom_17"
    
    results = []
    
    for encoder_name in ENCODERS_TO_TEST:
        if encoder_name not in ENCODER_CONFIGS:
            logger.warning(f"Encoder {encoder_name} not available, skipping")
            continue
            
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {encoder_name}")
        logger.info(f"{'='*50}")
        
        start_time = datetime.now()
        
        try:
            # Initialize extractor
            extractor = MultiEncoderFeatureExtractor(encoder_name, device, use_fp16=True)
            
            # Create dataset and dataloader
            dataset = ImageFeatureDataset(image_dir, train_df['id'].values, 17, extractor.transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            # Extract features
            features = extractor.extract(dataloader)
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            # Compute basic statistics
            feature_matrix = np.array([features[pid] for pid in train_df['id'].values if pid in features])
            
            # Feature statistics
            mean_norm = np.mean(np.linalg.norm(feature_matrix, axis=1))
            std_norm = np.std(np.linalg.norm(feature_matrix, axis=1))
            coverage = len(features) / len(train_df)
            
            # Variance explained by top PCA components
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(100, feature_matrix.shape[1]))
            pca.fit(feature_matrix)
            var_50 = np.sum(pca.explained_variance_ratio_[:50])
            var_100 = np.sum(pca.explained_variance_ratio_[:100])
            
            result = {
                'encoder': encoder_name,
                'feature_dim': extractor.feature_dim,
                'input_size': extractor.input_size,
                'extraction_time_sec': extraction_time,
                'coverage': coverage,
                'mean_norm': mean_norm,
                'std_norm': std_norm,
                'pca_var_50': var_50,
                'pca_var_100': var_100,
            }
            results.append(result)
            
            logger.info(f"  Time: {extraction_time:.1f}s")
            logger.info(f"  Coverage: {coverage:.1%}")
            logger.info(f"  Feature dim: {extractor.feature_dim}")
            logger.info(f"  Mean norm: {mean_norm:.2f} ± {std_norm:.2f}")
            logger.info(f"  PCA var@50: {var_50:.1%}, @100: {var_100:.1%}")
            
            # Clear GPU memory
            del extractor
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error testing {encoder_name}: {e}")
            results.append({
                'encoder': encoder_name,
                'error': str(e)
            })
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ENCODER COMPARISON SUMMARY")
    logger.info(f"{'='*60}")
    
    results_df = pd.DataFrame(results)
    if 'extraction_time_sec' in results_df.columns:
        results_df = results_df.sort_values('pca_var_100', ascending=False)
    
    print(results_df.to_string())
    
    # Save results
    output_file = project_root / "encoder_comparison_results.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to {output_file}")
    
    # Recommendations
    logger.info(f"\n{'='*60}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*60}")
    
    if len(results_df) > 0 and 'pca_var_100' in results_df.columns:
        best = results_df.iloc[0]
        logger.info(f"Best encoder by PCA variance: {best['encoder']}")
        logger.info(f"  - PCA var@100: {best['pca_var_100']:.1%}")
        logger.info(f"  - Extraction time: {best['extraction_time_sec']:.1f}s")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Run full extraction: python extract_multi_encoder_features.py --encoder {best['encoder']}")
        logger.info(f"  2. Train fusion: python train_fusion_gbm.py --encoder {best['encoder']}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Compare encoders for feature extraction")
    parser.add_argument("--project-root", type=str, default=".", help="Project root")
    parser.add_argument("--sample-size", type=int, default=5000, help="Sample size for testing")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    run_encoder_comparison(args.project_root, args.sample_size, args.batch_size)


if __name__ == "__main__":
    main()
