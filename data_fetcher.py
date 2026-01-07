"""
Satellite Image Fetcher using Google Maps Static API.

This module handles downloading satellite imagery for properties using
the Google Maps Static API. It supports:
- Multi-scale imagery (zoom 16, 17, and 18)
- Disk caching to avoid redundant downloads
- Rate limiting and retry logic with exponential backoff
- Image validation
- Batch processing with progress tracking
- Green dominance feature extraction

Usage:
    python src/data_fetcher.py --fetch-all
    python src/data_fetcher.py --fetch-train
    python src/data_fetcher.py --fetch-test
    python src/data_fetcher.py --sample 100

Author: Competition Submission
Date: December 2024
"""

import sys
import os
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from io import BytesIO

import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_FILE, TEST_FILE,
    IMAGES_ZOOM_16_DIR, IMAGES_ZOOM_17_DIR, IMAGES_ZOOM_18_DIR,
    GOOGLE_MAPS_API_KEY, GOOGLE_MAPS_BASE_URL,
    IMAGE_SIZE, ZOOM_LEVELS, IMAGE_FORMAT, MAP_TYPE,
    GOOGLE_REQUESTS_PER_MINUTE,
    RETRY_ATTEMPTS, RETRY_DELAY_SECONDS, RETRY_BACKOFF_MULTIPLIER
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoogleMapsImageFetcher:
    """
    Fetches satellite imagery from Google Maps Static API.
    
    Features:
    - Automatic caching to disk
    - Rate limiting
    - Retry with exponential backoff
    - Image validation
    - Multi-scale support (zoom 16, 17, and 18)
    """
    
    def __init__(
        self,
        api_key: str = GOOGLE_MAPS_API_KEY,
        requests_per_minute: int = GOOGLE_REQUESTS_PER_MINUTE,
        cache_dir_z16: Path = IMAGES_ZOOM_16_DIR,
        cache_dir_z17: Path = IMAGES_ZOOM_17_DIR,
        cache_dir_z18: Path = IMAGES_ZOOM_18_DIR
    ):
        """
        Initialize the Google Maps image fetcher.
        
        Args:
            api_key: Google Maps API key
            requests_per_minute: Rate limit for API calls
            cache_dir_z16: Directory for zoom 16 images
            cache_dir_z17: Directory for zoom 17 images
            cache_dir_z18: Directory for zoom 18 images
        """
        self.api_key = api_key
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        
        self.cache_dirs = {
            16: Path(cache_dir_z16),
            17: Path(cache_dir_z17),
            18: Path(cache_dir_z18)
        }
        
        # Ensure cache directories exist
        for cache_dir in self.cache_dirs.values():
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Statistics
        self.stats = {
            'fetched': 0,
            'cached': 0,
            'failed': 0,
            'total_time': 0
        }
        
        if not self.api_key:
            logger.warning("⚠️  GOOGLE_MAPS_API_KEY not set!")
            logger.warning("   Set it via: export GOOGLE_MAPS_API_KEY='your_key'")
    
    def _get_cache_path(self, property_id: str, zoom: int) -> Path:
        """Get the cache file path for a property image."""
        return self.cache_dirs[zoom] / f"{property_id}_z{zoom}.jpg"
    
    def _is_cached(self, property_id: str, zoom: int) -> bool:
        """Check if image is already cached and valid."""
        cache_path = self._get_cache_path(property_id, zoom)
        if not cache_path.exists():
            return False
        
        # Validate cached image
        try:
            with Image.open(cache_path) as img:
                img.verify()
            return True
        except Exception:
            # Invalid cache, remove it
            cache_path.unlink(missing_ok=True)
            return False
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
    
    def _build_url(self, lat: float, lon: float, zoom: int) -> str:
        """
        Build Google Maps Static API URL.
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level (16, 17, or 18)
            
        Returns:
            Complete API URL
        """
        params = {
            'center': f"{lat},{lon}",
            'zoom': zoom,
            'size': f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            'maptype': MAP_TYPE,
            'format': IMAGE_FORMAT,
            'key': self.api_key
        }
        
        query_string = '&'.join(f"{k}={v}" for k, v in params.items())
        return f"{GOOGLE_MAPS_BASE_URL}?{query_string}"
    
    def _fetch_single_image(
        self,
        lat: float,
        lon: float,
        zoom: int
    ) -> Optional[bytes]:
        """
        Fetch a single image from Google Maps API with retry logic.
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level
            
        Returns:
            Image bytes or None if failed
        """
        url = self._build_url(lat, lon, zoom)
        
        for attempt in range(RETRY_ATTEMPTS):
            try:
                self._rate_limit()
                
                response = self.session.get(url, timeout=30)
                
                # Check for API errors
                if response.status_code == 200:
                    # Validate it's actually an image
                    content_type = response.headers.get('Content-Type', '')
                    if 'image' in content_type:
                        # Additional validation
                        try:
                            img = Image.open(BytesIO(response.content))
                            img.verify()
                            return response.content
                        except Exception:
                            logger.warning(f"Invalid image data received")
                    else:
                        # Might be an error response (JSON)
                        logger.warning(f"Non-image response: {content_type}")
                        if 'application/json' in content_type:
                            logger.warning(f"API Error: {response.text[:200]}")
                
                elif response.status_code == 403:
                    logger.error("API key invalid or quota exceeded")
                    return None
                
                elif response.status_code == 429:
                    # Rate limited - back off
                    wait_time = RETRY_DELAY_SECONDS * (RETRY_BACKOFF_MULTIPLIER ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.warning(f"HTTP {response.status_code} for ({lat}, {lon})")
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error: {e}")
            
            # Exponential backoff
            if attempt < RETRY_ATTEMPTS - 1:
                wait_time = RETRY_DELAY_SECONDS * (RETRY_BACKOFF_MULTIPLIER ** attempt)
                time.sleep(wait_time)
        
        return None
    
    def fetch_property_images(
        self,
        property_id: str,
        lat: float,
        lon: float,
        zoom_levels: List[int] = ZOOM_LEVELS,
        force: bool = False
    ) -> Dict[int, bool]:
        """
        Fetch satellite images for a property at multiple zoom levels.
        
        Args:
            property_id: Unique property identifier
            lat: Latitude
            lon: Longitude
            zoom_levels: List of zoom levels to fetch
            force: If True, refetch even if cached
            
        Returns:
            Dict mapping zoom level to success status
        """
        results = {}
        
        for zoom in zoom_levels:
            cache_path = self._get_cache_path(property_id, zoom)
            
            # Check cache first
            if not force and self._is_cached(property_id, zoom):
                self.stats['cached'] += 1
                results[zoom] = True
                continue
            
            # Fetch from API
            image_data = self._fetch_single_image(lat, lon, zoom)
            
            if image_data:
                # Save to cache
                try:
                    with open(cache_path, 'wb') as f:
                        f.write(image_data)
                    self.stats['fetched'] += 1
                    results[zoom] = True
                except IOError as e:
                    logger.error(f"Failed to save image: {e}")
                    self.stats['failed'] += 1
                    results[zoom] = False
            else:
                self.stats['failed'] += 1
                results[zoom] = False
        
        return results
    
    def fetch_batch(
        self,
        df: pd.DataFrame,
        id_col: str = 'id',
        lat_col: str = 'lat',
        lon_col: str = 'long',
        zoom_levels: List[int] = ZOOM_LEVELS,
        desc: str = "Fetching images"
    ) -> pd.DataFrame:
        """
        Fetch images for a batch of properties.
        
        Args:
            df: DataFrame with property data
            id_col: Column name for property ID
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            zoom_levels: List of zoom levels
            desc: Progress bar description
            
        Returns:
            DataFrame with image availability columns added
        """
        start_time = time.time()
        
        # Reset stats
        self.stats = {'fetched': 0, 'cached': 0, 'failed': 0, 'total_time': 0}
        
        # Count how many need fetching
        needs_fetch = 0
        for _, row in df.iterrows():
            for zoom in zoom_levels:
                if not self._is_cached(str(row[id_col]), zoom):
                    needs_fetch += 1
        
        logger.info(f"Total properties: {len(df)}")
        logger.info(f"Images to fetch: {needs_fetch} (already cached: {len(df) * len(zoom_levels) - needs_fetch})")
        
        # Track availability
        availability = {f'has_z{z}': [] for z in zoom_levels}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            results = self.fetch_property_images(
                property_id=str(row[id_col]),
                lat=row[lat_col],
                lon=row[lon_col],
                zoom_levels=zoom_levels
            )
            
            for zoom in zoom_levels:
                availability[f'has_z{zoom}'].append(results.get(zoom, False))
        
        # Add availability columns
        for col, values in availability.items():
            df[col] = values
        
        self.stats['total_time'] = time.time() - start_time
        
        # Print summary
        logger.info(f"\n{'='*40}")
        logger.info("Fetch Summary:")
        logger.info(f"  Fetched: {self.stats['fetched']}")
        logger.info(f"  From cache: {self.stats['cached']}")
        logger.info(f"  Failed: {self.stats['failed']}")
        logger.info(f"  Total time: {self.stats['total_time']:.1f}s")
        
        return df
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached images."""
        stats = {}
        for zoom, cache_dir in self.cache_dirs.items():
            count = len(list(cache_dir.glob('*.jpg')))
            stats[f'zoom_{zoom}'] = count
        return stats


def compute_green_dominance(image_path: Path) -> Optional[float]:
    """
    Compute green dominance metric from satellite image.
    
    This is an RGB-based vegetation proxy (NOT true NDVI).
    Higher values indicate more vegetation/greenery.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Green dominance ratio (0-1) or None if failed
    """
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img.convert('RGB'))
            
            r = img_array[:, :, 0].astype(float)
            g = img_array[:, :, 1].astype(float)
            b = img_array[:, :, 2].astype(float)
            
            # Green dominance: how much green exceeds red and blue
            total = r + g + b + 1e-6  # Avoid division by zero
            green_ratio = g / total
            
            # Vegetation tends to have g > r and g > b
            veg_mask = (g > r) & (g > b)
            green_dominance = np.mean(veg_mask) * np.mean(green_ratio)
            
            return float(green_dominance)
    
    except Exception as e:
        logger.debug(f"Error computing green dominance: {e}")
        return None


def compute_image_stats(image_path: Path) -> Optional[Dict[str, float]]:
    """
    Compute various statistics from satellite image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary of image statistics or None if failed
    """
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img.convert('RGB')).astype(float)
            
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            stats = {
                'green_dominance': compute_green_dominance(image_path),
                'brightness': np.mean(img_array) / 255.0,
                'contrast': np.std(img_array) / 255.0,
                'green_ratio': np.mean(g) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6),
                'blue_ratio': np.mean(b) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6),
            }
            
            return stats
    
    except Exception:
        return None


def compute_all_image_features(
    df: pd.DataFrame,
    id_col: str = 'id',
    cache_dir_z16: Path = IMAGES_ZOOM_16_DIR,
    cache_dir_z17: Path = IMAGES_ZOOM_17_DIR,
    cache_dir_z18: Path = IMAGES_ZOOM_18_DIR
) -> pd.DataFrame:
    """
    Compute image-derived features for all properties at all zoom levels.
    
    Args:
        df: DataFrame with property IDs
        id_col: Column name for property ID
        cache_dir_z16: Directory for zoom 16 images
        cache_dir_z17: Directory for zoom 17 images
        cache_dir_z18: Directory for zoom 18 images
        
    Returns:
        DataFrame with image features added
    """
    df = df.copy()
    
    # Initialize columns for all zoom levels
    for zoom in [16, 17, 18]:
        df[f'green_dominance_z{zoom}'] = np.nan
        df[f'brightness_z{zoom}'] = np.nan
        df[f'contrast_z{zoom}'] = np.nan
    
    cache_dirs = {16: cache_dir_z16, 17: cache_dir_z17, 18: cache_dir_z18}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing image features"):
        prop_id = str(row[id_col])
        
        # Process each zoom level
        for zoom in [16, 17, 18]:
            img_path = cache_dirs[zoom] / f"{prop_id}_z{zoom}.jpg"
            if img_path.exists():
                stats = compute_image_stats(img_path)
                if stats:
                    df.loc[idx, f'green_dominance_z{zoom}'] = stats['green_dominance']
                    df.loc[idx, f'brightness_z{zoom}'] = stats['brightness']
                    df.loc[idx, f'contrast_z{zoom}'] = stats['contrast']
    
    # Fill missing with median
    for zoom in [16, 17, 18]:
        for feature in ['green_dominance', 'brightness', 'contrast']:
            col = f'{feature}_z{zoom}'
            median_val = df[col].median()
            df[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)
    
    return df


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description='Fetch satellite images from Google Maps Static API'
    )
    parser.add_argument(
        '--fetch-train', action='store_true',
        help='Fetch images for training set only'
    )
    parser.add_argument(
        '--fetch-test', action='store_true',
        help='Fetch images for test set only'
    )
    parser.add_argument(
        '--fetch-all', action='store_true',
        help='Fetch images for both train and test sets'
    )
    parser.add_argument(
        '--sample', type=int, default=None,
        help='Fetch only N random samples (for testing)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force re-download even if cached'
    )
    parser.add_argument(
        '--stats', action='store_true',
        help='Show cache statistics only'
    )
    parser.add_argument(
        '--zoom', type=int, nargs='+', default=None,
        help='Specific zoom levels to fetch (e.g., --zoom 16 17 18)'
    )
    
    args = parser.parse_args()
    
    # Initialize fetcher
    fetcher = GoogleMapsImageFetcher()
    
    # Show stats only
    if args.stats:
        stats = fetcher.get_cache_stats()
        print("\nCache Statistics:")
        print(f"  Zoom 16 images: {stats['zoom_16']}")
        print(f"  Zoom 17 images: {stats['zoom_17']}")
        print(f"  Zoom 18 images: {stats['zoom_18']}")
        print(f"  Total: {stats['zoom_16'] + stats['zoom_17'] + stats['zoom_18']}")
        return
    
    # Check API key
    if not GOOGLE_MAPS_API_KEY:
        print("\n❌ ERROR: GOOGLE_MAPS_API_KEY not set!")
        print("   Please set it via:")
        print("   export GOOGLE_MAPS_API_KEY='your_api_key_here'")
        print("\n   Get a key at: https://console.cloud.google.com/apis/credentials")
        return
    
    # Determine what to fetch
    fetch_train = args.fetch_train or args.fetch_all
    fetch_test = args.fetch_test or args.fetch_all
    
    if not (fetch_train or fetch_test):
        print("No action specified. Use --fetch-train, --fetch-test, or --fetch-all")
        parser.print_help()
        return
    
    # Determine zoom levels
    zoom_levels = args.zoom if args.zoom else ZOOM_LEVELS
    
    print("="*60)
    print("Google Maps Satellite Image Fetcher")
    print("="*60)
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Zoom levels: {zoom_levels}")
    print(f"  Z16: Regional context (~2.4km coverage)")
    print(f"  Z17: Neighborhood context (~1.2km coverage)")
    print(f"  Z18: Property detail (~600m coverage)")
    print(f"Rate limit: {GOOGLE_REQUESTS_PER_MINUTE} req/min")
    
    # Fetch training images
    if fetch_train and TRAIN_FILE.exists():
        print(f"\n--- Fetching Training Set Images ---")
        train_df = pd.read_csv(TRAIN_FILE)
        
        if args.sample:
            train_df = train_df.sample(n=min(args.sample, len(train_df)), random_state=42)
            print(f"Sampling {len(train_df)} properties")
        
        fetcher.fetch_batch(train_df, zoom_levels=zoom_levels, desc="Training images")
    
    # Fetch test images
    if fetch_test and TEST_FILE.exists():
        print(f"\n--- Fetching Test Set Images ---")
        test_df = pd.read_csv(TEST_FILE)
        
        if args.sample:
            test_df = test_df.sample(n=min(args.sample, len(test_df)), random_state=42)
            print(f"Sampling {len(test_df)} properties")
        
        fetcher.fetch_batch(test_df, zoom_levels=zoom_levels, desc="Test images")
    
    # Final stats
    print("\n" + "="*60)
    print("Final Cache Statistics:")
    stats = fetcher.get_cache_stats()
    print(f"  Zoom 16: {stats['zoom_16']} images")
    print(f"  Zoom 17: {stats['zoom_17']} images")
    print(f"  Zoom 18: {stats['zoom_18']} images")
    print(f"  Total: {sum(stats.values())} images")
    print("="*60)


if __name__ == "__main__":
    main()