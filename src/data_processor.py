"""
Sand Dune Encroachment Tracker - Data Processor
Handles satellite imagery processing from Sentinel-1, Sentinel-2, and Landsat-8
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class SatelliteDataProcessor:
    """
    Process satellite imagery for sand dune monitoring and change detection
    """

    def __init__(self, config_path: str = 'config.json'):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.tile_size = self.config['processing']['tile_size']
        self.overlap = self.config['processing']['overlap']
        self.normalize = self.config['processing']['normalize']

    def calculate_ndvi(self, red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index"""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir_band - red_band) / (nir_band + red_band)
            ndvi = np.where(np.isnan(ndvi), 0, ndvi)
            ndvi = np.clip(ndvi, -1, 1)
        return ndvi

    def calculate_ndsi(self, green_band: np.ndarray, swir_band: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Sand Index"""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndsi = (green_band - swir_band) / (green_band + swir_band)
            ndsi = np.where(np.isnan(ndsi), 0, ndsi)
            ndsi = np.clip(ndsi, -1, 1)
        return ndsi

    def calculate_mndsi(self, green_band: np.ndarray, swir1_band: np.ndarray) -> np.ndarray:
        """Calculate Modified Normalized Difference Sand Index"""
        with np.errstate(divide='ignore', invalid='ignore'):
            mndsi = (green_band - swir1_band) / (green_band + swir1_band)
            mndsi = np.where(np.isnan(mndsi), 0, mndsi)
            mndsi = np.clip(mndsi, -1, 1)
        return mndsi

    def calculate_ndesi(self, blue_band: np.ndarray, red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Enhanced Sand Index"""
        with np.errstate(divide='ignore', invalid='ignore'):
            # NDESI = (Blue - Red) / (Blue + Red + NIR)
            ndesi = (blue_band - red_band) / (blue_band + red_band + nir_band)
            ndesi = np.where(np.isnan(ndesi), 0, ndesi)
            ndesi = np.clip(ndesi, -1, 1)
        return ndesi

    def process_sentinel1_sar(self, vv_band: np.ndarray, vh_band: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process Sentinel-1 SAR data
        """
        # Convert to dB scale
        vv_db = 10 * np.log10(np.maximum(vv_band, 1e-10))
        vh_db = 10 * np.log10(np.maximum(vh_band, 1e-10))

        # Calculate radar vegetation index
        with np.errstate(divide='ignore', invalid='ignore'):
            rvi = (4 * vh_band) / (vv_band + vh_band)
            rvi = np.where(np.isnan(rvi), 0, rvi)

        # Calculate dual-pol SAR vegetation index
        dvpi = np.sqrt(vv_band * vh_band)

        # Surface roughness indicator
        roughness = vv_band / vh_band
        roughness = np.where(np.isnan(roughness), 1, roughness)

        return {
            'vv_db': vv_db,
            'vh_db': vh_db,
            'rvi': rvi,
            'dvpi': dvpi,
            'roughness': roughness
        }

    def process_optical_imagery(self, bands: Dict[str, np.ndarray], sensor: str = 'sentinel2') -> Dict[str, np.ndarray]:
        """
        Process optical imagery (Sentinel-2 or Landsat-8)
        """
        results = {}

        if sensor == 'sentinel2':
            # Sentinel-2 band mapping
            blue = bands.get('B2', np.zeros((256, 256)))
            green = bands.get('B3', np.zeros((256, 256)))
            red = bands.get('B4', np.zeros((256, 256)))
            nir = bands.get('B8', np.zeros((256, 256)))
            swir1 = bands.get('B11', np.zeros((256, 256)))
            swir2 = bands.get('B12', np.zeros((256, 256)))

        elif sensor == 'landsat8':
            # Landsat-8 band mapping
            blue = bands.get('B2', np.zeros((256, 256)))
            green = bands.get('B3', np.zeros((256, 256)))
            red = bands.get('B4', np.zeros((256, 256)))
            nir = bands.get('B5', np.zeros((256, 256)))
            swir1 = bands.get('B6', np.zeros((256, 256)))
            swir2 = bands.get('B7', np.zeros((256, 256)))

        # Calculate spectral indices
        results['ndvi'] = self.calculate_ndvi(red, nir)
        results['ndsi'] = self.calculate_ndsi(green, swir1)
        results['mndsi'] = self.calculate_mndsi(green, swir1)
        results['ndesi'] = self.calculate_ndesi(blue, red, nir)

        # Calculate additional indices for sand detection
        with np.errstate(divide='ignore', invalid='ignore'):
            # Brightness index
            results['brightness'] = (blue + green + red + nir) / 4

            # Sand-specific indices
            results['sand_ratio'] = red / (nir + 1e-10)
            results['moisture_index'] = (nir - swir1) / (nir + swir1)

        return results

    def detect_changes(self, image_t1: np.ndarray, image_t2: np.ndarray, method: str = 'simple_diff') -> np.ndarray:
        """
        Detect changes between two time periods
        """
        if method == 'simple_diff':
            return np.abs(image_t2 - image_t1)

        elif method == 'normalized_diff':
            with np.errstate(divide='ignore', invalid='ignore'):
                change = (image_t2 - image_t1) / (image_t2 + image_t1 + 1e-10)
                return np.where(np.isnan(change), 0, change)

        elif method == 'change_vector':
            # Calculate magnitude of change vector
            return np.sqrt((image_t2 - image_t1) ** 2)

        else:
            raise ValueError(f"Unknown change detection method: {method}")

    def create_time_series_data(self, image_stack: np.ndarray, window_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series data for LSTM training
        """
        n_timesteps, height, width = image_stack.shape
        X, y = [], []

        for i in range(window_size, n_timesteps):
            X.append(image_stack[i-window_size:i])
            y.append(image_stack[i])

        return np.array(X), np.array(y)

    def preprocess_for_cnn(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess images for CNN training
        """
        # Stack images and normalize
        image_stack = np.stack(images, axis=0)

        if self.normalize:
            # Normalize to [0, 1] range
            image_stack = (image_stack - image_stack.min()) / (image_stack.max() - image_stack.min() + 1e-10)

        # Add channel dimension if needed
        if len(image_stack.shape) == 3:
            image_stack = np.expand_dims(image_stack, axis=-1)

        return image_stack

    def create_synthetic_data(self, n_samples: int = 1000, image_size: Tuple[int, int] = (256, 256)) -> Dict[str, np.ndarray]:
        """
        Create synthetic data for testing and demonstration
        """
        np.random.seed(42)  # For reproducibility

        # Create synthetic sand dune patterns
        height, width = image_size
        x = np.linspace(0, 4*np.pi, width)
        y = np.linspace(0, 4*np.pi, height)
        X, Y = np.meshgrid(x, y)

        images = []
        movements = []

        for i in range(n_samples):
            # Create base dune pattern with random parameters
            amplitude = np.random.uniform(0.3, 0.8)
            frequency = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)

            # Generate dune pattern
            dune_pattern = amplitude * np.sin(frequency * X + phase) * np.cos(frequency * Y)

            # Add noise and variations
            noise = np.random.normal(0, 0.1, (height, width))
            dune_pattern += noise

            # Simulate sand movement (shift pattern)
            shift_x = np.random.uniform(-5, 5)
            shift_y = np.random.uniform(-5, 5)

            # Create shifted version for movement simulation
            shifted_pattern = np.roll(np.roll(dune_pattern, int(shift_x), axis=1), int(shift_y), axis=0)

            images.append(dune_pattern)
            movements.append([shift_x, shift_y])

        return {
            'images': np.array(images),
            'movements': np.array(movements),
            'metadata': {
                'n_samples': n_samples,
                'image_size': image_size,
                'synthetic': True
            }
        }

    def save_processed_data(self, data: Dict, output_path: str):
        """Save processed data to disk"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, **data)
        print(f"Data saved to {output_path}")

    def load_processed_data(self, input_path: str) -> Dict:
        """Load processed data from disk"""
        data = np.load(input_path)
        return dict(data)

# Example usage and testing
if __name__ == "__main__":
    processor = SatelliteDataProcessor()

    # Create synthetic data for demonstration
    synthetic_data = processor.create_synthetic_data(n_samples=100)

    print(f"Created synthetic dataset:")
    print(f"Images shape: {synthetic_data['images'].shape}")
    print(f"Movements shape: {synthetic_data['movements'].shape}")

    # Test spectral indices calculation
    test_bands = {
        'B2': np.random.uniform(0, 1, (256, 256)),  # Blue
        'B3': np.random.uniform(0, 1, (256, 256)),  # Green
        'B4': np.random.uniform(0, 1, (256, 256)),  # Red
        'B8': np.random.uniform(0, 1, (256, 256)),  # NIR
        'B11': np.random.uniform(0, 1, (256, 256)), # SWIR1
        'B12': np.random.uniform(0, 1, (256, 256))  # SWIR2
    }

    indices = processor.process_optical_imagery(test_bands, 'sentinel2')
    print(f"\nCalculated spectral indices: {list(indices.keys())}")

    # Test change detection
    change_map = processor.detect_changes(indices['ndvi'], indices['ndsi'])
    print(f"Change detection completed. Max change: {change_map.max():.4f}")
