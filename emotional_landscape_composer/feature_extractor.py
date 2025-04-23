"""
Feature extraction module for the Emotional Landscape Composer.
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional


class TerrainFeatureExtractor:
    """
    Extract features from terrain data for emotional interpretation.
    """
    
    def __init__(self, resolution: int = 128):
        """
        Initialize the feature extractor.
        
        Args:
            resolution: Resolution for feature extraction grid
        """
        self.resolution = resolution
        
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract features from terrain data.
        
        Args:
            data: Dictionary containing terrain data
            
        Returns:
            Dictionary of extracted features
        """
        # Make sure we have elevation data
        if 'elevation' not in data:
            raise ValueError("Elevation data is required for feature extraction")
        
        # Extract the elevation data
        elevation = data['elevation']
        
        # Normalize elevation to [0, 1] range
        elevation_norm = self._normalize(elevation)
        
        # Calculate derived features
        features = {
            'elevation': elevation_norm,
            'slope': self._calculate_slope(elevation_norm),
            'ruggedness': self._calculate_ruggedness(elevation_norm),
            'relief': self._calculate_relief(elevation_norm),
            'complexity': self._calculate_complexity(elevation_norm)
        }
        
        return features
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0, 1] range.
        
        Args:
            data: NumPy array to normalize
            
        Returns:
            Normalized array
        """
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val == min_val:
            return np.zeros_like(data)
        
        return (data - min_val) / (max_val - min_val)
    
    def _calculate_slope(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate slope from elevation data.
        
        Args:
            elevation: Normalized elevation data
            
        Returns:
            Slope values normalized to [0, 1]
        """
        # Calculate gradients
        dy, dx = np.gradient(elevation)
        
        # Calculate slope
        slope = np.sqrt(dx**2 + dy**2)
        
        # Normalize to [0, 1]
        return self._normalize(slope)
    
    def _calculate_aspect(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate aspect (direction of slope) from elevation data.
        
        Args:
            elevation: Normalized elevation data
            
        Returns:
            Aspect values normalized to [0, 1]
        """
        # Calculate gradients
        dy, dx = np.gradient(elevation)
        
        # Calculate aspect
        aspect = np.arctan2(dy, dx)
        
        # Convert to [0, 1] range (from [-pi, pi])
        aspect = (aspect + np.pi) / (2 * np.pi)
        
        return aspect
    
    def _calculate_ruggedness(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate terrain ruggedness index (TRI).
        
        The TRI calculates the sum of the absolute differences between 
        the elevation of a cell and its 8 surrounding cells.
        
        Args:
            elevation: Normalized elevation data
            
        Returns:
            Ruggedness values normalized to [0, 1]
        """
        # Calculate differences to neighboring cells in all 8 directions
        ruggedness = np.zeros_like(elevation)
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                
                # Shift the elevation grid
                shifted = np.roll(elevation, shift=(i, j), axis=(0, 1))
                
                # Add the absolute difference
                ruggedness += np.abs(elevation - shifted)
        
        # Normalize to [0, 1]
        return self._normalize(ruggedness)
    
    def _calculate_relief(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate local relief (difference between max and min elevation in neighborhood).
        
        Args:
            elevation: Normalized elevation data
            
        Returns:
            Relief values normalized to [0, 1]
        """
        # Define neighborhood size (adjust as needed)
        neighborhood_size = max(3, min(elevation.shape) // 10)
        
        # Calculate maximum and minimum in neighborhood
        max_filter = ndimage.maximum_filter(elevation, size=neighborhood_size)
        min_filter = ndimage.minimum_filter(elevation, size=neighborhood_size)
        
        # Calculate relief
        relief = max_filter - min_filter
        
        # Normalize to [0, 1]
        return self._normalize(relief)
    
    def _calculate_complexity(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate terrain complexity using entropy filter.
        
        Args:
            elevation: Normalized elevation data
            
        Returns:
            Complexity values normalized to [0, 1]
        """
        # Convert to 8-bit for histogram calculation
        elev_8bit = (elevation * 255).astype(np.uint8)
        
        # Define neighborhood size
        neighborhood_size = max(5, min(elevation.shape) // 20)
        
        # Calculate entropy (measure of disorder/complexity)
        complexity = ndimage.generic_filter(
            elev_8bit, 
            self._local_entropy, 
            size=neighborhood_size
        )
        
        # Normalize to [0, 1]
        return self._normalize(complexity)
    
    def _local_entropy(self, values: np.ndarray) -> float:
        """
        Calculate local entropy of values.
        
        Args:
            values: Array of values in the neighborhood
            
        Returns:
            Entropy value
        """
        # Calculate histogram
        hist, _ = np.histogram(values, bins=256, range=(0, 255))
        
        # Normalize histogram to get probabilities
        hist = hist / np.sum(hist)
        
        # Remove zeros
        hist = hist[hist > 0]
        
        # Calculate entropy
        return -np.sum(hist * np.log2(hist))
    
    def visualize_features(self, features: Dict[str, np.ndarray]) -> None:
        """
        Visualize the extracted features.
        
        Args:
            features: Dictionary of extracted features
        """
        # Create a figure with subplots for each feature
        n_features = len(features)
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4))
        
        if n_features == 1:
            axes = [axes]
            
        for ax, (feature_name, feature_data) in zip(axes, features.items()):
            im = ax.imshow(feature_data, cmap='viridis')
            ax.set_title(feature_name)
            plt.colorbar(im, ax=ax)
            
        plt.tight_layout()
        plt.show()
    
    def calculate_feature_statistics(self, features: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each feature.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary of feature statistics
        """
        stats = {}
        
        for feature_name, feature_data in features.items():
            stats[feature_name] = {
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'mean': float(np.mean(feature_data)),
                'median': float(np.median(feature_data)),
                'std': float(np.std(feature_data)),
                'q1': float(np.percentile(feature_data, 25)),
                'q3': float(np.percentile(feature_data, 75))
            }
            
        return stats