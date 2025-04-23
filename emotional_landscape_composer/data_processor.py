"""
Data processing module for the Emotional Landscape Composer.
"""

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import matplotlib.pyplot as plt
from typing import Dict, Union, List, Tuple, Any, Optional

class DataProcessor:
    """
    Load and process geographic data for the Emotional Landscape Composer.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the data processor.
        
        Args:
            cache_dir: Directory for caching processed data
        """
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def load(self, source: Union[str, np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load geographic data from various sources.
        
        Args:
            source: Path to a GeoTIFF file, NumPy array, or dictionary with data
            
        Returns:
            Dictionary containing processed data
        """
        if isinstance(source, str):
            if source.endswith(('.tif', '.tiff')):
                return self.load_geotiff(source)
            elif source.endswith(('.shp')):
                return self.load_shapefile(source)
            else:
                raise ValueError(f"Unsupported file format: {source}")
        elif isinstance(source, np.ndarray):
            # Assume it's elevation data
            return {
                'elevation': source,
                'source_type': 'array',
                'crs': None,
                'bounds': None
            }
        elif isinstance(source, dict):
            # Already processed data
            return source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
    
    def load_geotiff(self, filepath: str) -> Dict[str, Any]:
        """
        Load data from a GeoTIFF file.
        
        Args:
            filepath: Path to the GeoTIFF file
            
        Returns:
            Dictionary containing the loaded data and metadata
        """
        with rasterio.open(filepath) as src:
            data = src.read(1)  # Read the first band
            
            return {
                'elevation': data,
                'source_type': 'geotiff',
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'resolution': (src.width, src.height),
                'filepath': filepath
            }
    
    def load_shapefile(self, filepath: str) -> Dict[str, Any]:
        """
        Load data from a shapefile.
        
        Args:
            filepath: Path to the shapefile
            
        Returns:
            Dictionary containing the loaded data and metadata
        """
        gdf = gpd.read_file(filepath)
        
        return {
            'geodataframe': gdf,
            'source_type': 'vector',
            'crs': gdf.crs,
            'bounds': gdf.total_bounds,
            'filepath': filepath
        }
    
    def clip_raster_with_shapefile(
        self, 
        raster_file: str, 
        shapefile: str
    ) -> Dict[str, Any]:
        """
        Clip a raster file using a shapefile.
        
        Args:
            raster_file: Path to the raster file
            shapefile: Path to the shapefile with the clipping geometry
            
        Returns:
            Dictionary containing the clipped data and metadata
        """
        # Load the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Ensure the CRS matches the raster
        with rasterio.open(raster_file) as src:
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            
            # Get the geometry
            geometry = gdf.geometry.values
            
            # Perform the clip
            out_image, out_transform = mask(src, geometry, crop=True)
            
            # Update the metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            return {
                'elevation': out_image[0],  # First band
                'source_type': 'clipped_geotiff',
                'crs': src.crs,
                'transform': out_transform,
                'bounds': rasterio.transform.array_bounds(
                    out_image.shape[1], out_image.shape[2], out_transform
                ),
                'resolution': (out_image.shape[2], out_image.shape[1]),
                'original_filepath': raster_file,
                'clip_filepath': shapefile
            }
    
    def resample(
        self, 
        data: Dict[str, Any], 
        target_resolution: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Resample raster data to a different resolution.
        
        Args:
            data: Dictionary containing raster data
            target_resolution: Target resolution as (width, height)
            
        Returns:
            Dictionary containing the resampled data
        """
        if data['source_type'] not in ['geotiff', 'clipped_geotiff', 'array']:
            raise ValueError(f"Resampling not supported for source type: {data['source_type']}")
        
        elevation = data['elevation']
        current_height, current_width = elevation.shape
        target_width, target_height = target_resolution
        
        # Create resampled array using simple interpolation
        # (For a more sophisticated approach, you could use rasterio.warp.reproject)
        import scipy.ndimage
        resampled = scipy.ndimage.zoom(
            elevation, 
            (target_height / current_height, target_width / current_width),
            order=1
        )
        
        # Create new data dictionary with updated values
        resampled_data = data.copy()
        resampled_data['elevation'] = resampled
        resampled_data['resolution'] = target_resolution
        
        # If transform is available, update it
        if 'transform' in data:
            from rasterio.transform import Affine
            old_transform = data['transform']
            scale_x = current_width / target_width
            scale_y = current_height / target_height
            new_transform = Affine(
                old_transform.a * scale_x,
                old_transform.b,
                old_transform.c,
                old_transform.d,
                old_transform.e * scale_y,
                old_transform.f
            )
            resampled_data['transform'] = new_transform
        
        return resampled_data
    
    def calculate_slope(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Calculate slope from elevation data.
        
        Args:
            data: Dictionary containing elevation data
            
        Returns:
            NumPy array containing slope values
        """
        elevation = data['elevation']
        
        # Simple gradient-based slope calculation
        dy, dx = np.gradient(elevation)
        slope = np.sqrt(dx**2 + dy**2)
        
        return slope
    
    def calculate_aspect(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Calculate aspect (direction of slope) from elevation data.
        
        Args:
            data: Dictionary containing elevation data
            
        Returns:
            NumPy array containing aspect values in degrees (0-360)
        """
        elevation = data['elevation']
        
        # Calculate gradients
        dy, dx = np.gradient(elevation)
        
        # Calculate aspect
        aspect = np.degrees(np.arctan2(dy, dx))
        
        # Convert to 0-360 degrees
        aspect = 90 - aspect
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        
        return aspect
    
    def visualize(self, data: Dict[str, Any], title: str = "Geographic Data") -> None:
        """
        Visualize the geographic data.
        
        Args:
            data: Dictionary containing data to visualize
            title: Plot title
        """
        if data['source_type'] in ['geotiff', 'clipped_geotiff', 'array']:
            # Visualize raster data
            plt.figure(figsize=(10, 8))
            plt.imshow(data['elevation'], cmap='terrain')
            plt.colorbar(label='Elevation')
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        elif data['source_type'] == 'vector':
            # Visualize vector data
            gdf = data['geodataframe']
            gdf.plot(figsize=(10, 8))
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError(f"Visualization not supported for source type: {data['source_type']}")


def load_sample(name: str) -> Dict[str, Any]:
    """
    Load a sample dataset by name.
    
    Args:
        name: Name of the sample dataset
        
    Returns:
        Dictionary containing the sample data
    """
    from . import SAMPLE_DATASETS
    
    if name not in SAMPLE_DATASETS:
        raise ValueError(f"Sample dataset not found: {name}")
    
    # Get the path to the sample file
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        SAMPLE_DATASETS[name]
    )
    
    # Load the sample data
    processor = DataProcessor()
    return processor.load(sample_path)


def load_geotiff(filepath: str) -> Dict[str, Any]:
    """
    Load a GeoTIFF file.
    
    Args:
        filepath: Path to the GeoTIFF file
        
    Returns:
        Dictionary containing the loaded data
    """
    processor = DataProcessor()
    return processor.load_geotiff(filepath)


def load_shapefile(filepath: str) -> Dict[str, Any]:
    """
    Load a shapefile.
    
    Args:
        filepath: Path to the shapefile
        
    Returns:
        Dictionary containing the loaded data
    """
    processor = DataProcessor()
    return processor.load_shapefile(filepath)