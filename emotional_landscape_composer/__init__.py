"""
Emotional Landscape Composer
----------------------------
A Python library for transforming geographical data into musical compositions.
"""

__version__ = '0.1.0'

# Import main components for easy access
from .composer import LandscapeComposer
from .data_processor import load_sample, load_geotiff, load_shapefile
from .feature_extractor import TerrainFeatureExtractor
from .emotion_interpreter import EmotionModel
from .music_generator import MusicGenerator

# Make sample datasets easily accessible
SAMPLE_DATASETS = {
    'mountain_valley': 'samples/mountain_valley.tif',
    'coastal_region': 'samples/coastal_region.tif',
    'urban_landscape': 'samples/urban_landscape.tif',
    'desert_plateau': 'samples/desert_plateau.tif',
    'river_delta': 'samples/river_delta.tif'
}