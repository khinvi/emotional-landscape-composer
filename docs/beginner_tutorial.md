# Beginner's Guide to the Emotional Landscape Composer

Welcome to the wonderful world of landscape music! This guide will get you started with turning geographic data into beautiful musical compositions in just a few minutes.

## Getting Set Up

First, make sure you have the Emotional Landscape Composer installed:

```bash
# Clone the repository
git clone https://github.com/yourusername/emotional-landscape-composer.git
cd emotional-landscape-composer

# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start: The Command Line Tool

The easiest way to start is with our command line tool:

```bash
# Create music from a mountain (synthetic landscape)
python compose_music.py --landscape mountain --output mountain_music.mid

# Try different landscape types
python compose_music.py --landscape valley --output valley_music.mid
python compose_music.py --landscape hills --output hills_music.mid
python compose_music.py --landscape canyon --output canyon_music.mid

# Play the music as it's generated
python compose_music.py --landscape archipelago --play

# Visualize the features and composition
python compose_music.py --landscape random --visualize

# Change the musical key
python compose_music.py --landscape mountain --key C_minor

# Make a longer composition (5 minutes)
python compose_music.py --landscape canyon --duration 5.0
```

## Using Your Own Landscapes

If you have a GeoTIFF file with elevation data, you can use it:

```bash
python compose_music.py --input path/to/your/elevation.tif --output my_landscape.mid
```

## Where to Find Landscape Data

Here are some great sources for free elevation data:

1. **USGS Earth Explorer**: https://earthexplorer.usgs.gov/
2. **NASA SRTM Data**: https://search.earthdata.nasa.gov/
3. **OpenTopography**: https://opentopography.org/

For most of these sites, you'll want to look for "Digital Elevation Model" (DEM) data in GeoTIFF format.

## Creating Your Own Landscapes in Python

Want to make your own custom landscape? Here's a simple example:

```python
import numpy as np
import matplotlib.pyplot as plt
from emotional_landscape_composer import LandscapeComposer

# Create a grid for our landscape
size = 100
x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))

# Make a cool landscape with multiple mountains
elevation = np.zeros((size, size))

# Add three mountain peaks
for peak in [(0, 0), (1, 1), (-1, -1)]:
    px, py = peak
    mountain = np.exp(-0.5 * ((x-px)**2 + (y-py)**2))
    elevation += mountain

# Normalize to [0, 1]
elevation = (elevation - np.min(elevation)) / (np.max(elevation) - np.min(elevation))

# Visualize it
plt.figure(figsize=(10, 8))
plt.imshow(elevation, cmap='terrain')
plt.colorbar(label='Elevation')
plt.title('My Custom Mountain Range')
plt.savefig('my_mountains.png')
plt.close()

# Create the composer and generate music
composer = LandscapeComposer({
    'elevation': elevation,
    'source_type': 'array'
})

composer.analyze_terrain()
composer.generate_composition(duration_minutes=2.0)
composer.save('my_mountain_range.mid')
```

## Experimenting with Different Musical Settings

You can customize the music generation in many ways:

```python
from emotional_landscape_composer import LandscapeComposer

# Load a sample or create your own landscape
# ...

# Try different musical keys
composer = LandscapeComposer(
    data_source=terrain_data,
    key='C_minor'  # Sounds more mysterious
)

# Other keys to try:
# - C_major (bright, happy)
# - C_minor (sad, reflective)
# - C_lydian (magical, dreamy)
# - C_dorian (jazzy, sophisticated)
# - C_pentatonic (Eastern, ethereal)
# - C_blues (soulful, emotional)

# Try different tempo ranges
composer = LandscapeComposer(
    data_source=terrain_data,
    tempo_range=(40, 60)  # Slow and meditative
)

# Other tempo ranges to try:
# - (40, 60) Slow, meditative
# - (70, 90) Moderate, walking pace
# - (100, 130) Energetic, upbeat
# - (160, 180) Very fast, exciting
```

## Understanding the Music

Here's what different landscape features tend to create in the music:

- **Mountains** → Higher pitches, stronger dynamics, often majestic sounds
- **Valleys** → Lower pitches, softer dynamics, more contemplative music
- **Rugged terrain** → More complex rhythms, dissonant harmonies
- **Smooth terrain** → Flowing melodies, consonant harmonies
- **Slopes** → Rising or falling melodic lines
- **Complex features** → More instrument variety and textural complexity

## Fun Project Ideas

Here are some fun things to try with the Emotional Landscape Composer:

1. **Famous Mountain Range Challenge**: Create music from famous mountain ranges around the world and compare them
2. **Musical Map**: Compose music for different parts of your town/city and create a "musical map"
3. **Time-Lapse Music**: Create landscapes that represent the same location at different times of day or seasons
4. **Fantasy Worlds**: Design imaginary landscapes (volcanoes, alien terrain, underwater mountains)
5. **Landscape Morphing**: Create a series of evolving landscapes that gradually change from mountains to valleys
6. **Musical Geology**: Try to recreate landscapes from different geological eras

## Troubleshooting

**Problem**: Error about missing dependencies
**Solution**: Make sure you've installed all requirements with `pip install -r requirements.txt`

**Problem**: Can't play music directly
**Solution**: Make sure pygame is installed, or open the MIDI file with any music player

**Problem**: WAV export doesn't work
**Solution**: Install FluidSynth on your system (this is optional but nice to have)

**Problem**: Visualization doesn't show
**Solution**: Make sure matplotlib is installed and check that you're running in an environment that can display plots

## Going Further

Once you're comfortable with the basics, check out these more advanced topics:

- [How It Works](how_it_works.md) - A deeper dive into the project's internals
- Explore the code in `emotional_landscape_composer/` to see how everything fits together
- Try modifying the code to create your own custom features!

Happy composing! 🏔️🎵