# 🎵 Emotional Landscape Composer 🗺️

Transform your favorite landscapes into music! This hobby project combines geography, emotion, and music in a fun and creative way.

## What is This?

The Emotional Landscape Composer is a fun Python tool that turns geographical terrain data into musical compositions. Feed it a mountain, get a majestic symphony. Give it a valley, hear a serene melody. Each landscape feature creates different emotions that are translated into music!

## Features

- 🏔️ **Any Terrain**: Use elevation data from your favorite places
- 🧠 **AI-Powered**: Neural networks help translate landscape to emotions
- 🎹 **Custom Music**: Generates unique MIDI compositions
- 🎨 **Visualization**: See the relationship between terrain and music
- 🔊 **Play It**: Listen to your landscape directly in the app

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/emotional-landscape-composer.git
cd emotional-landscape-composer

# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the example script
python examples/simple_composition.py
```

## How Does It Work?

1. **Terrain Analysis**: The program analyzes elevation data to find features like mountains, valleys, and slopes
2. **Emotion Mapping**: These features are mapped to emotions using neural networks
3. **Music Generation**: The emotions are translated into musical elements like melody, harmony, and rhythm
4. **Output**: The result is a beautiful musical piece that captures the "feeling" of the landscape

## Example

Here's a simple example of how to use the composer:

```python
from emotional_landscape_composer import LandscapeComposer

# Create a composer with a digital elevation model file
composer = LandscapeComposer('mountain_range.tif')

# Analyze the terrain
composer.analyze_terrain()

# Generate a 2-minute musical piece
composer.generate_composition(duration_minutes=2)

# Play the composition
composer.play()

# Save it as a MIDI file
composer.save('mountain_music.mid')
```

## Cool Ideas to Try

- 🌋 Generate music from famous mountains or landscapes
- 🏙️ Compare natural vs. urban terrain music
- 🌊 Create a "journey" by connecting multiple landscapes
- 🎞️ Make a video syncing landscape flyovers with the generated music
- 🏔️ Use real hiking GPS data to create a musical diary of your journey

## Dependencies

This project uses:
- TensorFlow (for the neural nets)
- GeoPandas (for geographic data)
- Hugging Face Transformers (for the emotion models)
- MIDIUtil & Pygame (for music generation and playback)

## Installation

```bash
# Option 1: Install directly
pip install -e .

# Option 2: Install dependencies only
pip install -r requirements.txt
```

## Contributing

This is a hobby project, so feel free to play around, modify, and improve it! Some ideas:
- Add new emotion mappings
- Create better visualization tools
- Improve the music generation algorithms
- Add support for more geographic data formats

## License

This project is under the MIT License - do whatever you want with it and have fun!
