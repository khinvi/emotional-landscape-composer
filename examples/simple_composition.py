#!/usr/bin/env python
"""
Make some musical mountains! This example creates a simple
synthetic landscape and turns it into music.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from emotional_landscape_composer import LandscapeComposer


def create_cool_landscape():
    """Create a fun synthetic landscape with mountains and valleys."""
    print("🏔️  Creating a cool synthetic landscape...")
    
    # Create a simple terrain with mountains and valleys
    size = 100
    x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
    
    # Create a mountain range
    mountains = np.exp(-0.1 * (x**2 + y**2))
    
    # Add some random hills
    hills = (0.3 * np.sin(x * 3) * np.cos(y * 2) + 
             0.1 * np.sin(x * 8) * np.cos(y * 7) + 
             0.1 * np.sin(x * 15) * np.sin(y * 15))
    
    # Add a valley
    valley = 0.3 * np.exp(-0.2 * ((x - 1)**2 + y**2))
    
    # Combine features
    elevation = mountains + hills - valley
    
    # Normalize to [0, 1]
    elevation = (elevation - np.min(elevation)) / (np.max(elevation) - np.min(elevation))
    
    # Visualize the synthetic terrain
    plt.figure(figsize=(10, 8))
    plt.imshow(elevation, cmap='terrain')
    plt.colorbar(label='Elevation')
    plt.title('Our Awesome Synthetic Terrain')
    plt.savefig('synthetic_terrain.png')
    print("✅ Terrain image saved as 'synthetic_terrain.png'")
    plt.close()
    
    return {
        'elevation': elevation,
        'source_type': 'array',
        'crs': None,
        'bounds': None
    }


def main():
    """Run the entire music generation process."""
    # Create a cool landscape
    terrain_data = create_cool_landscape()
    
    print("\n🎵 Creating our landscape composer...")
    
    # Create a composer with the synthetic terrain
    composer = LandscapeComposer(
        data_source=terrain_data,
        emotion_model='transformer_xl',  # The fancy version
        tempo_range=(70, 110),          # Not too slow, not too fast
        key='C_lydian'                  # A dreamy, magical-sounding key
    )
    
    # Analyze terrain and visualize features
    print("\n🔍 Analyzing the terrain features...")
    features = composer.analyze_terrain(visualize=True)
    
    # Generate a 2-minute composition
    print("\n🎹 Generating music from our landscape...")
    composition = composer.generate_composition(duration_minutes=2.0)
    
    # Visualize the composition
    print("\n📊 Creating a visualization of our music...")
    composer.visualize_composition()
    
    # Save the composition as a MIDI file
    output_file = 'mountain_music.mid'
    print(f"\n💾 Saving our musical creation to '{output_file}'")
    composer.save(output_file)
    
    # Optionally export as WAV if FluidSynth is installed
    try:
        wav_file = 'mountain_music.wav'
        print(f"\n🔊 Exporting to WAV: '{wav_file}'")
        composer.export(wav_file, format='wav')
        print("✅ WAV file created successfully!")
    except Exception as e:
        print(f"❌ WAV export didn't work: {e}")
        print("💡 To enable WAV export, install FluidSynth.")
    
    print("\n🎉 All done! Try playing your landscape music in any MIDI player.")
    print("💡 Tip: You can also run 'composer.play()' to hear it directly!")


if __name__ == "__main__":
    print("="*60)
    print("🏔️ 🎵  EMOTIONAL LANDSCAPE COMPOSER  🎵 🏔️")
    print("="*60)
    print("Creating music from landscapes - how cool is that?!\n")
    main()
    print("\nThanks for using the Emotional Landscape Composer!")
    print("="*60)