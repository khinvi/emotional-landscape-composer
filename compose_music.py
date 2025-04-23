#!/usr/bin/env python
"""
Emotional Landscape Composer - Main Entry Point

This is a simple command-line interface for turning landscapes into music.
Run it directly to transform geographic data into musical compositions!
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from emotional_landscape_composer import LandscapeComposer, load_geotiff


def print_fancy_title():
    """Print a fancy ASCII art title."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                                                          ║")
    print("║   🏔️  EMOTIONAL LANDSCAPE COMPOSER  🎵                  ║")
    print("║   Turn your favorite landscapes into beautiful music!    ║")
    print("║                                                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\n")


def create_synthetic_landscape(landscape_type):
    """Create a synthetic landscape of the specified type."""
    print(f"Creating a synthetic {landscape_type} landscape...")
    
    size = 100
    x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))
    
    if landscape_type == "mountain":
        # Create a single mountain peak
        elevation = np.exp(-0.3 * (x**2 + y**2))
        title = "Mountain Peak"
        
    elif landscape_type == "valley":
        # Create a valley between mountains
        elevation = 1 - np.exp(-0.3 * (x**2 + y**2))
        elevation = elevation / np.max(elevation)
        title = "Valley"
        
    elif landscape_type == "hills":
        # Create rolling hills
        elevation = (0.3 * np.sin(x) * np.cos(y) + 
                     0.3 * np.sin(x*2) * np.cos(y*2) + 
                     0.3 * np.sin(x*4) * np.cos(y*4))
        elevation = (elevation - np.min(elevation)) / (np.max(elevation) - np.min(elevation))
        title = "Rolling Hills"
        
    elif landscape_type == "canyon":
        # Create a canyon
        ridge = np.exp(-0.1 * x**2) 
        canyon = np.exp(-5 * y**2)
        elevation = ridge * (1 - canyon)
        elevation = (elevation - np.min(elevation)) / (np.max(elevation) - np.min(elevation))
        title = "Canyon"
        
    elif landscape_type == "archipelago":
        # Create an archipelago of small islands
        elevation = np.zeros_like(x)
        for i in range(10):
            cx = np.random.uniform(-2, 2)
            cy = np.random.uniform(-2, 2)
            size = np.random.uniform(0.1, 0.3)
            elevation += np.exp(-((x-cx)**2 + (y-cy)**2) / size)
        elevation = (elevation - np.min(elevation)) / (np.max(elevation) - np.min(elevation))
        elevation = elevation**0.5  # Make islands more distinct
        title = "Archipelago"
        
    else:  # "random" or any other value
        # Create a random, complex landscape
        elevation = np.zeros_like(x)
        for i in range(5):
            scale = 1/(i+1)
            elevation += scale * np.sin(x * (i+1) * 1.5) * np.cos(y * (i+1) * 1.5)
        elevation = (elevation - np.min(elevation)) / (np.max(elevation) - np.min(elevation))
        title = "Random Landscape"
    
    # Visualize the landscape
    plt.figure(figsize=(10, 8))
    plt.imshow(elevation, cmap='terrain')
    plt.colorbar(label='Elevation')
    plt.title(title)
    out_file = f"{landscape_type}_terrain.png"
    plt.savefig(out_file)
    plt.close()
    print(f"✅ Terrain image saved as '{out_file}'")
    
    return {
        'elevation': elevation,
        'source_type': 'array',
        'crs': None,
        'bounds': None
    }


def main():
    """Run the main program."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Transform landscapes into musical compositions"
    )
    parser.add_argument(
        "--input", "-i", 
        help="Input GeoTIFF file (if not provided, creates a synthetic landscape)"
    )
    parser.add_argument(
        "--output", "-o", 
        default="landscape_music.mid",
        help="Output MIDI file name (default: landscape_music.mid)"
    )
    parser.add_argument(
        "--duration", "-d", 
        type=float, 
        default=2.0,
        help="Duration of the composition in minutes (default: 2.0)"
    )
    parser.add_argument(
        "--key", "-k",
        default="C_major",
        choices=[
            "C_major", "C_minor", "C_dorian", "C_phrygian",
            "C_lydian", "C_mixolydian", "C_pentatonic", "C_blues"
        ],
        help="Musical key for the composition (default: C_major)"
    )
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["base", "transformer_xl", "bert"],
        help="Type of emotion model to use (default: base)"
    )
    parser.add_argument(
        "--landscape", "-l",
        default="mountain",
        choices=["mountain", "valley", "hills", "canyon", "archipelago", "random"],
        help="Type of synthetic landscape to create if no input file (default: mountain)"
    )
    parser.add_argument(
        "--play", "-p",
        action="store_true",
        help="Play the composition after generating it"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Visualize the features and composition"
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print_fancy_title()
    
    # Load or create the landscape data
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found!")
            return
            
        print(f"Loading landscape from '{args.input}'...")
        terrain_data = load_geotiff(args.input)
    else:
        terrain_data = create_synthetic_landscape(args.landscape)
    
    # Create the composer
    print("\n🎵 Creating the landscape composer...")
    composer = LandscapeComposer(
        data_source=terrain_data,
        key=args.key,
        emotion_model=args.model
    )
    
    # Analyze terrain
    print("\n🔍 Analyzing the terrain features...")
    composer.analyze_terrain(visualize=args.visualize)
    
    # Generate the composition
    print(f"\n🎹 Generating a {args.duration}-minute composition in {args.key}...")
    composer.generate_composition(duration_minutes=args.duration)
    
    # Visualize the composition if requested
    if args.visualize:
        print("\n📊 Creating a visualization of the composition...")
        composer.visualize_composition()
    
    # Save the composition
    print(f"\n💾 Saving composition to '{args.output}'")
    composer.save(args.output)
    
    # Try to export as WAV
    try:
        wav_file = os.path.splitext(args.output)[0] + ".wav"
        print(f"\n🔊 Exporting to WAV: '{wav_file}'")
        composer.export(wav_file, format='wav')
        print("✅ WAV file created successfully!")
    except Exception as e:
        print(f"❓ WAV export didn't work: {e}")
        print("💡 To enable WAV export, install FluidSynth.")
    
    # Play the composition if requested
    if args.play:
        print("\n🎧 Playing the composition... (Press Ctrl+C to stop)")
        try:
            composer.play()
        except KeyboardInterrupt:
            print("\n⏹️  Playback stopped.")
    
    print("\n🎉 All done! Thanks for using the Emotional Landscape Composer!")
    print("For more information, check out the README.md and docs folder.")


if __name__ == "__main__":
    main()