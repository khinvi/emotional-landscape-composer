"""
Main Composer class for the Emotional Landscape Composer.

This is the heart of the project - it ties together all the components
and provides a simple interface for turning landscapes into music.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from midiutil import MIDIFile
import pygame
import time
from typing import Optional, Union, Dict, Tuple, List, Any

from .data_processor import DataProcessor
from .feature_extractor import TerrainFeatureExtractor
from .emotion_interpreter import EmotionModel
from .music_generator import MusicGenerator

class LandscapeComposer:
    """
    Main class for the Emotional Landscape Composer system.
    
    This class brings everything together to transform geographical data
    into musical compositions based on the emotional character of the landscape.
    """
    
    def __init__(
        self, 
        data_source: Union[str, np.ndarray, Dict[str, Any]], 
        emotion_model: str = 'base',
        tempo_range: Tuple[int, int] = (60, 120),
        key: str = 'C_major',
        resolution: int = 128,
        **kwargs
    ):
        """
        Initialize the Landscape Composer.
        
        Args:
            data_source: Path to a GeoTIFF file, NumPy array, or dictionary with data
            emotion_model: Model name for emotion interpretation ('base', 'transformer_xl', 'bert')
            tempo_range: Range of tempos (min, max) in beats per minute
            key: Musical key for the composition (e.g., 'C_major', 'A_minor')
            resolution: Resolution for feature extraction
            **kwargs: Additional parameters for the components
        """
        self.data_processor = DataProcessor()
        self.terrain_extractor = TerrainFeatureExtractor(resolution=resolution)
        self.emotion_model = EmotionModel(model_name=emotion_model)
        self.music_generator = MusicGenerator(
            tempo_range=tempo_range,
            key=key,
            **kwargs
        )
        
        # Load the data
        self.data = self.data_processor.load(data_source)
        
        # Initialize state variables
        self.terrain_features = None
        self.emotional_mapping = None
        self.composition = None
        self.midi_data = None
        
        # Configuration
        self.config = {
            'tempo_range': tempo_range,
            'key': key,
            'resolution': resolution,
            'emotion_model': emotion_model,
            **kwargs
        }
    
    def analyze_terrain(self, visualize: bool = False) -> Dict[str, np.ndarray]:
        """
        Analyze the terrain data and extract features.
        
        Args:
            visualize: Whether to visualize the extracted features
            
        Returns:
            Dictionary of extracted terrain features
        """
        print("Analyzing terrain features...")
        self.terrain_features = self.terrain_extractor.extract_features(self.data)
        
        if visualize:
            self._visualize_features()
            
        return self.terrain_features
    
    def interpret_emotions(self) -> Dict[str, np.ndarray]:
        """
        Map terrain features to emotional qualities.
        
        Returns:
            Dictionary of emotional mappings
        """
        if self.terrain_features is None:
            self.analyze_terrain()
            
        print("Interpreting emotional qualities of landscape...")
        self.emotional_mapping = self.emotion_model.interpret(self.terrain_features)
        return self.emotional_mapping
    
    def generate_composition(self, duration_minutes: float = 3.0) -> Dict[str, Any]:
        """
        Generate a musical composition based on the emotional mapping.
        
        Args:
            duration_minutes: Target duration of the composition in minutes
            
        Returns:
            Dictionary containing the composition data
        """
        if self.emotional_mapping is None:
            self.interpret_emotions()
            
        print(f"Generating {duration_minutes:.1f}-minute composition...")
        self.composition = self.music_generator.generate(
            self.emotional_mapping, 
            duration_minutes=duration_minutes
        )
        
        # Create MIDI data
        self.midi_data = self._create_midi()
        
        return self.composition
    
    def compose(self, duration_minutes: float = 3.0, visualize: bool = False) -> Dict[str, Any]:
        """
        Execute the full composition process in one step.
        
        Args:
            duration_minutes: Target duration of the composition in minutes
            visualize: Whether to visualize terrain features
            
        Returns:
            Dictionary containing the composition data
        """
        self.analyze_terrain(visualize=visualize)
        self.interpret_emotions()
        return self.generate_composition(duration_minutes=duration_minutes)
    
    def play(self, autoplay: bool = True) -> None:
        """
        Play the generated composition.
        
        Args:
            autoplay: Whether to start playback automatically
        """
        if self.composition is None:
            raise ValueError("No composition has been generated yet. Call generate_composition() first.")
        
        # Save to a temporary file
        temp_file = "_temp_composition.mid"
        self.save(temp_file)
        
        # Initialize pygame for playback
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        
        print(f"Playing composition (duration: {self.composition['duration_seconds'] / 60:.1f} minutes)")
        if autoplay:
            pygame.mixer.music.play()
            
            # Keep the program running while the music plays
            try:
                while pygame.mixer.music.get_busy():
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pygame.mixer.music.stop()
                print("\nPlayback stopped by user.")
            
            # Clean up
            pygame.mixer.quit()
            if os.path.exists(temp_file):
                os.remove(temp_file)
        else:
            print("Ready to play. Use pygame.mixer.music controls for playback.")
    
    def save(self, filename: str) -> None:
        """
        Save the composition to a MIDI file.
        
        Args:
            filename: Path where the MIDI file will be saved
        """
        if self.midi_data is None:
            if self.composition is None:
                raise ValueError("No composition has been generated yet. Call generate_composition() first.")
            self.midi_data = self._create_midi()
            
        with open(filename, 'wb') as output_file:
            self.midi_data.writeFile(output_file)
        print(f"Composition saved to {filename}")
    
    def export(self, filename: str, format: str = None) -> None:
        """
        Export the composition to various audio formats.
        
        Args:
            filename: Output filename
            format: Output format (MIDI, WAV, MP3); if None, inferred from filename
        """
        if format is None:
            # Infer format from filename
            _, ext = os.path.splitext(filename)
            format = ext.lower().lstrip('.')
        
        if format == 'mid' or format == 'midi':
            self.save(filename)
        elif format == 'wav':
            # First save as MIDI
            midi_file = f"_temp_{int(time.time())}.mid"
            self.save(midi_file)
            
            # Convert MIDI to WAV using fluidsynth (requires external library)
            try:
                import subprocess
                soundfont = os.path.join(os.path.dirname(__file__), 'resources', 'soundfont.sf2')
                subprocess.run([
                    'fluidsynth', '-ni', soundfont, midi_file, 
                    '-F', filename, '-r', '44100'
                ])
                print(f"Composition exported to {filename}")
            except Exception as e:
                print(f"Error exporting to WAV: {e}")
                print("Make sure FluidSynth is installed on your system.")
            finally:
                if os.path.exists(midi_file):
                    os.remove(midi_file)
        elif format == 'mp3':
            # First export to WAV
            wav_file = f"_temp_{int(time.time())}.wav"
            self.export(wav_file, format='wav')
            
            # Convert WAV to MP3
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(wav_file)
                audio.export(filename, format='mp3')
                print(f"Composition exported to {filename}")
            except Exception as e:
                print(f"Error exporting to MP3: {e}")
                print("Make sure PyDub and FFmpeg are installed on your system.")
            finally:
                if os.path.exists(wav_file):
                    os.remove(wav_file)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _create_midi(self) -> MIDIFile:
        """
        Create a MIDI file from the composition.
        
        Returns:
            MIDIFile object with the composition
        """
        # Extract composition elements
        tracks = self.composition['tracks']
        tempo = self.composition['tempo']
        
        # Create MIDI file with one track per instrument
        midi = MIDIFile(len(tracks))
        
        # Add notes from each track
        for i, track in enumerate(tracks):
            # Set track name and tempo
            midi.addTrackName(i, 0, track['name'])
            midi.addTempo(i, 0, tempo)
            
            # Add notes
            for note in track['notes']:
                # Ensure pitch is within valid MIDI range (0-127)
                pitch = max(0, min(127, note['pitch']))
                
                # Ensure other values are valid
                velocity = max(0, min(127, note['velocity']))
                time = max(0, note['time'])
                duration = max(0.1, note['duration'])  # Avoid zero duration
                
                midi.addNote(
                    track=i,
                    channel=0,
                    pitch=pitch,
                    time=time,
                    duration=duration,
                    volume=velocity
                )
                
        return midi
    
    def _visualize_features(self) -> None:
        """
        Visualize the extracted terrain features.
        """
        if self.terrain_features is None:
            raise ValueError("No terrain features available. Call analyze_terrain() first.")
        
        # Create a figure with subplots for each feature
        n_features = len(self.terrain_features)
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4))
        
        if n_features == 1:
            axes = [axes]
            
        for ax, (feature_name, feature_data) in zip(axes, self.terrain_features.items()):
            im = ax.imshow(feature_data, cmap='viridis')
            ax.set_title(feature_name)
            plt.colorbar(im, ax=ax)
            
        plt.tight_layout()
        plt.show()

    def visualize_composition(self, show_terrain: bool = True):
        """
        Visualize the composition alongside the terrain data.
        
        Args:
            show_terrain: Whether to show the terrain data alongside the music
        """
        if self.composition is None:
            raise ValueError("No composition has been generated yet. Call generate_composition() first.")
            
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
        
        # Plot terrain elevation in top subplot if available
        if show_terrain and 'elevation' in self.terrain_features:
            im = axes[0].imshow(self.terrain_features['elevation'], cmap='terrain')
            axes[0].set_title('Terrain Elevation')
            plt.colorbar(im, ax=axes[0])
            
        # Plot musical notes in piano roll format in bottom subplot
        pitch_min, pitch_max = 127, 0
        for track in self.composition['tracks']:
            for note in track['notes']:
                pitch = note['pitch']
                pitch_min = min(pitch_min, pitch)
                pitch_max = max(pitch_max, pitch)
                axes[1].add_patch(plt.Rectangle(
                    (note['time'], pitch),
                    note['duration'],
                    1,
                    color=track.get('color', 'blue'),
                    alpha=0.7
                ))
        
        # Set piano roll axis limits and labels
        axes[1].set_ylim(pitch_min - 2, pitch_max + 2)
        axes[1].set_xlim(0, self.composition['duration_seconds'])
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Pitch (MIDI note number)')
        axes[1].set_title('Musical Composition')
        axes[1].grid(True, alpha=0.3)
        
        # Add legend for tracks
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=track.get('color', 'blue'), 
                  alpha=0.7,
                  label=track['name'])
            for track in self.composition['tracks']
        ]
        axes[1].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()