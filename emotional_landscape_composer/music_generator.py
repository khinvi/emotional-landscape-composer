"""
Music generation module for the Emotional Landscape Composer.

This is where the magic happens! This module takes the emotional
interpretations of landscapes and turns them into actual music.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import math
import random


class MusicGenerator:
    """
    Generate music based on emotional mappings of terrain.
    """
    
    # Define musical scales
    SCALES = {
        'C_major': [60, 62, 64, 65, 67, 69, 71, 72],  # C D E F G A B C
        'C_minor': [60, 62, 63, 65, 67, 68, 70, 72],  # C D Eb F G Ab Bb C
        'C_dorian': [60, 62, 63, 65, 67, 69, 70, 72], # C D Eb F G A Bb C
        'C_phrygian': [60, 61, 63, 65, 67, 68, 70, 72], # C Db Eb F G Ab Bb C
        'C_lydian': [60, 62, 64, 66, 67, 69, 71, 72], # C D E F# G A B C
        'C_mixolydian': [60, 62, 64, 65, 67, 69, 70, 72], # C D E F G A Bb C
        'C_aeolian': [60, 62, 63, 65, 67, 68, 70, 72], # C D Eb F G Ab Bb C
        'C_locrian': [60, 61, 63, 65, 66, 68, 70, 72], # C Db Eb F Gb Ab Bb C
        'C_pentatonic': [60, 62, 64, 67, 69, 72],     # C D E G A C
        'C_minor_pentatonic': [60, 63, 65, 67, 70, 72], # C Eb F G Bb C
        'C_blues': [60, 63, 65, 66, 67, 70, 72],      # C Eb F F# G Bb C
    }
    
    # Define instrument types (MIDI programs)
    INSTRUMENTS = {
        'piano': 0,        # Acoustic Grand Piano
        'marimba': 12,     # Marimba
        'organ': 19,       # Church Organ
        'guitar': 24,      # Acoustic Guitar (nylon)
        'bass': 32,        # Acoustic Bass
        'strings': 48,     # String Ensemble 1
        'choir': 52,       # Choir Aahs
        'trumpet': 56,     # Trumpet
        'flute': 73,       # Flute
        'pad': 88,         # Pad 1 (new age)
        'percussion': 115  # Steel Drums
    }
    
    # Emotional associations with instruments
    EMOTION_INSTRUMENTS = {
        'serenity': ['piano', 'flute', 'strings'],
        'awe': ['organ', 'choir', 'strings'],
        'joy': ['marimba', 'piano', 'flute'],
        'melancholy': ['piano', 'strings', 'guitar'],
        'tension': ['strings', 'pad', 'percussion'],
        'power': ['organ', 'strings', 'trumpet'],
        'mysteriousness': ['pad', 'choir', 'flute'],
        'harshness': ['percussion', 'trumpet', 'organ']
    }
    
    # Emotional associations with scales
    EMOTION_SCALES = {
        'serenity': ['C_major', 'C_lydian'],
        'awe': ['C_lydian', 'C_major'],
        'joy': ['C_major', 'C_pentatonic', 'C_lydian'],
        'melancholy': ['C_minor', 'C_aeolian', 'C_dorian'],
        'tension': ['C_phrygian', 'C_locrian'],
        'power': ['C_mixolydian', 'C_dorian'],
        'mysteriousness': ['C_phrygian', 'C_locrian', 'C_pentatonic'],
        'harshness': ['C_locrian', 'C_blues']
    }
    
    def __init__(
        self, 
        tempo_range: Tuple[int, int] = (60, 120),
        key: str = 'C_major',
        use_tensorflow: bool = True,
        **kwargs
    ):
        """
        Initialize the music generator.
        
        Args:
            tempo_range: Range of tempos (min, max) in beats per minute
            key: Musical key for the composition
            use_tensorflow: Whether to use TensorFlow for generation
            **kwargs: Additional parameters
        """
        self.tempo_range = tempo_range
        self.key = key
        self.use_tensorflow = use_tensorflow
        
        # Check if key is valid
        if key not in self.SCALES:
            valid_keys = list(self.SCALES.keys())
            raise ValueError(f"Invalid key: {key}. Valid keys are: {valid_keys}")
        
        # Additional parameters
        self.transpose = kwargs.get('transpose', 0)
        self.duration_range = kwargs.get('duration_range', (0.25, 2.0))
        self.dynamic_range = kwargs.get('dynamic_range', (40, 100))
        self.density_range = kwargs.get('density_range', (1, 8))
        
        # Initialize TensorFlow model if needed
        if use_tensorflow:
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        Initialize a TensorFlow model for music generation.
        """
        try:
            # Define a simple LSTM model for melody generation
            input_dim = 128  # MIDI note range
            
            # Create model
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(None, input_dim)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(input_dim, activation='softmax')
            ])
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Generate some random weights (in a real scenario these would be trained)
            # But we'll use a simpler algorithm for generation anyway
            print("TensorFlow model initialized (note: using algorithmic generation)")
            
        except Exception as e:
            print(f"Error initializing TensorFlow model: {e}")
            print("Falling back to algorithmic generation")
            self.use_tensorflow = False
    
    def generate(
        self, 
        emotions: Dict[str, np.ndarray], 
        duration_minutes: float = 3.0
    ) -> Dict[str, Any]:
        """
        Generate a musical composition based on emotional mappings.
        
        Args:
            emotions: Dictionary of emotional mappings
            duration_minutes: Target duration of the composition in minutes
            
        Returns:
            Dictionary containing the composition data
        """
        # Calculate overall emotional character
        emotion_means = {name: float(np.mean(values)) for name, values in emotions.items()}
        total = sum(emotion_means.values())
        if total > 0:
            emotion_means = {name: strength / total for name, strength in emotion_means.items()}
        
        # Sort emotions by strength
        dominant_emotions = sorted(
            emotion_means.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select the top emotions for the composition
        top_emotions = dominant_emotions[:3]
        print(f"Top emotions: {[name for name, _ in top_emotions]}")
        
        # Determine the tempo based on emotional character
        tempo = self._determine_tempo(emotion_means)
        
        # Determine the scale based on emotional character
        scale = self._determine_scale(emotion_means)
        scale_notes = self._get_scale(scale, self.transpose)
        
        # Calculate beat duration in seconds
        beat_duration = 60.0 / tempo
        
        # Calculate total number of beats
        total_beats = int(duration_minutes * 60 / beat_duration)
        
        # Generate tracks for each dominant emotion
        tracks = []
        
        # Select instruments based on emotions
        used_instruments = set()
        
        for emotion_name, emotion_strength in top_emotions:
            # Select instruments for this emotion
            instruments = self._select_instruments(emotion_name, used_instruments)
            
            for instrument in instruments:
                # Generate a track for this instrument
                track = self._generate_track(
                    emotion_name=emotion_name,
                    emotion_data=emotions[emotion_name],
                    emotion_strength=emotion_strength,
                    instrument=instrument,
                    scale_notes=scale_notes,
                    total_beats=total_beats,
                    beat_duration=beat_duration
                )
                
                tracks.append(track)
                used_instruments.add(instrument)
        
        # Add bass track
        if 'bass' not in used_instruments:
            bass_track = self._generate_bass_track(
                dominant_emotion=top_emotions[0][0],
                scale_notes=scale_notes,
                total_beats=total_beats,
                beat_duration=beat_duration
            )
            tracks.append(bass_track)
            used_instruments.add('bass')
        
        # Add percussion track
        if 'percussion' not in used_instruments and random.random() < 0.7:
            percussion_track = self._generate_percussion_track(
                dominant_emotion=top_emotions[0][0],
                total_beats=total_beats,
                beat_duration=beat_duration
            )
            tracks.append(percussion_track)
            used_instruments.add('percussion')
        
        # Create the composition
        composition = {
            'tracks': tracks,
            'tempo': tempo,
            'scale': scale,
            'key': self.key,
            'emotions': {name: float(strength) for name, strength in top_emotions},
            'duration_seconds': total_beats * beat_duration,
            'num_beats': total_beats,
            'beat_duration': beat_duration
        }
        
        return composition
    
    def _determine_tempo(self, emotions: Dict[str, float]) -> int:
        """
        Determine the tempo based on emotional character.
        
        Args:
            emotions: Dictionary of emotion strengths
            
        Returns:
            Tempo in beats per minute
        """
        min_tempo, max_tempo = self.tempo_range
        
        # Define emotional influences on tempo
        tempo_influences = {
            'serenity': -0.7,   # Slower
            'awe': -0.3,        # Slightly slower
            'joy': 0.5,         # Faster
            'melancholy': -0.8, # Much slower
            'tension': 0.2,     # Slightly faster
            'power': 0.4,       # Faster
            'mysteriousness': -0.4, # Slower
            'harshness': 0.6    # Faster
        }
        
        # Calculate tempo modifier
        tempo_mod = 0.0
        for emotion, strength in emotions.items():
            if emotion in tempo_influences:
                tempo_mod += tempo_influences[emotion] * strength
        
        # Clamp to [-1, 1]
        tempo_mod = max(-1.0, min(1.0, tempo_mod))
        
        # Map to tempo range
        tempo_range = max_tempo - min_tempo
        tempo = int(min_tempo + (tempo_range * (tempo_mod + 1) / 2))
        
        return tempo
    
    def _determine_scale(self, emotions: Dict[str, float]) -> str:
        """
        Determine the musical scale based on emotional character.
        
        Args:
            emotions: Dictionary of emotion strengths
            
        Returns:
            Scale name
        """
        # Get candidate scales
        candidate_scales = set()
        for emotion, strength in emotions.items():
            if emotion in self.EMOTION_SCALES and strength > 0.1:
                for scale in self.EMOTION_SCALES[emotion]:
                    candidate_scales.add(scale)
        
        # If no candidates, use the default key
        if not candidate_scales:
            return self.key
        
        # Find the scale most associated with the dominant emotions
        scale_scores = {scale: 0.0 for scale in candidate_scales}
        
        for emotion, strength in emotions.items():
            if emotion in self.EMOTION_SCALES:
                for scale in self.EMOTION_SCALES[emotion]:
                    if scale in scale_scores:
                        # Higher weight for stronger emotions
                        scale_scores[scale] += strength
        
        # Select the highest scoring scale
        if scale_scores:
            return max(scale_scores.items(), key=lambda x: x[1])[0]
        
        return self.key
    
    def _get_scale(self, scale_name: str, transpose: int = 0) -> List[int]:
        """
        Get the MIDI note numbers for a scale, with optional transposition.
        
        Args:
            scale_name: Name of the scale
            transpose: Number of semitones to transpose
            
        Returns:
            List of MIDI note numbers
        """
        if scale_name not in self.SCALES:
            scale_name = self.key
            
        # Get the base scale
        scale = self.SCALES[scale_name]
        
        # Apply transposition
        transposed_scale = [note + transpose for note in scale]
        
        # Add more octaves (3 octaves total)
        full_scale = []
        for octave in [-1, 0, 1]:
            full_scale.extend([note + (12 * octave) for note in transposed_scale])
        
        return full_scale
    
    def _select_instruments(
        self, 
        emotion: str, 
        used_instruments: set
    ) -> List[str]:
        """
        Select instruments for an emotion that aren't already in use.
        
        Args:
            emotion: Emotion name
            used_instruments: Set of instruments already in use
            
        Returns:
            List of selected instruments
        """
        if emotion not in self.EMOTION_INSTRUMENTS:
            candidate_instruments = list(self.INSTRUMENTS.keys())
        else:
            candidate_instruments = self.EMOTION_INSTRUMENTS[emotion]
        
        # Filter out already used instruments
        available_instruments = [
            inst for inst in candidate_instruments 
            if inst not in used_instruments
        ]
        
        # If all candidate instruments are used, pick any unused instrument
        if not available_instruments:
            all_instruments = list(self.INSTRUMENTS.keys())
            available_instruments = [
                inst for inst in all_instruments 
                if inst not in used_instruments
            ]
        
        # If still no instruments available, reuse one
        if not available_instruments:
            return [candidate_instruments[0]]
        
        # Pick 1-2 instruments
        num_instruments = min(len(available_instruments), random.randint(1, 2))
        selected_instruments = random.sample(available_instruments, num_instruments)
        
        return selected_instruments
    
    def _generate_track(
        self,
        emotion_name: str,
        emotion_data: np.ndarray,
        emotion_strength: float,
        instrument: str,
        scale_notes: List[int],
        total_beats: int,
        beat_duration: float
    ) -> Dict[str, Any]:
        """
        Generate a track for a specific instrument based on an emotion.
        
        Args:
            emotion_name: Name of the emotion
            emotion_data: 2D array of emotion values
            emotion_strength: Overall strength of the emotion
            instrument: Instrument name
            scale_notes: List of MIDI note numbers in the scale
            total_beats: Total number of beats in the composition
            beat_duration: Duration of each beat in seconds
            
        Returns:
            Track data dictionary
        """
        # Get instrument program number
        program = self.INSTRUMENTS[instrument]
        
        # Determine track characteristics based on emotion
        density = self._determine_note_density(emotion_name, emotion_strength)
        duration_range = self._determine_duration_range(emotion_name)
        velocity_range = self._determine_velocity_range(emotion_name)
        register = self._determine_register(emotion_name, instrument)
        
        # Generate notes
        notes = []
        
        # Current position in beats
        current_beat = 0.0
        
        while current_beat < total_beats:
            # Determine the next note onset
            if random.random() < 0.3:
                # Occasional rest
                current_beat += random.uniform(0.5, 1.5)
                continue
            
            # Select note pitch
            if self.use_tensorflow and random.random() < 0.3:
                # Sometimes use TensorFlow model for note prediction
                # (In a real implementation, this would be more sophisticated)
                pitch = self._predict_next_note(notes, scale_notes, register)
            else:
                # Otherwise use algorithmic approach
                pitch = self._select_note_from_scale(scale_notes, register)
            
            # Determine note duration
            duration = random.uniform(duration_range[0], duration_range[1])
            
            # Determine note velocity (volume)
            velocity = random.randint(velocity_range[0], velocity_range[1])
            
            # Add the note
            notes.append({
                'pitch': pitch,
                'time': current_beat * beat_duration,  # Convert to seconds
                'duration': duration * beat_duration,  # Convert to seconds
                'velocity': velocity
            })
            
            # Move to next position
            current_beat += duration * random.uniform(0.8, 1.2)  # Add some timing variation
        
        # Sort notes by time
        notes.sort(key=lambda x: x['time'])
        
        # Create track data
        track = {
            'name': f"{emotion_name.capitalize()} {instrument}",
            'program': program,
            'notes': notes,
            'emotion': emotion_name,
            'color': self._get_emotion_color(emotion_name)
        }
        
        return track
    
    def _generate_bass_track(
        self,
        dominant_emotion: str,
        scale_notes: List[int],
        total_beats: int,
        beat_duration: float
    ) -> Dict[str, Any]:
        """
        Generate a bass track for the composition.
        
        Args:
            dominant_emotion: Name of the dominant emotion
            scale_notes: List of MIDI note numbers in the scale
            total_beats: Total number of beats in the composition
            beat_duration: Duration of each beat in seconds
            
        Returns:
            Bass track data dictionary
        """
        program = self.INSTRUMENTS['bass']
        
        # Bass is usually in a lower register
        register = (-2, -1)
        
        # Generate bass notes
        notes = []
        
        # Bass often plays on the strong beats
        current_beat = 0.0
        while current_beat < total_beats:
            # Select bass note (usually root, fifth, or other chord tone)
            bass_note_options = [0, 4, 7]  # Root, fourth, fifth scale degrees
            bass_note_idx = random.choice(bass_note_options)
            
            # Get the actual MIDI note
            # Make sure we have a valid note in the scale before calculating
            if len(scale_notes) == 0:
                # Emergency fallback - use middle C
                pitch = 36  # C2 (low C)
            else:
                # Ensure we use an existing note from the scale
                base_idx = 0
                while base_idx < len(scale_notes) and scale_notes[base_idx] < 60:
                    base_idx += 1
                if base_idx > 0:
                    base_idx -= 1  # Go back to a lower note
                
                base_pitch = scale_notes[base_idx % len(scale_notes)]
                
                # Calculate pitch based on root note and selected register
                pitch = (base_pitch % 12) + bass_note_idx + (12 * random.randint(register[0], register[1]))
                
                # Ensure pitch is within MIDI range (0-127)
                pitch = max(0, min(127, pitch))
            
            # Determine note duration (usually longer for bass)
            duration = random.uniform(1.0, 4.0)
            
            # Determine note velocity (volume)
            velocity = random.randint(70, 90)
            
            # Add the note
            notes.append({
                'pitch': pitch,
                'time': current_beat * beat_duration,  # Convert to seconds
                'duration': duration * beat_duration,  # Convert to seconds
                'velocity': velocity
            })
            
            # Move to next position
            current_beat += duration
        
        # Sort notes by time
        notes.sort(key=lambda x: x['time'])
        
        # Create track data
        track = {
            'name': "Bass",
            'program': program,
            'notes': notes,
            'emotion': dominant_emotion,
            'color': '#404040'  # Dark gray
        }
        
        return track
    
    def _generate_percussion_track(
        self,
        dominant_emotion: str,
        total_beats: int,
        beat_duration: float
    ) -> Dict[str, Any]:
        """
        Generate a percussion track for the composition.
        
        Args:
            dominant_emotion: Name of the dominant emotion
            total_beats: Total number of beats in the composition
            beat_duration: Duration of each beat in seconds
            
        Returns:
            Percussion track data dictionary
        """
        # MIDI percussion channel uses program 0 but channel 9
        program = 0
        
        # MIDI note numbers for percussion sounds
        DRUM_SOUNDS = {
            'kick': 36,
            'snare': 38,
            'closed_hat': 42,
            'open_hat': 46,
            'low_tom': 41,
            'mid_tom': 45,
            'high_tom': 48,
            'crash': 49,
            'ride': 51,
            'clap': 39
        }
        
        # Create different patterns based on the dominant emotion
        if dominant_emotion in ['tension', 'harshness']:
            # More complex, intense patterns
            pattern_length = random.choice([3, 5, 7])
            kick_pattern = [1 if i % pattern_length == 0 or random.random() < 0.2 else 0 
                           for i in range(16)]
            snare_pattern = [1 if i % pattern_length == (pattern_length // 2) else 0 
                            for i in range(16)]
            hat_pattern = [1 if i % 2 == 0 or random.random() < 0.3 else 0 
                          for i in range(16)]
        elif dominant_emotion in ['joy', 'power']:
            # Strong, regular patterns
            kick_pattern = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
            snare_pattern = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
            hat_pattern = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:
            # Subtle, sparse patterns
            kick_pattern = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
            snare_pattern = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
            hat_pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        
        # Generate percussion notes
        notes = []
        
        # Number of pattern repetitions
        num_patterns = math.ceil(total_beats / 4)  # Assuming 4/4 time
        
        for pattern_idx in range(num_patterns):
            for step in range(len(kick_pattern)):
                step_beat = pattern_idx * 4 + step * 0.25
                
                if step_beat >= total_beats:
                    break
                
                # Add kick drum notes
                if kick_pattern[step]:
                    notes.append({
                        'pitch': DRUM_SOUNDS['kick'],
                        'time': step_beat * beat_duration,
                        'duration': 0.25 * beat_duration,
                        'velocity': random.randint(80, 100)
                    })
                
                # Add snare drum notes
                if snare_pattern[step]:
                    notes.append({
                        'pitch': DRUM_SOUNDS['snare'],
                        'time': step_beat * beat_duration,
                        'duration': 0.25 * beat_duration,
                        'velocity': random.randint(70, 90)
                    })
                
                # Add hi-hat notes
                if hat_pattern[step]:
                    notes.append({
                        'pitch': DRUM_SOUNDS['closed_hat'],
                        'time': step_beat * beat_duration,
                        'duration': 0.25 * beat_duration,
                        'velocity': random.randint(50, 70)
                    })
        
        # Add some variation and fills
        for pattern_idx in range(num_patterns):
            # Add occasional crash cymbal at pattern start
            if random.random() < 0.2:
                notes.append({
                    'pitch': DRUM_SOUNDS['crash'],
                    'time': pattern_idx * 4 * beat_duration,
                    'duration': 1.0 * beat_duration,
                    'velocity': random.randint(70, 90)
                })
            
            # Add occasional fills at the end of patterns
            if random.random() < 0.3:
                fill_start = (pattern_idx * 4 + 3) * beat_duration
                for i in range(4):
                    drum = random.choice(['low_tom', 'mid_tom', 'high_tom', 'snare'])
                    notes.append({
                        'pitch': DRUM_SOUNDS[drum],
                        'time': fill_start + (i * 0.125 * beat_duration),
                        'duration': 0.1 * beat_duration,
                        'velocity': random.randint(70, 90)
                    })
        
        # Sort notes by time
        notes.sort(key=lambda x: x['time'])
        
        # Create track data
        track = {
            'name': "Percussion",
            'program': program,
            'notes': notes,
            'emotion': dominant_emotion,
            'color': '#808080'  # Gray
        }
        
        return track
    
    def _determine_note_density(
        self, 
        emotion: str, 
        emotion_strength: float
    ) -> float:
        """
        Determine the density of notes based on emotion.
        
        Args:
            emotion: Emotion name
            emotion_strength: Strength of the emotion
            
        Returns:
            Note density (notes per beat)
        """
        # Define base densities for emotions
        base_densities = {
            'serenity': 0.5,    # Sparse, peaceful
            'awe': 0.8,         # Moderate
            'joy': 1.2,         # Dense, energetic
            'melancholy': 0.4,  # Sparse, thoughtful
            'tension': 1.0,     # Moderate to dense
            'power': 1.5,       # Dense, imposing
            'mysteriousness': 0.6, # Moderate
            'harshness': 1.8    # Very dense
        }
        
        # Get base density for this emotion
        density = base_densities.get(emotion, 1.0)
        
        # Scale by emotion strength
        density *= emotion_strength
        
        # Add some randomness
        density *= random.uniform(0.8, 1.2)
        
        return density
    
    def _determine_duration_range(self, emotion: str) -> Tuple[float, float]:
        """
        Determine the range of note durations based on emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Tuple of (min_duration, max_duration) in beats
        """
        # Define duration ranges for emotions
        duration_ranges = {
            'serenity': (1.0, 4.0),     # Longer, flowing notes
            'awe': (0.5, 3.0),          # Moderate to long
            'joy': (0.25, 1.0),         # Shorter, bouncy notes
            'melancholy': (1.0, 4.0),   # Longer, lingering notes
            'tension': (0.25, 1.0),     # Shorter, tense notes
            'power': (0.5, 2.0),        # Moderate
            'mysteriousness': (0.5, 3.0), # Varied
            'harshness': (0.125, 0.5)   # Very short, staccato
        }
        
        return duration_ranges.get(emotion, self.duration_range)
    
    def _determine_velocity_range(self, emotion: str) -> Tuple[int, int]:
        """
        Determine the range of note velocities (volumes) based on emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Tuple of (min_velocity, max_velocity)
        """
        # Define velocity ranges for emotions
        velocity_ranges = {
            'serenity': (40, 70),     # Softer
            'awe': (50, 90),          # Moderate to loud
            'joy': (60, 90),          # Moderate to loud
            'melancholy': (30, 60),   # Softer
            'tension': (40, 80),      # Varied
            'power': (70, 100),       # Louder
            'mysteriousness': (30, 70), # Varied
            'harshness': (70, 100)    # Louder
        }
        
        return velocity_ranges.get(emotion, self.dynamic_range)
    
    def _determine_register(
        self, 
        emotion: str, 
        instrument: str
    ) -> Tuple[int, int]:
        """
        Determine the pitch register based on emotion and instrument.
        
        Args:
            emotion: Emotion name
            instrument: Instrument name
            
        Returns:
            Tuple of (min_octave, max_octave) relative to middle C
        """
        # Define base registers for emotions
        emotion_registers = {
            'serenity': (0, 1),      # Middle to high
            'awe': (-1, 1),          # Wide range
            'joy': (0, 2),           # Middle to high
            'melancholy': (-1, 0),   # Low to middle
            'tension': (-1, 1),      # Wide range
            'power': (-1, 1),        # Wide range
            'mysteriousness': (-2, 0), # Low to middle
            'harshness': (-1, 2)     # Very wide range
        }
        
        # Define instrument-specific register adjustments
        instrument_adjustments = {
            'piano': (0, 0),       # No adjustment
            'marimba': (0, 1),     # Middle to high
            'organ': (-1, 0),      # Low to middle
            'guitar': (-1, 0),     # Low to middle
            'bass': (-2, -1),      # Very low
            'strings': (0, 0),     # No adjustment
            'choir': (0, 0),       # No adjustment
            'trumpet': (0, 1),     # Middle to high
            'flute': (1, 2),       # High
            'pad': (-1, 1),        # Wide range
            'percussion': (0, 0)   # No adjustment
        }
        
        # Get base register for this emotion
        base_register = emotion_registers.get(emotion, (0, 0))
        
        # Get adjustment for this instrument
        adjustment = instrument_adjustments.get(instrument, (0, 0))
        
        # Combine
        return (
            base_register[0] + adjustment[0],
            base_register[1] + adjustment[1]
        )
    
    def _select_note_from_scale(
        self, 
        scale: List[int], 
        register: Tuple[int, int]
    ) -> int:
        """
        Select a note from the scale within the given register.
        
        Args:
            scale: List of MIDI note numbers in the scale
            register: Tuple of (min_octave, max_octave) relative to middle C
            
        Returns:
            Selected MIDI note number
        """
        # Filter notes to the desired register
        min_note = 60 + (register[0] * 12)  # 60 is middle C
        max_note = 60 + (register[1] * 12) + 11
        
        # Ensure min and max are within valid MIDI note range (0-127)
        min_note = max(0, min(127, min_note))
        max_note = max(0, min(127, max_note))
        
        valid_notes = [note for note in scale if min_note <= note <= max_note]
        
        if not valid_notes:
            # Fallback if no notes in register
            return 60  # Default to middle C
        
        return random.choice(valid_notes)
    
    def _predict_next_note(
        self, 
        previous_notes: List[Dict[str, Any]], 
        scale: List[int],
        register: Tuple[int, int]
    ) -> int:
        """
        Predict the next note based on previous notes using a model.
        
        Args:
            previous_notes: List of previous notes
            scale: List of MIDI note numbers in the scale
            register: Tuple of (min_octave, max_octave) relative to middle C
            
        Returns:
            Predicted MIDI note number
        """
        # If no previous notes, select random note from scale
        if not previous_notes:
            return self._select_note_from_scale(scale, register)
        
        # Get the last few notes
        last_notes = previous_notes[-3:]
        last_pitches = [note['pitch'] for note in last_notes]
        
        # Simple melodic pattern detection
        if len(last_pitches) >= 2:
            # Check if there's a consistent interval pattern
            intervals = [last_pitches[i+1] - last_pitches[i] 
                        for i in range(len(last_pitches)-1)]
            
            if len(intervals) >= 2 and intervals[-1] == intervals[-2]:
                # Continue the pattern
                next_pitch = last_pitches[-1] + intervals[-1]
                
                # Check if the note is in the scale
                if next_pitch in scale:
                    return next_pitch
        
        # If no pattern detected or note not in scale, use markov-like approach
        # Based on the last note, prefer common intervals
        last_pitch = last_pitches[-1]
        
        # Common melodic intervals (in semitones)
        common_intervals = [0, 2, -2, 3, -3, 4, -4, 5, -5, 7, -7]
        weights = [2, 3, 3, 2, 2, 2, 2, 1, 1, 3, 3]  # Unison, 2nds, 3rds, 5ths more common
        
        # Generate candidate notes
        candidates = []
        
        for interval, weight in zip(common_intervals, weights):
            candidate = last_pitch + interval
            
            # Check if the note is in the scale
            if candidate in scale:
                # Check if note is in the register
                min_note = 60 + (register[0] * 12)
                max_note = 60 + (register[1] * 12) + 11
                
                if min_note <= candidate <= max_note:
                    candidates.extend([candidate] * weight)
        
        if candidates:
            return random.choice(candidates)
        
        # Fallback to random note from scale
        return self._select_note_from_scale(scale, register)
    
    def _get_emotion_color(self, emotion: str) -> str:
        """
        Get a color associated with an emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Hex color code
        """
        emotion_colors = {
            'serenity': '#ADD8E6',    # Light blue
            'awe': '#800080',         # Purple
            'joy': '#FFD700',         # Gold
            'melancholy': '#483D8B',  # Dark slate blue
            'tension': '#FF4500',     # Orange red
            'power': '#8B0000',       # Dark red
            'mysteriousness': '#2F4F4F', # Dark slate gray
            'harshness': '#A52A2A'    # Brown
        }
        
        return emotion_colors.get(emotion, '#000000')  # Default to black