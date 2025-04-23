"""
Emotion interpretation module for the Emotional Landscape Composer.
"""

import os
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Any, Optional


class EmotionModel:
    """
    Neural network model for interpreting emotional qualities of landscapes.
    """
    
    # Emotional dimensions and their descriptions
    EMOTION_DIMENSIONS = {
        'serenity': 'Calm, peaceful, and tranquil feelings',
        'awe': 'Feelings of wonder, amazement, and being overwhelmed by grandeur',
        'joy': 'Happiness, delight, and pleasure',
        'melancholy': 'Thoughtful sadness or pensive reflection',
        'tension': 'Feelings of stress, anxiety, or suspense',
        'power': 'Strength, might, and dominance',
        'mysteriousness': 'Enigmatic, curious, and intriguing qualities',
        'harshness': 'Rough, severe, and unforgiving aspects'
    }
    
    def __init__(
        self, 
        model_name: str = 'base',
        pretrained_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the emotion interpretation model.
        
        Args:
            model_name: Type of model to use ('base', 'transformer_xl', 'bert')
            pretrained_path: Path to pretrained model weights (if None, uses default)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.model_name = model_name
        self.pretrained_path = pretrained_path
        
        # Set up device
        self.use_gpu = use_gpu and tf.config.list_physical_devices('GPU')
        
        # Initialize the model
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """
        Initialize the neural network model.
        """
        try:
            if self.model_name == 'base':
                self._initialize_base_model()
            elif self.model_name == 'transformer_xl':
                try:
                    self._initialize_transformer_model()
                except Exception as e:
                    print(f"Error initializing transformer model: {e}")
                    print("Falling back to base model...")
                    self._initialize_base_model()
            elif self.model_name == 'bert':
                try:
                    self._initialize_bert_model()
                except Exception as e:
                    print(f"Error initializing BERT model: {e}")
                    print("Falling back to base model...")
                    self._initialize_base_model()
            else:
                print(f"Unknown model type: {self.model_name}")
                print("Falling back to base model...")
                self._initialize_base_model()
        except Exception as e:
            print(f"Error during model initialization: {e}")
            print("Initializing a simple model...")
            self._initialize_simple_model()
            
    def _initialize_base_model(self) -> None:
        """
        Initialize a simple feed-forward neural network model.
        """
        print("Initializing base emotion model...")
        
        # Define input shape based on our feature vector size
        input_dim = 5  # elevation, slope, ruggedness, relief, complexity
        
        # Create a simple feed-forward model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.EMOTION_DIMENSIONS), activation='sigmoid')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Load pretrained weights if available
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            print(f"Loading pretrained weights from {self.pretrained_path}")
            self.model.load_weights(self.pretrained_path)
    
    def _initialize_transformer_model(self) -> None:
        """
        Initialize a Transformer-XL based model for emotion interpretation.
        """
        print("Initializing Transformer-XL emotion model...")
        
        # Define input shape
        input_dim = 5  # elevation, slope, ruggedness, relief, complexity
        sequence_length = 16  # We'll reshape features into sequences
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        
        # Reshape to sequence format - Fixed to handle dimensions correctly
        x = tf.keras.layers.Reshape((1, input_dim))(inputs)
        # Using Lambda layer instead of RepeatVector to avoid dimension issues
        x = tf.keras.layers.Lambda(
            lambda x: tf.tile(x, [1, sequence_length, 1])
        )(x)
        
        # Add positional encoding
        x = self._positional_encoding(x)
        
        # Transformer blocks
        for _ in range(2):
            x = self._transformer_block(x, input_dim, num_heads=4)
        
        # Global pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(len(self.EMOTION_DIMENSIONS), activation='sigmoid')(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Load pretrained weights if available
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            print(f"Loading pretrained weights from {self.pretrained_path}")
            self.model.load_weights(self.pretrained_path)
    
    def _initialize_bert_model(self) -> None:
        """
        Initialize a BERT-based model for emotion interpretation using Hugging Face.
        """
        print("Initializing BERT-based emotion model using Hugging Face Transformers...")
        
        try:
            # Load pretrained BERT model
            self.bert_model = TFAutoModel.from_pretrained('distilbert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Input layers
            input_dim = 5  # elevation, slope, ruggedness, relief, complexity
            terrain_input = tf.keras.layers.Input(shape=(input_dim,), name='terrain_features')
            
            # Convert terrain features to text descriptions
            # This will be handled by the interpret method
            
            # Create custom model
            x = tf.keras.layers.Dense(128, activation='relu')(terrain_input)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(len(self.EMOTION_DIMENSIONS), activation='sigmoid')(x)
            
            # Create model
            self.model = tf.keras.Model(inputs=terrain_input, outputs=outputs)
            
            # Compile the model
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Load pretrained weights if available
            if self.pretrained_path and os.path.exists(self.pretrained_path):
                print(f"Loading pretrained weights from {self.pretrained_path}")
                self.model.load_weights(self.pretrained_path)
                
        except Exception as e:
            print(f"Error initializing BERT model: {e}")
            print("Falling back to base model...")
            self._initialize_base_model()
 
    
    def _positional_encoding(self, x: tf.Tensor) -> tf.Tensor:
        """
        Add positional encoding to the input.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with positional encoding added
        """
        seq_length = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        
        # Create position indices
        positions = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        indices = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        
        # Calculate positional encoding
        angle_rates = 1 / tf.pow(10000, (2 * (indices // 2)) / tf.cast(d_model, tf.float32))
        angles = positions * angle_rates
        
        # Apply sin/cos to even/odd indices
        sines = tf.sin(angles[:, 0::2])
        cosines = tf.cos(angles[:, 1::2])
        
        # Combine and reshape
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        # Add to input
        return x + tf.cast(pos_encoding, x.dtype)
    
    def _transformer_block(
        self, 
        x: tf.Tensor, 
        embed_dim: int, 
        num_heads: int
    ) -> tf.Tensor:
        """
        Create a transformer block.
        
        Args:
            x: Input tensor
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            
        Returns:
            Output tensor from transformer block
        """
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim
        )(x, x)
        
        # Add & normalize
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim * 4, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        
        ffn_output = ffn(x)
        
        # Add & normalize
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    def interpret(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Interpret the emotional qualities of terrain features.
        
        Args:
            features: Dictionary of terrain features
            
        Returns:
            Dictionary of emotional mappings
        """
        # Extract the feature data
        elevation = features['elevation']
        slope = features['slope']
        ruggedness = features['ruggedness']
        relief = features['relief']
        complexity = features['complexity']
        
        # Calculate global statistics
        elevation_mean = np.mean(elevation)
        slope_mean = np.mean(slope)
        ruggedness_mean = np.mean(ruggedness)
        relief_mean = np.mean(relief)
        complexity_mean = np.mean(complexity)
        
        # Create feature vector
        feature_vector = np.array([
            elevation_mean,
            slope_mean,
            ruggedness_mean,
            relief_mean,
            complexity_mean
        ]).reshape(1, -1)
        
        # Use the model to predict emotional qualities
        emotion_values = self.model.predict(feature_vector)[0]
        
        # Create detailed emotional mapping
        emotions = {}
        for i, (name, _) in enumerate(self.EMOTION_DIMENSIONS.items()):
            # Create a 2D grid of emotion values
            emotion_grid = np.zeros_like(elevation)
            
            # Scale by local features
            emotion_grid += elevation * self._get_elevation_weight(name)
            emotion_grid += slope * self._get_slope_weight(name)
            emotion_grid += ruggedness * self._get_ruggedness_weight(name)
            emotion_grid += relief * self._get_relief_weight(name)
            emotion_grid += complexity * self._get_complexity_weight(name)
            
            # Scale to [0, 1]
            emotion_grid = (emotion_grid - np.min(emotion_grid)) / (np.max(emotion_grid) - np.min(emotion_grid) + 1e-8)
            
            # Modulate by the global emotion prediction
            emotion_grid *= emotion_values[i]
            
            emotions[name] = emotion_grid
        
        return emotions
    
    def _get_elevation_weight(self, emotion: str) -> float:
        """
        Get the weight of elevation for a specific emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Weight value
        """
        weights = {
            'serenity': 0.3,    # Moderate positive correlation
            'awe': 0.8,         # Strong positive correlation
            'joy': 0.4,         # Moderate positive correlation
            'melancholy': -0.3, # Moderate negative correlation
            'tension': 0.1,     # Weak positive correlation
            'power': 0.7,       # Strong positive correlation
            'mysteriousness': 0.5, # Moderate positive correlation
            'harshness': 0.2,   # Weak positive correlation
        }
        return weights.get(emotion, 0.0)
    
    def _get_slope_weight(self, emotion: str) -> float:
        """
        Get the weight of slope for a specific emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Weight value
        """
        weights = {
            'serenity': -0.5,   # Moderate negative correlation
            'awe': 0.6,         # Moderate positive correlation
            'joy': -0.2,        # Weak negative correlation
            'melancholy': 0.3,  # Moderate positive correlation
            'tension': 0.8,     # Strong positive correlation
            'power': 0.7,       # Strong positive correlation
            'mysteriousness': 0.4, # Moderate positive correlation
            'harshness': 0.9,   # Strong positive correlation
        }
        return weights.get(emotion, 0.0)
    
    def _get_ruggedness_weight(self, emotion: str) -> float:
        """
        Get the weight of ruggedness for a specific emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Weight value
        """
        weights = {
            'serenity': -0.7,   # Strong negative correlation
            'awe': 0.5,         # Moderate positive correlation
            'joy': -0.3,        # Moderate negative correlation
            'melancholy': 0.2,  # Weak positive correlation
            'tension': 0.7,     # Strong positive correlation
            'power': 0.8,       # Strong positive correlation
            'mysteriousness': 0.6, # Moderate positive correlation
            'harshness': 0.9,   # Strong positive correlation
        }
        return weights.get(emotion, 0.0)
    
    def _get_relief_weight(self, emotion: str) -> float:
        """
        Get the weight of relief for a specific emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Weight value
        """
        weights = {
            'serenity': -0.3,   # Moderate negative correlation
            'awe': 0.9,         # Strong positive correlation
            'joy': 0.1,         # Weak positive correlation
            'melancholy': 0.0,  # No correlation
            'tension': 0.5,     # Moderate positive correlation
            'power': 0.8,       # Strong positive correlation
            'mysteriousness': 0.7, # Strong positive correlation
            'harshness': 0.5,   # Moderate positive correlation
        }
        return weights.get(emotion, 0.0)
    
    def _get_complexity_weight(self, emotion: str) -> float:
        """
        Get the weight of complexity for a specific emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Weight value
        """
        weights = {
            'serenity': -0.6,   # Moderate negative correlation
            'awe': 0.7,         # Strong positive correlation
            'joy': 0.0,         # No correlation
            'melancholy': 0.4,  # Moderate positive correlation
            'tension': 0.6,     # Moderate positive correlation
            'power': 0.5,       # Moderate positive correlation
            'mysteriousness': 0.9, # Strong positive correlation
            'harshness': 0.6,   # Moderate positive correlation
        }
        return weights.get(emotion, 0.0)
    
    def visualize_emotions(self, emotions: Dict[str, np.ndarray]) -> None:
        """
        Visualize the emotional mappings.
        
        Args:
            emotions: Dictionary of emotional mappings
        """
        # Create a figure with subplots for each emotion
        n_emotions = len(emotions)
        n_cols = min(4, n_emotions)
        n_rows = (n_emotions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        
        # Flatten axes if necessary
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
            
        for i, (emotion_name, emotion_data) in enumerate(emotions.items()):
            if i < len(axes):
                im = axes[i].imshow(emotion_data, cmap='viridis')
                axes[i].set_title(emotion_name)
                plt.colorbar(im, ax=axes[i])
                
        # Hide any unused subplots
        for i in range(n_emotions, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
    
    def get_dominant_emotions(
        self, 
        emotions: Dict[str, np.ndarray], 
        threshold: float = 0.6
    ) -> Dict[str, float]:
        """
        Get the dominant emotions from the emotional mappings.
        
        Args:
            emotions: Dictionary of emotional mappings
            threshold: Threshold for considering an emotion dominant
            
        Returns:
            Dictionary of dominant emotions and their strengths
        """
        # Calculate the mean value for each emotion
        emotion_means = {
            name: float(np.mean(values))
            for name, values in emotions.items()
        }
        
        # Scale values to sum to 1.0
        total = sum(emotion_means.values())
        if total > 0:
            emotion_means = {
                name: value / total
                for name, value in emotion_means.items()
            }
        
        # Filter by threshold
        dominant_emotions = {
            name: value
            for name, value in emotion_means.items()
            if value >= threshold / len(emotions)
        }
        
        # Sort by strength
        dominant_emotions = dict(
            sorted(dominant_emotions.items(), key=lambda x: x[1], reverse=True)
        )
        
        return dominant_emotions