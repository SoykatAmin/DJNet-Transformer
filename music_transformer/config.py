"""
Configuration settings for the Music Transition Transformer
"""

class Config:
    """Configuration class for the Music Transition Transformer"""
    
    # Audio Processing Parameters
    sample_rate = 44100  # Audio sample rate (matches DJNet dataset)
    n_fft = 2048        # FFT window size
    hop_length = 512    # Hop length for STFT
    n_mels = 128        # Number of mel frequency bins
    mel_bins = 128      # Alias for n_mels (for compatibility)
    seq_len = 128       # Sequence length for each segment
    max_sequence_length = 512  # Maximum length for input sequences
    
    # Model Architecture
    d_model = 512  # Embedding dimension
    n_heads = 8    # Number of attention heads
    num_heads = 8  # Alias for n_heads (for compatibility)
    n_layers = 6   # Number of transformer layers
    num_layers = 6 # Alias for n_layers (for compatibility)
    d_ff = 2048    # Feed-forward dimension
    dropout = 0.1  # Dropout rate
    
    # Training Parameters
    batch_size = 8     # Reduced for larger spectrogram inputs
    learning_rate = 1e-4
    max_epochs = 100
    warmup_steps = 4000
    
    # Transition Parameters
    context_length = 128    # Length of context segments (before/after)
    transition_length = 64  # Length of generated transition
    
    # Audio Segmentation (in seconds)
    segment_duration = 8.0    # Duration of each segment in seconds
    transition_duration = 4.0 # Duration of transition segment
    
    # DJNet Dataset specific
    djnet_dataset_path = None  # Will be set when loading dataset
    
    def __repr__(self):
        return f"Config(d_model={self.d_model}, n_heads={self.n_heads}, n_layers={self.n_layers})"
