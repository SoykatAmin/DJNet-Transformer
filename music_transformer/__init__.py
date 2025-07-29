"""
Music Transition Transformer Package
A transformer-based model for creating seamless transitions between music segments.
"""

from .config import Config
from .model import MusicTransitionTransformer
from .audio_processor import AudioProcessor
from .dataset import create_synthetic_spectrogram_loaders
from .train import MusicLoss, WarmupScheduler, MusicTransitionTrainer

__version__ = "1.0.0"
__all__ = [
    'Config',
    'MusicTransitionTransformer', 
    'AudioProcessor',
    'create_synthetic_spectrogram_loaders',
    'MusicLoss',
    'WarmupScheduler',
    'MusicTransitionTrainer'
]
