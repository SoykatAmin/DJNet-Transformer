"""
Audio processing utilities for converting between audio and spectrogram representation
"""

import numpy as np
import librosa
import soundfile as sf


class AudioProcessor:
    """Handles conversion between audio and spectrogram representation for DJNet dataset"""
    
    def __init__(self, config):
        self.config = config
        self.sr = config.sample_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        # Use mel_bins if available, otherwise fall back to n_mels
        self.n_mels = getattr(config, 'mel_bins', config.n_mels)
        
    def audio_to_spectrogram(self, audio_path, duration=None):
        """Convert audio file to mel-spectrogram"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)
            
            # Convert to mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1] range
            mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            return mel_spec_normalized.T  # Shape: (time_frames, n_mels)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def audio_to_mel_spectrogram(self, audio_array):
        """Convert audio array to mel-spectrogram (expected by notebook)"""
        try:
            # Convert to mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_array, 
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1] range
            mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            return mel_spec_normalized.T  # Shape: (time_frames, n_mels)
            
        except Exception as e:
            print(f"Error processing audio array: {e}")
            return None
    
    def process_audio_array(self, audio_array):
        """Process audio array (numpy) to mel-spectrogram"""
        try:
            # Convert to mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_array, 
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1] range
            mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            return mel_spec_normalized.T  # Shape: (time_frames, n_mels)
            
        except Exception as e:
            print(f"Error processing audio array: {e}")
            return None
    
    def spectrogram_to_audio(self, mel_spec):
        """Convert mel-spectrogram back to audio (approximate reconstruction)"""
        try:
            # Transpose back to (n_mels, time_frames)
            if mel_spec.shape[0] != self.n_mels:
                mel_spec = mel_spec.T
            
            # Denormalize spectrogram (approximate)
            mel_spec_db = mel_spec * 80.0 - 80.0  # Approximate dB range
            
            # Convert from dB to power
            mel_spec_power = librosa.db_to_power(mel_spec_db)
            
            # Inverse mel-spectrogram (Griffin-Lim algorithm)
            audio = librosa.feature.inverse.mel_to_audio(
                mel_spec_power,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            return audio
            
        except Exception as e:
            print(f"Error converting spectrogram to audio: {e}")
            return None
    
    def get_spectrogram_shape(self, duration_seconds):
        """Calculate expected spectrogram shape for given duration"""
        n_samples = int(duration_seconds * self.sr)
        n_frames = 1 + (n_samples - self.n_fft) // self.hop_length
        return (n_frames, self.n_mels)
