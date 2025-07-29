# Music Transition Transformer

A transformer-based model for creating seamless transitions between music segments using mel-spectrogram representation.

## Project Structure

```
music_transformer/
├── __init__.py          # Package initialization
├── config.py           # Configuration settings
├── model.py            # Core transformer model
├── audio_processor.py  # Audio processing utilities
├── dataset.py          # Dataset creation and loading
└── train.py            # Training utilities and trainer class

test_music_transformer.py  # Main test script
example.py                 # Simple usage example
requirements.txt           # Python dependencies
```

## Features

- **Dual Encoder Architecture**: Separate encoders for preceding and following music segments
- **Mel-Spectrogram Processing**: Works with frequency-domain audio representation
- **Continuous Representation**: Generates smooth spectrograms instead of discrete tokens
- **Autoregressive Generation**: Creates transitions step-by-step with proper temporal coherence
- **Synthetic Data Support**: Includes synthetic data generation for testing

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Test Suite

```bash
python test_music_transformer.py
```

This will:
- Test model initialization
- Create synthetic spectrogram data
- Test forward pass with teacher forcing
- Test autoregressive generation
- Test training loop functionality

### 3. Run Simple Example

```bash
python example.py
```

This demonstrates basic usage of the model for generating transitions.

## Model Architecture

### Input/Output
- **Input**: Mel-spectrograms of shape `[batch, time, n_mels]` where `n_mels=128`
- **Context Length**: 128 time steps for preceding and following segments
- **Transition Length**: 64 time steps for generated transitions
- **Output**: Generated mel-spectrogram transition

### Architecture Details
- **Model Dimension**: 512
- **Attention Heads**: 8
- **Layers**: 6 encoder + 6 decoder layers
- **Feed-forward Dimension**: 2048
- **Parameters**: ~10M parameters

### Forward Pass
1. **Input Projection**: Convert mel-spectrograms to model dimension
2. **Positional Encoding**: Add temporal position information
3. **Dual Encoding**: Process preceding and following contexts separately
4. **Cross-Attention Decoding**: Generate transition using both contexts
5. **Output Projection**: Convert back to mel-spectrogram space

## Usage Example

```python
import torch
from music_transformer import (
    Config, 
    MusicTransitionTransformer,
    create_synthetic_spectrogram_loaders
)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()

# Create model
model = MusicTransitionTransformer(config).to(device)

# Create synthetic data
train_loader, _, _, _ = create_synthetic_spectrogram_loaders(config)
batch = next(iter(train_loader))

# Generate transition
preceding = batch['preceding'].to(device)
following = batch['following'].to(device)

model.eval()
with torch.no_grad():
    transition = model(preceding, following, teacher_forcing=False)
    print(f"Generated transition shape: {transition.shape}")
```

## Training

```python
from music_transformer import MusicTransitionTrainer

# Create trainer
trainer = MusicTransitionTrainer(model, config, device)

# Train model
train_history, val_history = trainer.train(
    train_loader, val_loader, num_epochs=50
)
```

## Configuration

Key configuration parameters in `config.py`:

```python
# Audio Processing
sample_rate = 44100     # Audio sample rate
n_mels = 128           # Number of mel frequency bins
context_length = 128   # Length of input contexts
transition_length = 64 # Length of generated transition

# Model Architecture  
d_model = 512          # Model dimension
n_heads = 8            # Number of attention heads
n_layers = 6           # Number of transformer layers
batch_size = 8         # Training batch size
```

## Loss Function

The model uses a combined loss function:
- **Reconstruction Loss**: MSE between generated and target spectrograms
- **BCE Loss**: Binary cross-entropy for spectrogram presence
- **Onset Loss**: Emphasizes temporal changes and note onsets
- **Continuity Loss**: Encourages smooth transitions

## Known Issues Fixed

✅ **Matrix Multiplication Error**: Fixed tensor dimension mismatch by adding automatic reshaping for flattened inputs
✅ **Attention Mask Broadcasting**: Corrected mask dimensions for multi-head attention
✅ **Synthetic Data Generation**: Proper spectrogram synthesis with realistic patterns

## Next Steps

1. **Real Audio Data**: Replace synthetic data with actual audio files from DJNet dataset
2. **Audio Reconstruction**: Implement Griffin-Lim or neural vocoder for spectrogram-to-audio
3. **Evaluation Metrics**: Add perceptual loss and audio quality metrics
4. **Model Optimization**: Experiment with different architectures and hyperparameters

## Troubleshooting

### Import Errors
Make sure you're running from the project root directory and have installed all dependencies.

### CUDA Issues
If you get CUDA errors, the model will automatically fall back to CPU processing.

### Memory Issues
Reduce `batch_size` in the configuration if you encounter out-of-memory errors.

## License

This project is for educational purposes as part of a Deep Learning course.
