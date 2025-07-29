"""
Dataset creation and data loading utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SyntheticSpectrogramDataset(Dataset):
    """Create synthetic spectrogram data for testing"""
    
    def __init__(self, config, num_samples=100):
        self.config = config
        self.num_samples = num_samples
        
        # Generate synthetic data
        self.data = self._generate_synthetic_data()
        
    def _generate_synthetic_data(self):
        """Generate synthetic spectrogram triplets"""
        data = []
        
        for i in range(self.num_samples):
            # Create synthetic spectrograms with different patterns
            
            # Preceding context: Gradual fade-in pattern
            preceding = np.random.rand(self.config.context_length, self.config.n_mels) * 0.3
            for t in range(self.config.context_length):
                fade_factor = t / self.config.context_length
                preceding[t] *= fade_factor
            
            # Following context: Different frequency emphasis
            following = np.random.rand(self.config.context_length, self.config.n_mels) * 0.4
            # Emphasize lower frequencies in following context
            for f in range(self.config.n_mels // 2):
                following[:, f] *= 1.5
            
            # Transition: Blend between preceding and following
            transition = np.random.rand(self.config.transition_length, self.config.n_mels) * 0.2
            # Add some harmonic structure
            for t in range(self.config.transition_length):
                blend_factor = t / self.config.transition_length
                # Simple interpolation between patterns
                base_pattern = (1 - blend_factor) * 0.3 + blend_factor * 0.4
                transition[t] += base_pattern * np.random.rand(self.config.n_mels)
            
            # Add some coherent structure
            if i % 3 == 0:  # Low frequency emphasis
                preceding[:, :32] *= 1.3
                following[:, :32] *= 1.3
                transition[:, :32] *= 1.3
            elif i % 3 == 1:  # Mid frequency emphasis
                preceding[:, 32:96] *= 1.2
                following[:, 32:96] *= 1.2
                transition[:, 32:96] *= 1.2
            else:  # High frequency emphasis
                preceding[:, 96:] *= 1.1
                following[:, 96:] *= 1.1
                transition[:, 96:] *= 1.1
            
            # Ensure values are in [0, 1] range
            preceding = np.clip(preceding, 0, 1)
            following = np.clip(following, 0, 1)
            transition = np.clip(transition, 0, 1)
            
            data.append({
                'preceding': preceding.astype(np.float32),
                'following': following.astype(np.float32),
                'transition': transition.astype(np.float32),
                'conditioning': {'tempo': 120 + np.random.randint(-20, 20)},
                'transition_dir': f'synthetic_{i}'
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        return {
            'preceding': torch.FloatTensor(item['preceding']),
            'following': torch.FloatTensor(item['following']),
            'transition': torch.FloatTensor(item['transition']),
            'conditioning': item['conditioning'],
            'transition_dir': item['transition_dir']
        }


def collate_fn_synthetic(batch):
    """Custom collate function for batching synthetic data"""
    preceding = torch.stack([item['preceding'] for item in batch])
    following = torch.stack([item['following'] for item in batch])
    transition = torch.stack([item['transition'] for item in batch])
    
    # Keep conditioning info and paths
    conditioning = [item['conditioning'] for item in batch]
    transition_dirs = [item['transition_dir'] for item in batch]
    
    return {
        'preceding': preceding,
        'following': following,
        'transition': transition,
        'conditioning': conditioning,
        'transition_dirs': transition_dirs
    }


def create_synthetic_spectrogram_loaders(config, batch_size=None):
    """Create synthetic data loaders for testing"""
    
    if batch_size is None:
        batch_size = config.batch_size
    
    print("Creating synthetic data loaders for testing...")
    
    # Create training dataset
    print("Creating synthetic spectrogram data for testing...")
    train_dataset = SyntheticSpectrogramDataset(config, num_samples=80)
    print(f"Created synthetic spectrogram dataset with {len(train_dataset)} samples")
    
    # Create test dataset
    print("Creating synthetic spectrogram data for testing...")
    test_dataset = SyntheticSpectrogramDataset(config, num_samples=20)
    print(f"Created synthetic spectrogram dataset with {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_synthetic,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_synthetic,
        drop_last=False
    )
    
    return train_loader, test_loader, train_dataset, test_dataset
