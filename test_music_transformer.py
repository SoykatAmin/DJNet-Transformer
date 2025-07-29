#!/usr/bin/env python3
"""
Main test script for the Music Transition Transformer project
"""

import torch
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from music_transformer import (
    Config,
    MusicTransitionTransformer,
    AudioProcessor,
    create_synthetic_spectrogram_loaders,
    MusicTransitionTrainer
)


def test_model():
    """Test the model with synthetic data"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    config = Config()
    print(f"Model Configuration: {config}")
    
    # Create model
    print("\nInitializing model...")
    model = MusicTransitionTransformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    try:
        train_loader, test_loader, train_dataset, test_dataset = create_synthetic_spectrogram_loaders(config)
        
        print(f"✓ Synthetic training dataset: {len(train_dataset)} segments")
        print(f"✓ Synthetic test dataset: {len(test_dataset)} segments")
        print(f"✓ Training batches: {len(train_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")
        
        # Get and display a sample batch
        sample_batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Preceding context: {sample_batch['preceding'].shape}")
        print(f"  Following context: {sample_batch['following'].shape}")
        print(f"  Target transition: {sample_batch['transition'].shape}")
        
        # Test model forward pass
        print(f"\nTesting model forward pass with synthetic data...")
        model.eval()
        with torch.no_grad():
            # Move data to device
            preceding = sample_batch['preceding'].to(device)
            following = sample_batch['following'].to(device)
            target = sample_batch['transition'].to(device)
            
            print(f"Debug - Input tensor shapes:")
            print(f"  preceding: {preceding.shape}")
            print(f"  following: {following.shape}")
            print(f"  target: {target.shape}")
            print(f"  Expected shape: [batch, time, {config.n_mels}]")
            
            # Test teacher forcing mode
            print("\nTesting teacher forcing mode...")
            output = model(preceding, following, target, teacher_forcing=True)
            print(f"✓ Model output shape: {output.shape}")
            
            # Test generation mode
            print("Testing generation mode...")
            generated = model(preceding, following, teacher_forcing=False)
            print(f"✓ Generated transition shape: {generated.shape}")
        
        print(f"\n✓ All tests passed! The model is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training():
    """Test the training loop"""
    
    print("\n" + "="*50)
    print("TESTING TRAINING LOOP")
    print("="*50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    
    # Create model and trainer
    model = MusicTransitionTransformer(config).to(device)
    trainer = MusicTransitionTrainer(model, config, device)
    
    # Create data
    train_loader, test_loader, _, _ = create_synthetic_spectrogram_loaders(config)
    
    try:
        # Test single training step
        print("Testing single training step...")
        sample_batch = next(iter(train_loader))
        loss_dict = trainer.train_step(sample_batch)
        
        print(f"✓ Training step successful!")
        print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
        print(f"  Reconstruction loss: {loss_dict['reconstruction_loss'].item():.4f}")
        print(f"  BCE loss: {loss_dict['bce_loss'].item():.4f}")
        print(f"  Onset loss: {loss_dict['onset_loss'].item():.4f}")
        print(f"  Continuity loss: {loss_dict['continuity_loss'].item():.4f}")
        
        # Test validation step
        print("\nTesting validation step...")
        val_loss_dict = trainer.validate_step(sample_batch)
        print(f"✓ Validation step successful!")
        print(f"  Validation loss: {val_loss_dict['total_loss'].item():.4f}")
        
        # Test full epoch (just 2 batches for speed)
        print("\nTesting training epoch (2 epochs)...")
        train_history, val_history = trainer.train(
            train_loader, test_loader, num_epochs=2
        )
        
        print(f"✓ Training epochs completed!")
        print(f"  Final training loss: {train_history[-1]['total_loss']:.4f}")
        print(f"  Final validation loss: {val_history[-1]['total_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during training test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    
    print("="*60)
    print("MUSIC TRANSITION TRANSFORMER - TEST SUITE")
    print("="*60)
    
    # Test model functionality
    model_success = test_model()
    
    if model_success:
        # Test training functionality
        training_success = test_training()
        
        if training_success:
            print("\n" + "="*60)
            print("🎉 ALL TESTS PASSED! 🎉")
            print("The Music Transition Transformer is working correctly!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ TRAINING TESTS FAILED")
            print("The model works but training has issues.")
            print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ MODEL TESTS FAILED")
        print("There are issues with the basic model functionality.")
        print("="*60)


if __name__ == "__main__":
    main()
