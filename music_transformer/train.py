"""
Training utilities: loss functions, schedulers, and trainer class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os


class MusicLoss(nn.Module):
    """
    Custom loss function combining multiple objectives for music generation
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Primary reconstruction loss
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Weights for different loss components
        self.reconstruction_weight = 1.0
        self.onset_weight = 2.0  # Emphasize note onsets
        self.continuity_weight = 0.5  # Smooth transitions
        
    def forward(self, predictions, targets):
        """
        Calculate combined loss
        
        Args:
            predictions: Model predictions [batch, time, mels]
            targets: Target spectrograms [batch, time, mels]
        """
        batch_size, seq_len, n_mels = predictions.shape
        
        # 1. Primary reconstruction loss (MSE)
        reconstruction_loss = self.mse_loss(predictions, targets)
        
        # 2. Binary cross-entropy for spectrogram presence
        bce_loss = self.bce_loss(predictions, targets)
        
        # 3. Onset detection loss (emphasize temporal changes)
        onset_loss = self.calculate_onset_loss(predictions, targets)
        
        # 4. Continuity loss (smooth transitions)
        continuity_loss = self.calculate_continuity_loss(predictions)
        
        # Combine losses
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.reconstruction_weight * bce_loss +
            self.onset_weight * onset_loss +
            self.continuity_weight * continuity_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'bce_loss': bce_loss,
            'onset_loss': onset_loss,
            'continuity_loss': continuity_loss
        }
    
    def calculate_onset_loss(self, predictions, targets):
        """Calculate loss that emphasizes temporal changes"""
        # Detect changes by looking at differences between consecutive time steps
        pred_changes = torch.abs(predictions[:, 1:] - predictions[:, :-1])
        target_changes = torch.abs(targets[:, 1:] - targets[:, :-1])
        
        return self.mse_loss(pred_changes, target_changes)
    
    def calculate_continuity_loss(self, predictions):
        """Calculate loss that encourages smooth transitions"""
        # Penalize large changes between consecutive time steps
        temporal_diff = torch.abs(predictions[:, 1:] - predictions[:, :-1])
        return torch.mean(temporal_diff)


class WarmupScheduler:
    """
    Learning rate scheduler with warmup (following Transformer paper)
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        lr = self.calculate_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def calculate_lr(self):
        """Calculate learning rate using warmup formula"""
        step = max(self.current_step, 1)
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )


class MusicTransitionTrainer:
    """
    Trainer class for the Music Transition Transformer
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Loss and optimizer
        self.criterion = MusicLoss(config)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler
        self.scheduler = WarmupScheduler(
            self.optimizer,
            config.d_model,
            config.warmup_steps
        )
        
        # Training history
        self.train_history = []
        self.val_history = []
        
    def train_step_batch(self, batch):
        """Single training step with batch dictionary"""
        self.model.train()
        
        # Move data to device
        preceding = batch['preceding'].to(self.device)
        following = batch['following'].to(self.device)
        target = batch['transition'].to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(preceding, following, target, teacher_forcing=True)
        
        # Calculate loss
        loss_dict = self.criterion(output, target)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        return loss_dict
    
    def validate_step(self, batch):
        """Single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            preceding = batch['preceding'].to(self.device)
            following = batch['following'].to(self.device)
            target = batch['transition'].to(self.device)
            
            # Forward pass
            output = self.model(preceding, following, target, teacher_forcing=True)
            
            # Calculate loss
            loss_dict = self.criterion(output, target)
            
        return loss_dict
    
    def train_epoch(self, train_loader, val_loader=None):
        """Train for one epoch"""
        train_losses = []
        
        # Training loop
        train_pbar = tqdm(train_loader, desc="Training")
        for batch in train_pbar:
            loss_dict = self.train_step_batch(batch)
            train_losses.append(loss_dict)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Calculate average training losses
        avg_train_loss = {
            key: torch.stack([loss[key] for loss in train_losses]).mean().item()
            for key in train_losses[0].keys()
        }
        
        # Validation loop
        val_losses = []
        if val_loader is not None:
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                loss_dict = self.validate_step(batch)
                val_losses.append(loss_dict)
                
                val_pbar.set_postfix({
                    'val_loss': f"{loss_dict['total_loss'].item():.4f}"
                })
            
            # Calculate average validation losses
            avg_val_loss = {
                key: torch.stack([loss[key] for loss in val_losses]).mean().item()
                for key in val_losses[0].keys()
            }
        else:
            avg_val_loss = None
        
        # Store history
        self.train_history.append(avg_train_loss)
        if avg_val_loss is not None:
            self.val_history.append(avg_val_loss)
        
        return avg_train_loss, avg_val_loss
    
    def train(self, train_loader, val_loader=None, num_epochs=None, save_dir=None):
        """Full training loop"""
        if num_epochs is None:
            num_epochs = self.config.max_epochs
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train and validate
            train_loss, val_loss = self.train_epoch(train_loader, val_loader)
            
            # Print epoch results
            print(f"Train Loss: {train_loss['total_loss']:.4f}")
            if val_loss is not None:
                print(f"Val Loss: {val_loss['total_loss']:.4f}")
                
                # Save best model
                if save_dir and val_loss['total_loss'] < best_val_loss:
                    best_val_loss = val_loss['total_loss']
                    self.save_checkpoint(save_dir, epoch, is_best=True)
            
            # Save regular checkpoint
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir, epoch)
        
        print("Training completed!")
        return self.train_history, self.val_history
    
    def save_checkpoint(self, save_dir, epoch, is_best=False):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        if is_best:
            path = os.path.join(save_dir, 'best_model.pt')
        else:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.current_step = checkpoint['scheduler_step']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch']
    
    def train_step(self, preceding, following, target):
        """
        Simple training step interface for notebook compatibility
        Args:
            preceding: [batch, time, mels] - preceding segment
            following: [batch, time, mels] - following segment  
            target: [batch, time, mels] - target transition
        Returns:
            float: loss value
        """
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with teacher forcing
        output = self.model(preceding, following, target)
        
        # Simple MSE loss for notebook compatibility
        loss = F.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()


# Alias for compatibility with notebook
Trainer = MusicTransitionTrainer
