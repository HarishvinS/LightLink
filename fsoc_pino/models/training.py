"""
Training utilities for PINO models.

This module provides training loops, optimization, and monitoring
for Physics-Informed Neural Operators.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import numpy as np
from pathlib import Path
import time
import json
from tqdm import tqdm

from .pino import PINO_FNO
from .losses import PINOLoss


class PINOTrainer:
    """
    Trainer for PINO models with physics-informed constraints.
    """
    
    def __init__(
        self,
        model: PINO_FNO,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        physics_loss_weight: float = 0.1,
        output_dir: Optional[Path] = None,
        wandb_project: Optional[str] = None
    ):
        """
        Initialize PINO trainer.
        
        Args:
            model: PINO model to train
            device: Device for training
            learning_rate: Learning rate
            physics_loss_weight: Weight for physics loss term
            output_dir: Directory for saving outputs
            wandb_project: Weights & Biases project name
        """
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = PINOLoss(
            data_weight=1.0,
            physics_weight=physics_loss_weight,
            grid_spacing=model.grid_spacing,
            wavelength=model.wavelength
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_data_loss': [],
            'train_physics_loss': [],
            'val_data_loss': [],
            'val_physics_loss': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Weights & Biases logging
        self.use_wandb = False
        if wandb_project:
            try:
                import wandb
                wandb.init(project=wandb_project)
                self.use_wandb = True
            except ImportError:
                print("Warning: wandb not installed, skipping experiment tracking")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'data_loss': 0.0,
            'physics_loss': 0.0
        }
        
        num_batches = len(train_loader)
        
        for batch_idx, (params, targets) in enumerate(tqdm(train_loader, desc="Training")):
            params = params.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions = self.model(params)
            
            # Compute loss
            loss_dict = self.criterion(predictions, targets, params)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
            'data_loss': 0.0,
            'physics_loss': 0.0
        }
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for params, targets in tqdm(val_loader, desc="Validation"):
                params = params.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(params)
                
                # Compute loss
                loss_dict = self.criterion(predictions, targets, params)
                
                # Accumulate losses
                for key in epoch_losses:
                    epoch_losses[key] += loss_dict[key].item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_every: int = 10,
        early_stopping_patience: int = 20
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Model parameters: {self.model.fno.count_parameters():,}")
        print(f"Device: {self.device}")
        
        # Setup parameter normalization
        self._setup_parameter_normalization(train_loader)
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_losses['total_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_losses['total_loss'])
            self.history['val_loss'].append(val_losses['total_loss'])
            self.history['train_data_loss'].append(train_losses['data_loss'])
            self.history['train_physics_loss'].append(train_losses['physics_loss'])
            self.history['val_data_loss'].append(val_losses['data_loss'])
            self.history['val_physics_loss'].append(val_losses['physics_loss'])
            self.history['learning_rate'].append(current_lr)
            
            # Check for best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_losses['total_loss']:.6f} "
                  f"(Data: {train_losses['data_loss']:.6f}, Physics: {train_losses['physics_loss']:.6f})")
            print(f"  Val Loss: {val_losses['total_loss']:.6f} "
                  f"(Data: {val_losses['data_loss']:.6f}, Physics: {val_losses['physics_loss']:.6f})")
            print(f"  LR: {current_lr:.2e}, Best Val: {self.best_val_loss:.6f} (Epoch {self.best_epoch+1})")
            
            # Weights & Biases logging
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_losses['total_loss'],
                    'val_loss': val_losses['total_loss'],
                    'train_data_loss': train_losses['data_loss'],
                    'train_physics_loss': train_losses['physics_loss'],
                    'val_data_loss': val_losses['data_loss'],
                    'val_physics_loss': val_losses['physics_loss'],
                    'learning_rate': current_lr
                })
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model and history
        self.save_checkpoint('final_model.pth')
        self.save_history()
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_time': total_time,
            'history': self.history
        }
    
    def _setup_parameter_normalization(self, train_loader: DataLoader):
        """Setup parameter normalization from training data."""
        print("Computing parameter normalization statistics...")
        
        all_params = []
        for params, _ in train_loader:
            all_params.append(params)
        
        all_params = torch.cat(all_params, dim=0)
        param_mean = torch.mean(all_params, dim=0)
        param_std = torch.std(all_params, dim=0)
        
        self.model.set_parameter_normalization(param_mean, param_std)
        print(f"Parameter normalization set: mean={param_mean.numpy()}, std={param_std.numpy()}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        filepath = self.output_dir / filename
        self.model.save(filepath)
    
    def save_history(self):
        """Save training history."""
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint."""
        self.model = PINO_FNO.load(filepath, device=self.device)
    
    def export_onnx(self, filepath: Path, use_dynamo: bool = True, fallback_to_torchscript: bool = True):
        """
        Export model to ONNX format with fallback options.

        Args:
            filepath: Path to save the model
            use_dynamo: Whether to use the new dynamo-based exporter
            fallback_to_torchscript: Whether to fallback to TorchScript if ONNX fails

        Returns:
            str: The actual export format used ('onnx' or 'torchscript')
        """
        return self.model.export_onnx(filepath, use_dynamo=use_dynamo,
                                     fallback_to_torchscript=fallback_to_torchscript)

    def export_torchscript(self, filepath: Path):
        """Export model to TorchScript format."""
        return self.model.export_torchscript(filepath)
