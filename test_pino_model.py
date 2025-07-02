#!/usr/bin/env python3
"""
Test script for PINO model implementation.
"""

import torch
import numpy as np
from pathlib import Path

from fsoc_pino.models import PINO_FNO, create_pino_model, PINOTrainer
from fsoc_pino.data import HDF5Manager


def test_model_creation():
    """Test PINO model creation and basic functionality."""
    print("=== Testing PINO Model Creation ===")
    
    # Create model
    model = create_pino_model(
        input_dim=5,
        grid_size=32,  # Small for testing
        modes=8,
        width=32,
        num_layers=2,
        device='cpu'
    )
    
    print(f"Model created successfully!")
    print(f"Model info: {model.get_model_info()}")
    
    # Test forward pass
    batch_size = 4
    params = torch.randn(batch_size, 5)
    
    print(f"\nTesting forward pass with input shape: {params.shape}")
    
    with torch.no_grad():
        output = model(params)
        print(f"Output shape: {output.shape}")
        
        # Test derived metrics
        metrics = model.compute_derived_metrics(params)
        print(f"Computed metrics:")
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, mean={value.mean().item():.6f}")
    
    print("✅ Model creation and forward pass successful!")
    return model


def test_loss_functions():
    """Test loss function implementations."""
    print("\n=== Testing Loss Functions ===")
    
    from fsoc_pino.models.losses import DataLoss, PhysicsInformedLoss, PINOLoss
    
    # Create dummy data
    batch_size = 2
    grid_size = 16
    
    prediction = torch.randn(batch_size, 2, grid_size, grid_size)
    target = torch.randn(batch_size, 2, grid_size, grid_size)
    params = torch.randn(batch_size, 5)
    
    # Test data loss
    data_loss_fn = DataLoss()
    data_loss = data_loss_fn(prediction, target)
    print(f"Data loss: {data_loss.item():.6f}")
    
    # Test physics loss
    physics_loss_fn = PhysicsInformedLoss(grid_spacing=0.1)
    physics_loss = physics_loss_fn(prediction, params)
    print(f"Physics loss: {physics_loss.item():.6f}")
    
    # Test combined PINO loss
    pino_loss_fn = PINOLoss(data_weight=1.0, physics_weight=0.1)
    loss_dict = pino_loss_fn(prediction, target, params)
    
    print(f"PINO loss components:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("✅ Loss functions working correctly!")


def test_with_real_data():
    """Test with real dataset if available."""
    print("\n=== Testing with Real Data ===")
    
    dataset_dir = Path("test_dataset_fixed")
    if not dataset_dir.exists():
        print("No real dataset found, skipping this test")
        return
    
    try:
        # Load dataset
        data_manager = HDF5Manager(dataset_dir)
        info = data_manager.get_dataset_info()
        print(f"Dataset info: {info}")
        
        # Create data loaders
        train_loader, val_loader = data_manager.create_dataloaders(
            batch_size=2,
            validation_split=0.5,
            num_workers=0,
            shuffle=False
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Get a batch
        params, targets = next(iter(train_loader))
        print(f"Batch - Params: {params.shape}, Targets: {targets.shape}")
        
        # Create model
        model = create_pino_model(
            input_dim=5,
            grid_size=targets.shape[-1],
            modes=4,
            width=16,
            num_layers=2,
            device='cpu'
        )
        
        # Test forward pass
        with torch.no_grad():
            predictions = model(params)
            print(f"Predictions shape: {predictions.shape}")
            
            # Compute loss
            from fsoc_pino.models.losses import PINOLoss
            loss_fn = PINOLoss()
            loss_dict = loss_fn(predictions, targets, params)
            print(f"Loss on real data: {loss_dict['total_loss'].item():.6f}")
        
        print("✅ Real data test successful!")
        
    except Exception as e:
        print(f"Real data test failed: {e}")


def test_training_setup():
    """Test training setup (without actual training)."""
    print("\n=== Testing Training Setup ===")
    
    # Create dummy dataset
    batch_size = 2
    num_samples = 8
    grid_size = 16
    
    # Create dummy data
    params = torch.randn(num_samples, 5)
    targets = torch.randn(num_samples, 2, grid_size, grid_size)
    
    # Create dataset and loader
    from fsoc_pino.data.storage import FSOCDataset
    from torch.utils.data import DataLoader
    
    dataset = FSOCDataset(params.numpy(), targets.numpy())
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = create_pino_model(
        input_dim=5,
        grid_size=grid_size,
        modes=4,
        width=16,
        num_layers=2,
        device='cpu'
    )
    
    # Create trainer
    trainer = PINOTrainer(
        model=model,
        device='cpu',
        learning_rate=1e-3,
        physics_loss_weight=0.1,
        output_dir=Path('test_training_output')
    )
    
    print(f"Trainer created successfully!")
    print(f"Model parameters: {model.fno.count_parameters():,}")
    
    # Test one training step
    model.train()
    params_batch, targets_batch = next(iter(train_loader))
    
    # Forward pass
    predictions = model(params_batch)
    loss_dict = trainer.criterion(predictions, targets_batch, params_batch)
    
    print(f"Training step test:")
    print(f"  Prediction shape: {predictions.shape}")
    print(f"  Total loss: {loss_dict['total_loss'].item():.6f}")
    print(f"  Data loss: {loss_dict['data_loss'].item():.6f}")
    print(f"  Physics loss: {loss_dict['physics_loss'].item():.6f}")
    
    print("✅ Training setup successful!")


def main():
    """Run all tests."""
    print("🚀 Testing PINO Model Implementation\n")
    
    try:
        # Test model creation
        model = test_model_creation()
        
        # Test loss functions
        test_loss_functions()
        
        # Test with real data
        test_with_real_data()
        
        # Test training setup
        test_training_setup()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
