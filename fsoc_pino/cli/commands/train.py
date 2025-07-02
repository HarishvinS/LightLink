"""
Training command for FSOC-PINO CLI.

This module implements the 'train' subcommand that trains PINO models
on generated datasets.
"""

import click
import time
from pathlib import Path
from typing import Optional


@click.command()
@click.option(
    "--dataset-path", "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to training dataset directory"
)
@click.option(
    "--output-dir", "-o", 
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for trained models"
)
@click.option(
    "--config-file", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Training configuration file (YAML)"
)
@click.option(
    "--epochs",
    type=int,
    default=100,
    help="Number of training epochs"
)
@click.option(
    "--batch-size",
    type=int, 
    default=32,
    help="Training batch size"
)
@click.option(
    "--learning-rate", "--lr",
    type=float,
    default=1e-3,
    help="Learning rate"
)
@click.option(
    "--physics-loss-weight",
    type=float,
    default=0.1,
    help="Weight for physics-informed loss term"
)
@click.option(
    "--validation-split",
    type=float,
    default=0.2,
    help="Fraction of data to use for validation"
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    help="Device to use for training"
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    help="Number of data loader workers"
)
@click.option(
    "--save-every",
    type=int,
    default=10,
    help="Save checkpoint every N epochs"
)
@click.option(
    "--early-stopping-patience",
    type=int,
    default=20,
    help="Early stopping patience (epochs)"
)
@click.option(
    "--wandb-project",
    type=str,
    help="Weights & Biases project name for experiment tracking"
)
@click.option(
    "--resume-from",
    type=click.Path(exists=True, path_type=Path),
    help="Resume training from checkpoint"
)
@click.option(
    "--export-format",
    type=click.Choice(["onnx", "torchscript", "both"]),
    default="both",
    help="Model export format for deployment"
)
@click.option(
    "--no-onnx-dynamo",
    is_flag=True,
    help="Disable dynamo-based ONNX export (use traditional method only)"
)
@click.pass_context
def train(
    ctx,
    dataset_path: Path,
    output_dir: Path,
    config_file: Optional[Path],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    physics_loss_weight: float,
    validation_split: float,
    device: str,
    num_workers: int,
    save_every: int,
    early_stopping_patience: int,
    wandb_project: Optional[str],
    resume_from: Optional[Path],
    export_format: str,
    no_onnx_dynamo: bool
):
    """
    Train PINO model on generated dataset.
    
    This command trains a Physics-Informed Neural Operator (PINO) model
    using the Fourier Neural Operator architecture with physics constraints.
    """
    logger = ctx.obj['logger']
    logger.info("Starting PINO model training...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log parameters
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Physics loss weight: {physics_loss_weight}")
    logger.info(f"Validation split: {validation_split}")
    logger.info(f"Device: {device}")
    
    try:
        # Import required modules
        from fsoc_pino.models import PINOTrainer, create_pino_model
        from fsoc_pino.data import HDF5Manager
        import torch

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load dataset
        logger.info("Loading dataset...")
        data_manager = HDF5Manager(dataset_path)
        dataset_info = data_manager.get_dataset_info()
        logger.info(f"Dataset: {dataset_info['total_samples']} samples, grid size: {dataset_info['grid_size']}")

        train_loader, val_loader = data_manager.create_dataloaders(
            batch_size=batch_size,
            validation_split=validation_split,
            num_workers=num_workers
        )

        # Create model
        logger.info("Creating PINO model...")
        model = create_pino_model(
            input_dim=dataset_info['num_parameters'],
            grid_size=dataset_info['grid_size'],
            modes=12,
            width=64,
            num_layers=4,
            device=device
        )

        logger.info(f"Model created with {model.fno.count_parameters():,} parameters")

        # Create trainer
        trainer = PINOTrainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            physics_loss_weight=physics_loss_weight,
            output_dir=output_dir,
            wandb_project=wandb_project
        )

        # Resume from checkpoint if specified
        if resume_from:
            logger.info(f"Resuming training from: {resume_from}")
            trainer.load_checkpoint(resume_from)

        # Train model
        logger.info("Starting training...")
        start_time = time.time()

        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            save_every=save_every,
            early_stopping_patience=early_stopping_patience
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")

        # Export model for deployment
        export_success = False
        exported_formats = []

        if export_format in ["onnx", "both"]:
            logger.info("Exporting model to ONNX format...")
            onnx_model_path = output_dir / "model.onnx"
            try:
                actual_format = trainer.export_onnx(
                    onnx_model_path,
                    use_dynamo=not no_onnx_dynamo,
                    fallback_to_torchscript=(export_format == "onnx")
                )
                exported_formats.append(actual_format)
                if actual_format == 'onnx':
                    logger.info(f"✅ ONNX model saved to: {onnx_model_path}")
                else:
                    logger.info(f"✅ Model exported as {actual_format} (ONNX fallback)")
                export_success = True
            except Exception as e:
                logger.error(f"❌ ONNX export failed: {e}")
                if export_format == "onnx":
                    logger.error("ONNX export was required but failed. Training completed but model export failed.")

        if export_format in ["torchscript", "both"] or (export_format == "onnx" and not export_success):
            logger.info("Exporting model to TorchScript format...")
            torchscript_model_path = output_dir / "model.pt"
            try:
                actual_format = trainer.export_torchscript(torchscript_model_path)
                exported_formats.append(actual_format)
                logger.info(f"✅ TorchScript model saved to: {torchscript_model_path}")
                export_success = True
            except Exception as e:
                logger.error(f"❌ TorchScript export failed: {e}")

        logger.info(f"Training completed successfully!")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Best validation loss: {training_history['best_val_loss']:.6f}")
        logger.info(f"Best epoch: {training_history['best_epoch'] + 1}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")

        if export_success:
            logger.info(f"Model exported in format(s): {', '.join(exported_formats)}")
        else:
            logger.warning("⚠️  Model training completed but export failed. PyTorch model is still available.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        ctx.exit(1)
