#!/usr/bin/env python3
"""
Convert PyTorch PINO model to ONNX format for edge deployment.

This script loads a trained PyTorch model and exports it to ONNX format
for deployment on edge devices or inference engines.
"""

import torch
from pathlib import Path
import argparse
from fsoc_pino.models import PINO_FNO


def convert_model_to_onnx(
    input_model_path: str,
    output_model_path: str,
    use_dynamo: bool = True,
    fallback_to_torchscript: bool = True
):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        input_model_path: Path to input PyTorch model (.pth file)
        output_model_path: Path to output ONNX model (.onnx file)
        use_dynamo: Whether to use dynamo-based ONNX export
        fallback_to_torchscript: Whether to fallback to TorchScript if ONNX fails
    """
    print(f"Loading model from: {input_model_path}")
    
    # Load the PyTorch model
    model = PINO_FNO.load(Path(input_model_path), device='cpu')
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model info: {model.get_model_info()}")
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_model_path}")
    
    try:
        export_format = model.export_onnx(
            Path(output_model_path),
            use_dynamo=use_dynamo,
            fallback_to_torchscript=fallback_to_torchscript
        )
        
        print(f"✅ Model successfully exported as {export_format}")
        
        # Verify the exported model
        if export_format == 'onnx':
            verify_onnx_model(output_model_path)
        elif export_format == 'torchscript':
            verify_torchscript_model(output_model_path)
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise


def verify_onnx_model(model_path: str):
    """Verify the exported ONNX model."""
    try:
        import onnxruntime as ort
        
        print("Verifying ONNX model...")
        session = ort.InferenceSession(model_path)
        
        # Get input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
        print(f"Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
        
        # Test inference with dummy data
        import numpy as np
        dummy_input = np.random.randn(1, 5).astype(np.float32)
        output = session.run(None, {input_info.name: dummy_input})
        
        print(f"Test inference successful! Output shape: {output[0].shape}")
        
    except ImportError:
        print("⚠️  onnxruntime not available for verification")
    except Exception as e:
        print(f"⚠️  ONNX verification failed: {e}")


def verify_torchscript_model(model_path: str):
    """Verify the exported TorchScript model."""
    try:
        print("Verifying TorchScript model...")
        model = torch.jit.load(model_path, map_location='cpu')
        
        # Test inference with dummy data
        dummy_input = torch.randn(1, 5)
        output = model(dummy_input)
        
        print(f"Test inference successful! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"⚠️  TorchScript verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch PINO model to ONNX")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="./models/model.pt",
        help="Input PyTorch model path (.pth file)"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str,
        default="./models/best_model.onnx",
        help="Output ONNX model path (.onnx file)"
    )
    parser.add_argument(
        "--no-dynamo",
        action="store_true",
        help="Disable dynamo-based ONNX export"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true", 
        help="Disable TorchScript fallback"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert model
    convert_model_to_onnx(
        args.input,
        args.output,
        use_dynamo=not args.no_dynamo,
        fallback_to_torchscript=not args.no_fallback
    )


if __name__ == "__main__":
    main()
