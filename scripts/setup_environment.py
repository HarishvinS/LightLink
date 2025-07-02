#!/usr/bin/env python3
"""
Environment setup script for FSOC-PINO development.

This script helps set up the development environment including:
- Virtual environment creation
- Dependency installation
- Pre-commit hooks setup
- Development tools configuration
"""

import os
import sys
import subprocess
import argparse


def run_command(cmd, check=True):
    """Run a shell command and handle errors."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result


def main():
    parser = argparse.ArgumentParser(description="Setup FSOC-PINO development environment")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--gpu", action="store_true", help="Install GPU-enabled PyTorch")
    args = parser.parse_args()
    
    print("Setting up FSOC-PINO development environment...")
    
    # Install package in development mode
    run_command("pip install -e .")
    
    if args.dev:
        print("Installing development dependencies...")
        run_command("pip install -r requirements-dev.txt")
    
    if args.gpu:
        print("Installing GPU-enabled PyTorch...")
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("Environment setup complete!")


if __name__ == "__main__":
    main()
