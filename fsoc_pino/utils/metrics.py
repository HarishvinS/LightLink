"""
Metrics utilities for FSOC-PINO benchmarking and evaluation.

This module provides functions to compute various metrics for comparing
PINO predictions against physics simulations or ground truth data.
"""

import numpy as np
from typing import List, Dict, Union, Optional
from skimage.metrics import structural_similarity as ssim
import logging

logger = logging.getLogger(__name__)


def l2_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Compute L2 (Euclidean) error between prediction and target.
    
    Args:
        prediction: Predicted field (complex or real)
        target: Target field (complex or real)
        
    Returns:
        L2 error normalized by target magnitude
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}")
    
    # Handle complex fields
    if np.iscomplexobj(prediction) or np.iscomplexobj(target):
        pred_mag = np.abs(prediction)
        target_mag = np.abs(target)
    else:
        pred_mag = prediction
        target_mag = target
    
    error = np.linalg.norm(pred_mag - target_mag)
    target_norm = np.linalg.norm(target_mag)
    
    return error / (target_norm + 1e-8)


def psnr(prediction: np.ndarray, target: np.ndarray, max_val: Optional[float] = None) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        prediction: Predicted field
        target: Target field
        max_val: Maximum possible value (if None, use target max)
        
    Returns:
        PSNR in dB
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}")
    
    # Handle complex fields by taking magnitude
    if np.iscomplexobj(prediction) or np.iscomplexobj(target):
        pred_mag = np.abs(prediction)
        target_mag = np.abs(target)
    else:
        pred_mag = prediction
        target_mag = target
    
    mse = np.mean((pred_mag - target_mag) ** 2)
    
    if mse == 0:
        return float('inf')
    
    if max_val is None:
        max_val = np.max(target_mag)
    
    return 20 * np.log10(max_val / np.sqrt(mse))


def ssim_metric(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        prediction: Predicted field
        target: Target field
        
    Returns:
        SSIM value between -1 and 1
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}")
    
    # Handle complex fields by taking magnitude
    if np.iscomplexobj(prediction) or np.iscomplexobj(target):
        pred_mag = np.abs(prediction)
        target_mag = np.abs(target)
    else:
        pred_mag = prediction
        target_mag = target
    
    # Normalize to [0, 1] range for SSIM
    pred_norm = (pred_mag - pred_mag.min()) / (pred_mag.max() - pred_mag.min() + 1e-8)
    target_norm = (target_mag - target_mag.min()) / (target_mag.max() - target_mag.min() + 1e-8)
    
    return ssim(target_norm, pred_norm, data_range=1.0)


def ber_error(pred_ber: float, target_ber: float) -> float:
    """
    Compute relative error in Bit Error Rate prediction.
    
    Args:
        pred_ber: Predicted BER
        target_ber: Target BER
        
    Returns:
        Relative error in BER
    """
    if target_ber == 0:
        return abs(pred_ber)
    
    return abs(pred_ber - target_ber) / target_ber


def scintillation_error(pred_si: float, target_si: float) -> float:
    """
    Compute relative error in scintillation index prediction.
    
    Args:
        pred_si: Predicted scintillation index
        target_si: Target scintillation index
        
    Returns:
        Relative error in scintillation index
    """
    if target_si == 0:
        return abs(pred_si)
    
    return abs(pred_si - target_si) / target_si


def compute_field_metrics(predictions: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute field-based metrics for a batch of predictions.
    
    Args:
        predictions: List of predicted fields
        targets: List of target fields
        
    Returns:
        Dictionary of computed metrics
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Number of predictions ({len(predictions)}) != number of targets ({len(targets)})")
    
    l2_errors = []
    psnr_values = []
    ssim_values = []
    
    for pred, target in zip(predictions, targets):
        l2_errors.append(l2_error(pred, target))
        psnr_values.append(psnr(pred, target))
        ssim_values.append(ssim_metric(pred, target))
    
    return {
        'l2_error_mean': np.mean(l2_errors),
        'l2_error_std': np.std(l2_errors),
        'psnr_mean': np.mean(psnr_values),
        'psnr_std': np.std(psnr_values),
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values)
    }


def compute_derived_metrics(pred_results: List[Dict], target_results: List[Dict]) -> Dict[str, float]:
    """
    Compute metrics for derived quantities (BER, scintillation index, etc.).
    
    Args:
        pred_results: List of prediction result dictionaries
        target_results: List of target result dictionaries
        
    Returns:
        Dictionary of computed metrics
    """
    ber_errors = []
    si_errors = []
    
    for pred, target in zip(pred_results, target_results):
        if 'bit_error_rate' in pred and 'bit_error_rate' in target:
            ber_errors.append(ber_error(pred['bit_error_rate'], target['bit_error_rate']))
        
        if 'scintillation_index' in pred and 'scintillation_index' in target:
            si_errors.append(scintillation_error(pred['scintillation_index'], target['scintillation_index']))
    
    metrics = {}
    
    if ber_errors:
        metrics.update({
            'ber_error_mean': np.mean(ber_errors),
            'ber_error_std': np.std(ber_errors)
        })
    
    if si_errors:
        metrics.update({
            'scintillation_error_mean': np.mean(si_errors),
            'scintillation_error_std': np.std(si_errors)
        })
    
    return metrics


def compute_benchmark_metrics(
    predictions: List[np.ndarray], 
    simulation_results: List[Dict], 
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Compute comprehensive benchmark metrics comparing PINO predictions to simulations.
    
    Args:
        predictions: List of PINO predictions (complex fields)
        simulation_results: List of simulation result dictionaries
        metrics: List of metric names to compute
        
    Returns:
        Dictionary of computed metrics
    """
    if metrics is None:
        metrics = ['l2_error', 'psnr', 'ssim']
    
    # Process PINO predictions to extract complex fields
    processed_predictions = []
    for pred in predictions:
        if pred.ndim == 4:  # (batch, channels, height, width)
            # Extract real and imaginary parts and combine into complex field
            real_part = pred[0, 0, :, :]  # First batch, first channel (real)
            imag_part = pred[0, 1, :, :]  # First batch, second channel (imaginary)
            complex_field = real_part + 1j * imag_part
            processed_predictions.append(complex_field)
        elif pred.ndim == 3:  # (channels, height, width)
            # Extract real and imaginary parts and combine into complex field
            real_part = pred[0, :, :]  # First channel (real)
            imag_part = pred[1, :, :]  # Second channel (imaginary)
            complex_field = real_part + 1j * imag_part
            processed_predictions.append(complex_field)
        else:
            # Assume it's already a complex field
            processed_predictions.append(pred)

    # Extract fields from simulation results
    target_fields = []
    for result in simulation_results:
        if hasattr(result, 'final_field'):
            target_fields.append(result.final_field)
        elif isinstance(result, dict) and 'final_field' in result:
            target_fields.append(result['final_field'])
        else:
            # Assume the result is the field itself
            target_fields.append(result)
    
    computed_metrics = {}
    
    # Compute field-based metrics
    if any(metric in ['l2_error', 'psnr', 'ssim'] for metric in metrics):
        field_metrics = compute_field_metrics(processed_predictions, target_fields)
        
        for metric in metrics:
            if metric == 'l2_error':
                computed_metrics['l2_error'] = field_metrics['l2_error_mean']
            elif metric == 'psnr':
                computed_metrics['psnr'] = field_metrics['psnr_mean']
            elif metric == 'ssim':
                computed_metrics['ssim'] = field_metrics['ssim_mean']
    
    # Compute derived metrics if available
    if any(metric in ['ber_error', 'scintillation_error'] for metric in metrics):
        # For this, we'd need to compute derived quantities from predictions
        # This is a simplified version - in practice, you'd compute BER and SI from the predicted fields
        logger.warning("Derived metrics (BER, scintillation) require additional computation from predicted fields")
    
    return computed_metrics
