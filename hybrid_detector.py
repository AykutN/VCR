#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid Deepfake Detection System

Combines rule-based detection with ML-based detection for improved accuracy.
"""

from batch_test import detect_deepfake
from ml_detector import detect_with_ml

def detect_hybrid(audio_path, real_dir='data/real', models_dir='models',
                 rule_threshold=0.34, ml_weight=0.5, rule_weight=0.5):
    """
    Hybrid detection combining rule-based and ML-based methods.
    
    Parameters:
    -----------
    audio_path : str
        Path to audio file to test
    real_dir : str
        Directory containing real audio samples (for rule-based)
    models_dir : str
        Directory containing trained ML models
    rule_threshold : float
        Threshold for rule-based detection
    ml_weight : float
        Weight for ML score (0-1)
    rule_weight : float
        Weight for rule-based score (0-1)
        Note: ml_weight + rule_weight should equal 1.0
    
    Returns:
    --------
    result : dict
        Dictionary containing:
        - is_fake: bool, Final prediction
        - hybrid_score: float, Combined score (0-1)
        - rule_score: float, Rule-based score
        - ml_score: float, ML-based score
        - confidence: float, Confidence in prediction
        - method_details: dict, Detailed results from each method
    """
    # Normalize weights
    total_weight = ml_weight + rule_weight
    if total_weight > 0:
        ml_weight = ml_weight / total_weight
        rule_weight = rule_weight / total_weight
    else:
        ml_weight = 0.5
        rule_weight = 0.5
    
    # Rule-based detection
    rule_result = detect_deepfake(
        audio_path,
        real_dir=real_dir,
        threshold=rule_threshold
    )
    
    # ML-based detection
    ml_result = detect_with_ml(audio_path, models_dir=models_dir)
    
    # Extract scores
    if rule_result.get('score') is not None:
        rule_score = rule_result['score']
    else:
        rule_score = 0.5  # Default if rule-based fails
    
    if ml_result.get('combined_score') is not None:
        ml_score = ml_result['combined_score']
    else:
        ml_score = 0.5  # Default if ML fails
    
    # Combine scores
    hybrid_score = (rule_weight * rule_score) + (ml_weight * ml_score)
    hybrid_score = max(0.0, min(1.0, hybrid_score))
    
    # Final decision (threshold = 0.5)
    is_fake = hybrid_score >= 0.5
    
    # Confidence
    confidence = abs(hybrid_score - 0.5) * 2
    confidence = min(1.0, confidence)
    
    return {
        'is_fake': is_fake,
        'hybrid_score': float(hybrid_score),
        'rule_score': float(rule_score),
        'ml_score': float(ml_score),
        'confidence': float(confidence),
        'method_details': {
            'rule_based': rule_result,
            'ml_based': ml_result
        },
        'weights': {
            'rule_weight': rule_weight,
            'ml_weight': ml_weight
        }
    }

