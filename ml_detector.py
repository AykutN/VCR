#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML-based Deepfake Detection

This module provides ML model-based detection using trained Logistic Regression and SVM models.
"""

import os
import pickle
import numpy as np
import librosa
from batch_test import extract_all_features, flatten_statistical_features

def load_ml_models(models_dir='models'):
    """
    Load trained ML models from disk.
    
    Parameters:
    -----------
    models_dir : str
        Directory containing saved models
    
    Returns:
    --------
    models : dict
        Dictionary containing loaded models and scaler
    """
    models = {}
    
    # Load Logistic Regression
    lr_path = os.path.join(models_dir, 'logistic_regression.pkl')
    if os.path.exists(lr_path):
        with open(lr_path, 'rb') as f:
            models['logistic_regression'] = pickle.load(f)
    else:
        raise FileNotFoundError(f"Logistic Regression model not found at {lr_path}")
    
    # Load SVM
    svm_path = os.path.join(models_dir, 'svm.pkl')
    if os.path.exists(svm_path):
        with open(svm_path, 'rb') as f:
            models['svm'] = pickle.load(f)
    else:
        raise FileNotFoundError(f"SVM model not found at {svm_path}")
    
    # Load Scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            models['scaler'] = pickle.load(f)
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    return models

def detect_with_ml(audio_path, models_dir='models'):
    """
    Detect deepfake using ML models.
    
    Parameters:
    -----------
    audio_path : str
        Path to audio file to test
    models_dir : str
        Directory containing trained models
    
    Returns:
    --------
    result : dict
        Dictionary containing:
        - is_fake: bool, True if detected as fake
        - lr_score: float, Logistic Regression probability (0-1)
        - svm_score: float, SVM probability (0-1)
        - combined_score: float, Average of LR and SVM scores
        - confidence: float, Confidence in prediction
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=22050)
    except Exception as e:
        return {
            'is_fake': None,
            'lr_score': None,
            'svm_score': None,
            'combined_score': None,
            'confidence': 0.0,
            'error': str(e)
        }
    
    # Extract features
    features = extract_all_features(audio, sr)
    feature_vector = flatten_statistical_features(features)
    
    if len(feature_vector) == 0:
        return {
            'is_fake': None,
            'lr_score': None,
            'svm_score': None,
            'combined_score': None,
            'confidence': 0.0,
            'error': 'Failed to extract features'
        }
    
    # Load models
    try:
        models = load_ml_models(models_dir)
    except Exception as e:
        return {
            'is_fake': None,
            'lr_score': None,
            'svm_score': None,
            'combined_score': None,
            'confidence': 0.0,
            'error': f'Failed to load models: {str(e)}'
        }
    
    # Normalize features
    feature_vector_scaled = models['scaler'].transform(feature_vector.reshape(1, -1))
    
    # Get predictions
    lr_proba = models['logistic_regression'].predict_proba(feature_vector_scaled)[0]
    svm_proba = models['svm'].predict_proba(feature_vector_scaled)[0]
    
    # Probability of being fake (class 1)
    lr_score = lr_proba[1]
    svm_score = svm_proba[1]
    
    # Combined score (average)
    combined_score = (lr_score + svm_score) / 2.0
    
    # Decision (threshold = 0.5) - ensure native Python bool to avoid numpy.bool_ in JSON
    is_fake = bool(combined_score >= 0.5)
    
    # Confidence (distance from 0.5)
    confidence = abs(combined_score - 0.5) * 2
    confidence = min(1.0, confidence)
    
    return {
        'is_fake': is_fake,
        'lr_score': float(lr_score),
        'svm_score': float(svm_score),
        'combined_score': float(combined_score),
        'confidence': float(confidence),
        'lr_prediction': bool(models['logistic_regression'].predict(feature_vector_scaled)[0]),
        'svm_prediction': bool(models['svm'].predict(feature_vector_scaled)[0])
    }

