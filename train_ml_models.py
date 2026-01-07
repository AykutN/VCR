#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML Model Training Script

This script trains Logistic Regression and SVM classifiers for deepfake detection.
It uses the same feature extraction system as the rule-based approach.

Usage:
    python train_ml_models.py --real-dir data/real --cloned-dir data/cloned --output models/
"""

import os
import numpy as np
import pickle
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import feature extraction functions from batch_test
from batch_test import (
    extract_all_features, 
    flatten_statistical_features,
    load_reference_samples
)
import librosa

def prepare_dataset(real_dir='data/real', cloned_dir='data/cloned'):
    """
    Prepare dataset from real and cloned audio files.
    
    Parameters:
    -----------
    real_dir : str
        Directory containing real audio samples
    cloned_dir : str
        Directory containing cloned audio samples
    
    Returns:
    --------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (0 = real, 1 = cloned)
    file_paths : list
        List of file paths corresponding to each sample
    """
    print("="*70)
    print("PREPARING DATASET")
    print("="*70)
    
    X = []
    y = []
    file_paths = []
    
    # Process real audio files (label = 0)
    print("\nProcessing REAL audio files...")
    speaker_dirs = [d for d in os.listdir(real_dir) 
                   if os.path.isdir(os.path.join(real_dir, d)) and not d.startswith('.')]
    
    real_count = 0
    for speaker_dir in speaker_dirs:
        speaker_path = os.path.join(real_dir, speaker_dir)
        wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
        
        for wav_file in wav_files:
            wav_path = os.path.join(speaker_path, wav_file)
            try:
                audio, sr = librosa.load(wav_path, sr=22050)
                features = extract_all_features(audio, sr)
                feature_vector = flatten_statistical_features(features)
                
                if len(feature_vector) > 0:
                    X.append(feature_vector)
                    y.append(0)  # Real = 0
                    file_paths.append(wav_path)
                    real_count += 1
                    if real_count % 5 == 0:
                        print(f"  Processed {real_count} real files...")
            except Exception as e:
                print(f"  Error processing {wav_path}: {e}")
                continue
    
    print(f"✓ Processed {real_count} real audio files")
    
    # Process cloned audio files (label = 1)
    print("\nProcessing CLONED audio files...")
    cloned_count = 0
    for speaker_dir in speaker_dirs:
        cloned_speaker_path = os.path.join(cloned_dir, speaker_dir)
        
        if not os.path.exists(cloned_speaker_path):
            continue
        
        wav_files = [f for f in os.listdir(cloned_speaker_path) if f.endswith('.wav')]
        
        for wav_file in wav_files:
            wav_path = os.path.join(cloned_speaker_path, wav_file)
            try:
                audio, sr = librosa.load(wav_path, sr=22050)
                features = extract_all_features(audio, sr)
                feature_vector = flatten_statistical_features(features)
                
                if len(feature_vector) > 0:
                    X.append(feature_vector)
                    y.append(1)  # Cloned = 1
                    file_paths.append(wav_path)
                    cloned_count += 1
                    if cloned_count % 5 == 0:
                        print(f"  Processed {cloned_count} cloned files...")
            except Exception as e:
                print(f"  Error processing {wav_path}: {e}")
                continue
    
    print(f"✓ Processed {cloned_count} cloned audio files")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset prepared:")
    print(f"  Total samples: {len(X)}")
    print(f"  Real samples: {np.sum(y == 0)}")
    print(f"  Cloned samples: {np.sum(y == 1)}")
    print(f"  Feature dimension: {X.shape[1]}")
    
    return X, y, file_paths

def train_models(X, y, file_paths=None, test_size=0.2, random_state=42):
    """
    Train Logistic Regression and SVM models.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
    
    Returns:
    --------
    models : dict
        Dictionary containing trained models and scaler
    results : dict
        Dictionary containing evaluation results
    """
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    # If file_paths provided, perform a group split by speaker to avoid leakage
    if file_paths is not None:
        # derive speaker group from file path parent directory
        groups = [os.path.basename(os.path.dirname(p)) for p in file_paths]
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        # Fallback to random stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # Train Logistic Regression
    print("\n" + "-"*70)
    print("Training Logistic Regression...")
    print("-"*70)
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced'  # Handle class imbalance
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # Evaluate Logistic Regression
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    lr_results = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'f1': f1_score(y_test, y_pred_lr),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist()
    }
    
    print(f"  Accuracy: {lr_results['accuracy']:.4f}")
    print(f"  Precision: {lr_results['precision']:.4f}")
    print(f"  Recall: {lr_results['recall']:.4f}")
    print(f"  F1-Score: {lr_results['f1']:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")
    
    models['logistic_regression'] = lr_model
    results['logistic_regression'] = lr_results
    
    # Train SVM
    print("\n" + "-"*70)
    print("Training SVM...")
    print("-"*70)
    svm_model = SVC(
        kernel='rbf',
        probability=True,
        random_state=random_state,
        class_weight='balanced'
    )
    svm_model.fit(X_train_scaled, y_train)
    
    # Evaluate SVM
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
    
    svm_results = {
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'precision': precision_score(y_test, y_pred_svm),
        'recall': recall_score(y_test, y_pred_svm),
        'f1': f1_score(y_test, y_pred_svm),
        'confusion_matrix': confusion_matrix(y_test, y_pred_svm).tolist()
    }
    
    print(f"  Accuracy: {svm_results['accuracy']:.4f}")
    print(f"  Precision: {svm_results['precision']:.4f}")
    print(f"  Recall: {svm_results['recall']:.4f}")
    print(f"  F1-Score: {svm_results['f1']:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_svm)}")
    
    models['svm'] = svm_model
    results['svm'] = svm_results
    models['scaler'] = scaler
    
    return models, results

def save_models(models, output_dir='models'):
    """
    Save trained models to disk.
    
    Parameters:
    -----------
    models : dict
        Dictionary containing trained models
    output_dir : str
        Output directory for saving models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Logistic Regression
    with open(os.path.join(output_dir, 'logistic_regression.pkl'), 'wb') as f:
        pickle.dump(models['logistic_regression'], f)
    
    # Save SVM
    with open(os.path.join(output_dir, 'svm.pkl'), 'wb') as f:
        pickle.dump(models['svm'], f)
    
    # Save Scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(models['scaler'], f)
    
    print(f"\n✓ Models saved to {output_dir}/")
    print(f"  - logistic_regression.pkl")
    print(f"  - svm.pkl")
    print(f"  - scaler.pkl")

def main():
    parser = argparse.ArgumentParser(description='Train ML models for deepfake detection')
    parser.add_argument('--real-dir', type=str, default='data/real',
                       help='Directory containing real audio samples')
    parser.add_argument('--cloned-dir', type=str, default='data/cloned',
                       help='Directory containing cloned audio samples')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for saved models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of test set (default: 0.2)')
    
    args = parser.parse_args()
    
    # Prepare dataset
    X, y, file_paths = prepare_dataset(
        real_dir=args.real_dir,
        cloned_dir=args.cloned_dir
    )
    
    if len(X) == 0:
        print("\n❌ Error: No samples found! Check your data directories.")
        return
    
    # Train models (pass file paths so we can group-split by speaker)
    models, results = train_models(X, y, file_paths=file_paths, test_size=args.test_size)
    
    # Save models
    save_models(models, output_dir=args.output)
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"\nLogistic Regression:")
    print(f"  Accuracy: {results['logistic_regression']['accuracy']:.2%}")
    print(f"  F1-Score: {results['logistic_regression']['f1']:.2%}")
    print(f"\nSVM:")
    print(f"  Accuracy: {results['svm']['accuracy']:.2%}")
    print(f"  F1-Score: {results['svm']['f1']:.2%}")
    
    print("\n" + "="*70)
    print("✓ Training completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()

